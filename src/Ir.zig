//! libcrv intermediate representation.
//!
//! The IR is a *flattened tree*: instead of allocating expression nodes on the
//! heap and linking them with pointers, every node lives in one contiguous
//! array and refers to other nodes by index. This is the data-oriented layout
//! used by modern compilers (e.g. Zig's own AST/ZIR) and it buys us three
//! things the rest of the library needs:
//!
//!   * **Cache locality** — walking or evaluating constraints is a linear scan
//!     over packed arrays, not a pointer chase.
//!   * **Cheap content hashing** — the whole IR is plain POD arrays, so a hash
//!     is just Blake3 over their bytes (see `hash`). Two structurally identical
//!     constraint sets hash identically, which is what lets a solver cache
//!     results keyed on the query.
//!   * **Cheap serialization** — `serialize`/`deserialize` blit the arrays to
//!     and from a byte buffer behind a small versioned, checksummed header. No
//!     pointer fix-ups.
//!
//! Storage is split into a few side tables, each a flat array:
//!
//!   * `nodes`       — the expression/constraint tree (struct-of-arrays).
//!   * `vars`        — declared random/state variables.
//!   * `constraints` — named constraint blocks, each a slice of statement nodes.
//!   * `extra`       — `u32` payload pool for nodes/constraints with a variable
//!                     number of children (set members, dist items, ...).
//!
//! Indices into these tables are distinct `enum(u32)` types (`Node.Index`,
//! `Variable.Index`, ...) so the compiler catches an index used against the
//! wrong table. The `extra` pool, by contrast, stays a raw `[]u32`: it
//! interleaves node indices with plain counts, and a uniform width keeps that
//! pool — and the on-disk format — flat. Builders convert typed indices to
//! `u32` on the way in; accessors convert back out. Where a node edge may be
//! absent (e.g. a missing `else`) `Node.Index` carries a `null` sentinel
//! variant (`maxInt(u32)`).
//!
//! Variables and constraints carry an opaque `u32` id (`Variable.Id`,
//! `Constraint.Id`), not stored text: the IR holds no string bytes at all. An
//! id is only meaningful within the IR instance that produced it — resolve it
//! against whatever symbol table the builder keeps on the side. This keeps the
//! IR purely numeric, so hashing and caching never have to canonicalize strings.
//!
//! Scope: this is an *architectural* skeleton aimed at the SystemVerilog
//! constraint subset (scalar bit-vector randomization). It is intentionally
//! not complete — arrays/`foreach` are staked out as tags/roadmap but not yet
//! fleshed out.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

const Ir = @This();

/// Expression and constraint nodes, stored struct-of-arrays for locality.
nodes: std.MultiArrayList(Node) = .{},
/// Declared variables (random and state), referenced by `.var_ref` nodes.
vars: std.MultiArrayList(Variable) = .{},
/// Named constraint blocks; each owns a run of statement nodes in `extra`.
constraints: std.MultiArrayList(Constraint) = .{},
/// Side pool of `u32` payloads for variable-arity nodes and constraint bodies.
extra: std.ArrayListUnmanaged(u32) = .empty,

pub fn deinit(ir: *Ir, gpa: Allocator) void {
    ir.nodes.deinit(gpa);
    ir.vars.deinit(gpa);
    ir.constraints.deinit(gpa);
    ir.extra.deinit(gpa);
    ir.* = undefined;
}

// -- Nodes -------------------------------------------------------------------

/// One entry in the flattened tree. `tag` selects how `data` is interpreted;
/// see `Tag` for the per-tag encoding. Kept to `tag` + two `u32`s so the hot
/// path stays small and the whole thing is bit-blittable.
pub const Node = struct {
    tag: Tag,
    data: Data,

    /// Index into `Ir.nodes`. The `null` variant is a sentinel for an absent
    /// edge (e.g. a missing `else` branch).
    pub const Index = enum(u32) {
        null = std.math.maxInt(u32),
        _,
    };

    /// `extern` so the field order/layout is fixed for hashing and on-disk use.
    /// Fields are a raw payload whose meaning depends on `tag`.
    pub const Data = extern struct {
        lhs: u32 = 0,
        rhs: u32 = 0,
    };

    pub const Tag = enum(u8) {
        // -- Leaves --
        /// Typed integer literal. `lhs` = `extra` index of `[nwords, word0,
        /// word1, ...]`: a `u32` little-endian magnitude of `nwords` words
        /// (`ceil(width / 32)`, masked to the width), length-prefixed so a
        /// literal of any width is stored inline. `rhs` = packed `Type` (the
        /// literal's width).
        int_literal,
        /// Boolean literal (a 1-bit unsigned value). `lhs` is 0 or 1.
        bool_literal,
        /// Reference to a declared variable. `lhs` = `Variable.Index`; its type
        /// is the variable's type.
        var_ref,

        // -- Unary: `lhs` = operand node --
        /// Arithmetic negation (`-a`).
        neg,
        /// Bitwise complement (`~a`).
        bnot,
        /// Logical negation (`!a`).
        lnot,

        // -- Binary: `lhs`, `rhs` = operand nodes. Types are just widths, so
        //    operators that depend on signedness come in signed (`s`) and
        //    unsigned (`u`) forms. --
        add,
        sub,
        mul,
        sdiv,
        udiv,
        smod,
        umod,
        band,
        bor,
        bxor,
        /// Shift left (logical).
        sll,
        /// Shift right logical (zero-filling).
        srl,
        /// Shift right arithmetic (sign-extending).
        sra,
        eq,
        ne,
        slt,
        ult,
        sle,
        ule,
        sgt,
        ugt,
        sge,
        uge,
        land,
        lor,
        /// Implication (`a -> b`).
        implies,
        /// Equivalence (`a <-> b`).
        iff,

        // -- Sets, ranges, distributions --
        /// Inclusive range `[lo:hi]`. `lhs` = low node, `rhs` = high node.
        /// Appears as a member of `in`/`dist`, not as a boolean on its own.
        range,
        /// Set membership — SystemVerilog `value inside { ... }`. `lhs` = value
        /// node, `rhs` = `extra` index of `[count, member0, member1, ...]` where
        /// each member is a value node or a `range` node.
        in,
        /// Distribution `value dist { ... }`. `lhs` = value node,
        /// `rhs` = `extra` index of `[count, item0, ...]` of `dist_*` nodes.
        dist,
        /// `dist` item with `:=` weighting. `lhs` = value/`range` node,
        /// `rhs` = weight node.
        dist_weight_eq,
        /// `dist` item with `:/` weighting (weight split across the range).
        dist_weight_div,

        // -- Structural constraints --
        /// `if (cond) then else else`. `lhs` = cond node,
        /// `rhs` = `extra` index of `[then_node, else_node]`; `else_node` is a
        /// `Node.Index` whose `.null` variant means there is no `else`.
        if_else,
        /// `unique { ... }`. `lhs` = `extra` index of `[count, node0, ...]`.
        unique,
        /// `solve a, b before c, d`. `lhs` = `extra` index of
        /// `[before_count, before..., after_count, after...]`, each entry a
        /// `Variable.Index`. An ordering hint, not a boolean.
        solve_before,
        /// `foreach (arr[i]) body` — array iteration. `lhs` = array
        /// `Variable.Index`, `rhs` = `extra` index of
        /// `[iter_var, body_count, body...]`. Roadmap: arrays are not modeled
        /// past this stub.
        foreach,

        // -- Sizing casts: `lhs` = operand node, `rhs` = target bit width.
        //    Widths only ever change through these; all other operators keep
        //    their operands' width. --
        /// Zero-extend to a wider, unsigned type.
        zext,
        /// Sign-extend to a wider, signed type.
        sext,
        /// Truncate to a narrower type, keeping the low bits.
        trunc,
    };
};

// -- Variables ---------------------------------------------------------------

/// A declared variable. `Kind` distinguishes solver-visible randoms from fixed
/// state read by constraints.
pub const Variable = struct {
    /// Opaque caller-assigned id (see `Id`).
    id: Id,
    /// Bit-vector width. Signedness is not part of the type — it lives on the
    /// operators (`slt`/`ult`, `sdiv`/`udiv`, `sext`/`zext`, ...).
    ty: Type,
    kind: Kind,

    /// Index into `Ir.vars`.
    pub const Index = enum(u32) { _ };

    /// Opaque per-IR identity handle the caller assigns; resolve it against the
    /// builder's own symbol table. The IR stores no text.
    pub const Id = enum(u32) { _ };

    pub const Kind = enum(u8) {
        /// Fixed input the solver may read but not assign.
        state,
        /// Randomized each solve.
        rand,
        /// Randomized cyclically (`randc`): every value before repeats.
        randc,
    };
};

/// A value's type is just its bit-vector width — signedness lives on the
/// operators. A `packed struct(u32)` so it is one word and bit-blittable (it
/// packs into a literal/cast node's `data`).
pub const Type = packed struct(u32) {
    /// Bit width (`1..=65535`). `0` is reserved for "unspecified/parameterized".
    width: u16,
    _reserved: u16 = 0,

    /// A vector of the given width.
    pub fn bit(width: u16) Type {
        return .{ .width = width };
    }
};

// -- Constraints -------------------------------------------------------------

/// A named constraint block: a sequence of boolean statement nodes that must
/// all hold. `body` points at a run of `Node.Index` values in `extra`.
pub const Constraint = struct {
    /// Opaque caller-assigned id (see `Id`).
    id: Id,
    flags: Flags,
    body: Extra.Slice,

    /// Index into `Ir.constraints`.
    pub const Index = enum(u32) { _ };

    /// Opaque per-IR identity handle the caller assigns; resolve it against the
    /// builder's own symbol table. The IR stores no text.
    pub const Id = enum(u32) { _ };

    pub const Flags = packed struct(u8) {
        /// `soft` constraint: may be dropped if it conflicts with a hard one.
        soft: bool = false,
        _reserved: u7 = 0,
    };
};

// -- Index helpers -----------------------------------------------------------

pub const Extra = struct {
    /// Element offset into `Ir.extra`.
    pub const Index = enum(u32) { _ };

    /// A `[start, len)` window into `extra`. `extern` for stable on-disk layout.
    pub const Slice = extern struct {
        start: Extra.Index,
        len: u32,
    };
};

// -- Construction ------------------------------------------------------------
//
// These are the low-level builders the query front-end will sit on top of.
// They append to the flat arrays and hand back typed indices; nothing here
// inspects or validates the tree — that is the job of later passes.

pub fn addNode(ir: *Ir, gpa: Allocator, node: Node) Allocator.Error!Node.Index {
    const index: u32 = @intCast(ir.nodes.len);
    try ir.nodes.append(gpa, node);
    return @enumFromInt(index);
}

/// Append `items` to the `extra` pool and return the start offset.
pub fn addExtra(ir: *Ir, gpa: Allocator, items: []const u32) Allocator.Error!Extra.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.appendSlice(gpa, items);
    return @enumFromInt(start);
}

pub fn addVariable(ir: *Ir, gpa: Allocator, v: Variable) Allocator.Error!Variable.Index {
    const index: u32 = @intCast(ir.vars.len);
    try ir.vars.append(gpa, v);
    return @enumFromInt(index);
}

/// Add a constraint block whose body is the given statement nodes. The node
/// indices are copied into `extra`.
pub fn addConstraint(
    ir: *Ir,
    gpa: Allocator,
    id: Constraint.Id,
    flags: Constraint.Flags,
    stmts: []const Node.Index,
) Allocator.Error!Constraint.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.ensureUnusedCapacity(gpa, stmts.len);
    for (stmts) |s| ir.extra.appendAssumeCapacity(@intFromEnum(s));
    const index: u32 = @intCast(ir.constraints.len);
    try ir.constraints.append(gpa, .{
        .id = id,
        .flags = flags,
        .body = .{ .start = @enumFromInt(start), .len = @intCast(stmts.len) },
    });
    return @enumFromInt(index);
}

// -- Convenience node constructors -------------------------------------------

/// A typed integer literal from a 64-bit value, taken modulo `ty.width` bits.
/// For wider constants (`ty.width > 64`), use `constBig`.
pub fn constInt(ir: *Ir, gpa: Allocator, value: u64, ty: Type) Allocator.Error!Node.Index {
    return ir.appendLiteral(gpa, ty, struct {
        v: u64,
        fn word(ctx: @This(), j: u32) u32 {
            const bit = j * 32;
            return if (bit >= 64) 0 else @truncate(ctx.v >> @intCast(bit));
        }
    }{ .v = value });
}

/// A typed integer literal from an arbitrary-precision magnitude, taken modulo
/// `ty.width` bits. `value` is treated as an unsigned bit pattern — its sign is
/// ignored, so to store a negative constant pass the intended two's-complement
/// pattern (or `sext` a narrower literal).
pub fn constBig(ir: *Ir, gpa: Allocator, value: std.math.big.int.Const, ty: Type) Allocator.Error!Node.Index {
    return ir.appendLiteral(gpa, ty, struct {
        v: std.math.big.int.Const,
        fn word(ctx: @This(), j: u32) u32 {
            const bits_per_limb = @bitSizeOf(std.math.big.Limb);
            const limb = (@as(usize, j) * 32) / bits_per_limb;
            if (limb >= ctx.v.limbs.len) return 0;
            const shift = (@as(usize, j) * 32) % bits_per_limb;
            return @truncate(ctx.v.limbs[limb] >> @intCast(shift));
        }
    }{ .v = value });
}

/// Number of `u32` words a `width`-bit literal magnitude occupies (`>= 1`).
fn litWords(width: u16) u32 {
    return @max(1, (@as(u32, width) + 31) / 32);
}

/// Mask for the most-significant `u32` word of a `width`-bit magnitude.
fn litTopMask(width: u16) u32 {
    const bits = @as(u32, width) - (litWords(width) - 1) * 32; // 1..=32
    return if (bits >= 32) ~@as(u32, 0) else (@as(u32, 1) << @intCast(bits)) - 1;
}

/// Encode an `int_literal`: `[nwords, word0, ...]`, with `src.word(j)` supplying
/// the little-endian `u32` words and the top word masked to `ty.width`.
fn appendLiteral(ir: *Ir, gpa: Allocator, ty: Type, src: anytype) Allocator.Error!Node.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    const n = litWords(ty.width);
    try ir.extra.ensureUnusedCapacity(gpa, 1 + n);
    ir.extra.appendAssumeCapacity(n);
    var j: u32 = 0;
    while (j < n) : (j += 1) {
        var w = src.word(j);
        if (j == n - 1) w &= litTopMask(ty.width);
        ir.extra.appendAssumeCapacity(w);
    }
    return ir.addNode(gpa, .{ .tag = .int_literal, .data = .{ .lhs = start, .rhs = @bitCast(ty) } });
}

pub fn boolLit(ir: *Ir, gpa: Allocator, value: bool) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .bool_literal, .data = .{ .lhs = @intFromBool(value) } });
}

/// Zero-extend `operand` to a wider unsigned type of `width` bits.
pub fn zext(ir: *Ir, gpa: Allocator, operand: Node.Index, width: u16) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .zext, .data = .{ .lhs = @intFromEnum(operand), .rhs = width } });
}

/// Sign-extend `operand` to a wider signed type of `width` bits.
pub fn sext(ir: *Ir, gpa: Allocator, operand: Node.Index, width: u16) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .sext, .data = .{ .lhs = @intFromEnum(operand), .rhs = width } });
}

/// Truncate `operand` to a narrower type of `width` bits (keeps the low bits).
pub fn trunc(ir: *Ir, gpa: Allocator, operand: Node.Index, width: u16) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .trunc, .data = .{ .lhs = @intFromEnum(operand), .rhs = width } });
}

pub fn varRef(ir: *Ir, gpa: Allocator, v: Variable.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .var_ref, .data = .{ .lhs = @intFromEnum(v) } });
}

pub fn unary(ir: *Ir, gpa: Allocator, tag: Node.Tag, operand: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(operand) } });
}

pub fn binary(ir: *Ir, gpa: Allocator, tag: Node.Tag, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

/// Inclusive range `[lo:hi]`, for use as an `in`/`dist` member.
pub fn range(ir: *Ir, gpa: Allocator, lo: Node.Index, hi: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .range, .data = .{ .lhs = @intFromEnum(lo), .rhs = @intFromEnum(hi) } });
}

/// Set membership — SystemVerilog `value inside { members... }`.
pub fn in(ir: *Ir, gpa: Allocator, value: Node.Index, members: []const Node.Index) Allocator.Error!Node.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.ensureUnusedCapacity(gpa, members.len + 1);
    ir.extra.appendAssumeCapacity(@intCast(members.len));
    for (members) |m| ir.extra.appendAssumeCapacity(@intFromEnum(m));
    return ir.addNode(gpa, .{ .tag = .in, .data = .{ .lhs = @intFromEnum(value), .rhs = start } });
}

// -- Accessors ---------------------------------------------------------------

/// The low 64 bits of an `int_literal` node's magnitude. For a literal wider
/// than 64 bits this drops the high bits; read `extra` directly (or reconstruct
/// a big int) when the full magnitude is needed.
pub fn intValue(ir: *const Ir, node: Node.Index) u64 {
    const d = ir.nodes.items(.data)[@intFromEnum(node)];
    const n = ir.extra.items[d.lhs];
    const lo = ir.extra.items[d.lhs + 1];
    const hi = if (n >= 2) ir.extra.items[d.lhs + 2] else 0;
    return @as(u64, lo) | (@as(u64, hi) << 32);
}

/// The declared type of an `int_literal` node.
pub fn literalType(ir: *const Ir, node: Node.Index) Type {
    return @bitCast(ir.nodes.items(.data)[@intFromEnum(node)].rhs);
}

/// The target width of a `zext`/`sext`/`trunc` node.
pub fn castWidth(ir: *const Ir, node: Node.Index) u16 {
    return @intCast(ir.nodes.items(.data)[@intFromEnum(node)].rhs);
}

/// Width assumed for a variable declared without an explicit one.
pub const default_width: u16 = 32;

/// Recursively resolve a node's type — its bit-vector width. Leaves, literals,
/// and casts are explicitly typed; every other operator propagates its left
/// operand's width, and comparisons/logical operators yield a 1-bit `bool`.
/// Widths only ever change through the `zext`/`sext`/`trunc` casts.
pub fn typeOf(ir: *const Ir, node: Node.Index) Type {
    const i = @intFromEnum(node);
    const d = ir.nodes.items(.data)[i];
    return switch (ir.nodes.items(.tag)[i]) {
        .int_literal => ir.literalType(node),
        .bool_literal => .{ .width = 1 },
        .var_ref => varType(ir.vars.items(.ty)[d.lhs]),

        .zext, .sext, .trunc => .{ .width = ir.castWidth(node) },

        .neg, .bnot => ir.typeOf(@enumFromInt(d.lhs)),
        .add, .sub, .mul, .sdiv, .udiv, .smod, .umod, .band, .bor, .bxor, .sll, .srl, .sra => ir.typeOf(@enumFromInt(d.lhs)),
        .range => ir.typeOf(@enumFromInt(d.lhs)),

        .eq, .ne, .slt, .ult, .sle, .ule, .sgt, .ugt, .sge, .uge, .lnot, .land, .lor, .implies, .iff, .in => .{ .width = 1 },
        .dist, .dist_weight_eq, .dist_weight_div, .if_else, .unique, .solve_before, .foreach => .{ .width = 1 },
    };
}

fn varType(t: Type) Type {
    return if (t.width == 0) .{ .width = default_width } else t;
}

/// The statement nodes making up a constraint block's body.
pub fn constraintBody(ir: *const Ir, index: Constraint.Index) []const Node.Index {
    const body = ir.constraints.items(.body)[@intFromEnum(index)];
    const raw = ir.extra.items[@intFromEnum(body.start)..][0..body.len];
    return @ptrCast(raw);
}

// -- Hashing -----------------------------------------------------------------

/// A Blake3 content digest — the full 256-bit output. Its collision probability
/// is astronomically small, so callers may treat the digest as collision-free.
pub const Digest = [Blake3.digest_length]u8;

/// Content hash over every flat array. Structurally identical IRs hash equal,
/// so a solver can key its result cache on this. Section lengths are folded in
/// so that shifting a boundary between two arrays cannot collide. Blake3 makes
/// the digest collision-resistant, so it doubles as a fingerprint safe to name
/// cache entries by.
pub fn hash(ir: *const Ir) Digest {
    var st = Blake3.init(.{});
    inline for (.{
        std.mem.sliceAsBytes(ir.nodes.items(.tag)),
        std.mem.sliceAsBytes(ir.nodes.items(.data)),
        std.mem.sliceAsBytes(ir.vars.items(.id)),
        std.mem.sliceAsBytes(ir.vars.items(.ty)),
        std.mem.sliceAsBytes(ir.vars.items(.kind)),
        std.mem.sliceAsBytes(ir.constraints.items(.id)),
        std.mem.sliceAsBytes(ir.constraints.items(.flags)),
        std.mem.sliceAsBytes(ir.constraints.items(.body)),
        std.mem.sliceAsBytes(ir.extra.items),
    }) |section| {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(section.len), .little);
        st.update(&len);
        st.update(section);
    }
    var out: Digest = undefined;
    st.final(&out);
    return out;
}

// -- Serialization -----------------------------------------------------------
//
// On-disk form:
//
//     [magic: 4 bytes]
//     [format_version: u32]
//     [ per section: u32 element count, then the section's raw little-endian
//       bytes ]
//     [checksum: Blake3 digest over everything above]
//
// Because every section is a paddingless array of POD, there are no pointers to
// relocate; the reader resizes and blits straight back in. The trailing Blake3
// checksum detects corruption or tampering, and it is verified *before* any
// length is trusted, so a bad cache file can never drive an allocation off a
// garbage count. Endianness is not converted, so a cache file is portable
// across machines of the same endianness (fine for a local build cache;
// documented here as a known limitation).

/// Identifies a libcrv cache blob.
pub const magic: [4]u8 = .{ 'C', 'R', 'V', 'B' };
/// On-disk format version. Bump on any layout change; readers reject mismatches.
pub const format_version: u32 = 3;

const checksum_len = Blake3.digest_length;
/// Bytes preceding the first section: `magic` + `format_version`.
const header_len = magic.len + @sizeOf(u32);

/// Serialize into a freshly allocated byte buffer owned by the caller.
pub fn serialize(ir: *const Ir, gpa: Allocator) Allocator.Error![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(gpa);

    try out.appendSlice(gpa, &magic);
    try writeU32(&out, gpa, format_version);

    try writeU32(&out, gpa, @intCast(ir.nodes.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.nodes.items(.tag)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.nodes.items(.data)));

    try writeU32(&out, gpa, @intCast(ir.vars.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.vars.items(.id)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.vars.items(.ty)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.vars.items(.kind)));

    try writeU32(&out, gpa, @intCast(ir.constraints.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.constraints.items(.id)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.constraints.items(.flags)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.constraints.items(.body)));

    try writeU32(&out, gpa, @intCast(ir.extra.items.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.extra.items));

    // Checksum over the whole payload, appended last.
    var digest: Digest = undefined;
    Blake3.hash(out.items, &digest, .{});
    try out.appendSlice(gpa, &digest);

    return out.toOwnedSlice(gpa);
}

pub const DeserializeError = error{
    /// Buffer does not start with `magic`.
    BadMagic,
    /// `format_version` is not one this build understands.
    UnsupportedVersion,
    /// The trailing checksum does not match the payload.
    ChecksumMismatch,
    /// Buffer ends mid-record.
    Truncated,
} || Allocator.Error;

/// Reconstruct an `Ir` from bytes produced by `serialize`. The result owns its
/// storage and must be `deinit`ed.
pub fn deserialize(gpa: Allocator, bytes: []const u8) DeserializeError!Ir {
    if (bytes.len < header_len + checksum_len) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..magic.len], &magic)) return error.BadMagic;
    if (std.mem.readInt(u32, bytes[4..8], .little) != format_version) return error.UnsupportedVersion;

    const payload = bytes[0 .. bytes.len - checksum_len];
    const stored = bytes[bytes.len - checksum_len ..];
    var digest: Digest = undefined;
    Blake3.hash(payload, &digest, .{});
    // Integrity check only (local cache); a constant-time compare is unneeded.
    if (!std.mem.eql(u8, &digest, stored)) return error.ChecksumMismatch;

    var ir: Ir = .{};
    errdefer ir.deinit(gpa);

    // Counts are trusted only now, after the checksum has been verified.
    var cur: Cursor = .{ .bytes = payload, .pos = header_len };

    try ir.nodes.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.nodes.items(.tag)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.nodes.items(.data)));

    try ir.vars.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.vars.items(.id)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.vars.items(.ty)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.vars.items(.kind)));

    try ir.constraints.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.constraints.items(.id)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.constraints.items(.flags)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.constraints.items(.body)));

    try ir.extra.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.extra.items));

    return ir;
}

fn writeU32(out: *std.ArrayListUnmanaged(u8), gpa: Allocator, v: u32) Allocator.Error!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try out.appendSlice(gpa, &buf);
}

const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(c: *Cursor, n: usize) error{Truncated}![]const u8 {
        if (c.pos + n > c.bytes.len) return error.Truncated;
        defer c.pos += n;
        return c.bytes[c.pos..][0..n];
    }

    fn readU32(c: *Cursor) error{Truncated}!u32 {
        const s = try c.take(4);
        return std.mem.readInt(u32, s[0..4], .little);
    }

    /// Copy the next `dst.len` bytes into `dst`.
    fn copyInto(c: *Cursor, dst: []u8) error{Truncated}!void {
        @memcpy(dst, try c.take(dst.len));
    }
};

// -- Tests -------------------------------------------------------------------

test "build, hash, and round-trip a small constraint set" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // Ids are caller-owned; here they stand in for a symbol table.
    const x_id: Variable.Id = @enumFromInt(1);
    const c_id: Constraint.Id = @enumFromInt(2);

    // rand bit [3:0] x;  constraint c { x inside {[0:15]}; }
    const x = try ir.addVariable(gpa, .{ .id = x_id, .ty = Type.bit(4), .kind = .rand });

    const x_ref = try ir.varRef(gpa, x);
    const lo = try ir.constInt(gpa, 0, Type.bit(4));
    const hi = try ir.constInt(gpa, 15, Type.bit(4));
    const zero_to_15 = try ir.range(gpa, lo, hi);
    const membership = try ir.in(gpa, x_ref, &.{zero_to_15});

    _ = try ir.addConstraint(gpa, c_id, .{}, &.{membership});

    try std.testing.expectEqual(x_id, ir.vars.items(.id)[@intFromEnum(x)]);
    try std.testing.expectEqual(@as(u64, 15), ir.intValue(hi));

    // Serialize -> deserialize preserves structure and content hash.
    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);

    var ir2 = try Ir.deserialize(gpa, bytes);
    defer ir2.deinit(gpa);

    try std.testing.expectEqual(ir.nodes.len, ir2.nodes.len);
    try std.testing.expectEqual(ir.vars.len, ir2.vars.len);
    try std.testing.expectEqual(ir.constraints.len, ir2.constraints.len);

    const h1 = ir.hash();
    const h2 = ir2.hash();
    try std.testing.expectEqualSlices(u8, &h1, &h2);

    // Opaque ids round-trip verbatim.
    try std.testing.expectEqual(x_id, ir2.vars.items(.id)[0]);
    try std.testing.expectEqual(c_id, ir2.constraints.items(.id)[0]);

    const body = ir2.constraintBody(@enumFromInt(0));
    try std.testing.expectEqual(@as(usize, 1), body.len);
    try std.testing.expectEqual(Node.Tag.in, ir2.nodes.items(.tag)[@intFromEnum(body[0])]);
}

test "cache format is versioned and checksum-protected" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
    defer ir.deinit(gpa);
    _ = try ir.constInt(gpa, 42, Type.bit(8));

    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);

    // Header carries the magic and a 32-bit format version.
    try std.testing.expectEqualSlices(u8, &magic, bytes[0..magic.len]);
    try std.testing.expectEqual(format_version, std.mem.readInt(u32, bytes[4..8], .little));

    // A short buffer is rejected before anything is parsed.
    try std.testing.expectError(error.Truncated, Ir.deserialize(gpa, bytes[0..8]));

    // Flipping a payload byte trips the checksum.
    const tampered = try gpa.dupe(u8, bytes);
    defer gpa.free(tampered);
    tampered[header_len + 2] ^= 0xff;
    try std.testing.expectError(error.ChecksumMismatch, Ir.deserialize(gpa, tampered));

    // Wrong magic.
    var bad_magic = [_]u8{0} ** (header_len + checksum_len);
    bad_magic[0] = 'X';
    try std.testing.expectError(error.BadMagic, Ir.deserialize(gpa, &bad_magic));

    // Recognized magic but an unknown version.
    var bad_version = [_]u8{0} ** (header_len + checksum_len);
    @memcpy(bad_version[0..magic.len], &magic);
    std.mem.writeInt(u32, bad_version[4..8], format_version + 1, .little);
    try std.testing.expectError(error.UnsupportedVersion, Ir.deserialize(gpa, &bad_version));
}

test "typeOf resolves recursively" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const u = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const sum = try ir.binary(gpa, .add, try ir.varRef(gpa, u), try ir.constInt(gpa, 3, Type.bit(8)));
    const wide = try ir.zext(gpa, sum, 16); // 16-bit
    const back = try ir.trunc(gpa, wide, 8); // 8-bit
    const cmp = try ir.binary(gpa, .ult, try ir.varRef(gpa, u), sum); // 1-bit bool

    try std.testing.expectEqual(Type.bit(8), ir.typeOf(sum));
    try std.testing.expectEqual(@as(u16, 16), ir.typeOf(wide).width);
    try std.testing.expectEqual(@as(u16, 8), ir.typeOf(back).width);
    try std.testing.expectEqual(@as(u16, 1), ir.typeOf(cmp).width);
}

test "wide integer literal round-trips" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // A 128-bit constant with bits set above 64: (1 << 100) | 0xdead_beef.
    var limbs = [_]std.math.big.Limb{ 0xdead_beef, 1 << 36 };
    const value: std.math.big.int.Const = .{ .limbs = &limbs, .positive = true };
    const lit = try ir.constBig(gpa, value, Type.bit(128));

    try std.testing.expectEqual(@as(u16, 128), ir.typeOf(lit).width);
    // `intValue` exposes the low 64 bits.
    try std.testing.expectEqual(@as(u64, 0xdead_beef), ir.intValue(lit));
    // The high bits survive: serialize/deserialize preserves the content hash.
    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);
    var ir2 = try Ir.deserialize(gpa, bytes);
    defer ir2.deinit(gpa);
    try std.testing.expectEqual(ir.hash(), ir2.hash());

    // A too-wide value is taken modulo the width: bit 128 is dropped at bit(128).
    var over = [_]std.math.big.Limb{ 1, 0, 1 };
    const truncated = try ir.constBig(gpa, .{ .limbs = &over, .positive = true }, Type.bit(128));
    try std.testing.expectEqual(@as(u64, 1), ir.intValue(truncated));
}

test {
    std.testing.refAllDecls(@This());
}

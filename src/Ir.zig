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

    /// What a node is. The numeric values are deliberate and load-bearing in
    /// two places: they are the on-disk encoding (so changing one is a
    /// `format_version` bump), and `crv.h` declares `crv_op` and `crv_cast`
    /// with exactly these values, so the C bindings decode an operator by
    /// checking which `Class` it lands in rather than by translating it
    /// through a table. Tags are numbered in gapped groups to leave each class
    /// room to grow; `0` is left unused, so a zeroed byte is never a valid tag.
    pub const Tag = enum(u8) {
        // -- Leaves: 1..15 --
        /// Typed integer literal. `lhs` = `extra` index of `[nwords, word0,
        /// word1, ...]`: a `u32` little-endian magnitude of `nwords` words
        /// (`ceil(width / 32)`, masked to the width), length-prefixed so a
        /// literal of any width is stored inline. `rhs` = packed `Type` (the
        /// literal's width).
        int_literal = 1,
        /// Boolean literal (a 1-bit unsigned value). `lhs` is 0 or 1.
        bool_literal = 2,
        /// Reference to a declared variable. `lhs` = `Variable.Index`; its type
        /// is the variable's type.
        var_ref = 3,

        // -- Unary: 16..31. `lhs` = operand node --
        /// Arithmetic negation (`-a`).
        neg = 16,
        /// Bitwise complement (`~a`).
        bnot = 17,
        /// Logical negation (`!a`).
        lnot = 18,

        // -- Binary, result width = operand width: 32..63. `lhs`, `rhs` =
        //    operand nodes. Types are just widths, so operators that depend on
        //    signedness come in signed (`s`) and unsigned (`u`) forms. --
        add = 32,
        sub = 33,
        mul = 34,
        sdiv = 35,
        udiv = 36,
        smod = 37,
        umod = 38,
        band = 39,
        bor = 40,
        bxor = 41,
        /// Shift left (logical).
        sll = 42,
        /// Shift right logical (zero-filling).
        srl = 43,
        /// Shift right arithmetic (sign-extending).
        sra = 44,

        // -- Binary, 1-bit result: 64..95 --
        eq = 64,
        ne = 65,
        slt = 66,
        ult = 67,
        sle = 68,
        ule = 69,
        sgt = 70,
        ugt = 71,
        sge = 72,
        uge = 73,
        land = 74,
        lor = 75,
        /// Implication (`a -> b`).
        implies = 76,
        /// Equivalence (`a <-> b`).
        iff = 77,

        // -- Sizing casts: 96..111. `lhs` = operand node, `rhs` = target bit
        //    width. Widths only ever change through these; all other operators
        //    keep their operands' width. --
        /// Zero-extend to a wider, unsigned type.
        zext = 96,
        /// Sign-extend to a wider, signed type.
        sext = 97,
        /// Truncate to a narrower type, keeping the low bits.
        trunc = 98,

        // -- Sets, ranges, distributions: 112..127 --
        /// Inclusive range `[lo:hi]`. `lhs` = low node, `rhs` = high node.
        /// Appears as a member of `in`/`dist`, not as a boolean on its own.
        range = 112,
        /// Set membership — SystemVerilog `value inside { ... }`. `lhs` = value
        /// node, `rhs` = `extra` index of `[count, member0, member1, ...]` where
        /// each member is a value node or a `range` node.
        in = 113,
        /// Distribution `value dist { ... }`. `lhs` = value node,
        /// `rhs` = `extra` index of `[count, item0, ...]` of `dist_*` nodes.
        dist = 114,
        /// `dist` item with `:=` weighting. `lhs` = value/`range` node,
        /// `rhs` = weight node.
        dist_weight_eq = 115,
        /// `dist` item with `:/` weighting (weight split across the range).
        dist_weight_div = 116,

        // -- Structural constraints: 128..143 --
        /// `if (cond) then else else`. `lhs` = cond node,
        /// `rhs` = `extra` index of `[then_node, else_node]`; `else_node` is a
        /// `Node.Index` whose `.null` variant means there is no `else`.
        if_else = 128,
        /// `unique { ... }`. `lhs` = `extra` index of `[count, node0, ...]`.
        unique = 129,
        /// `solve a, b before c, d`. `lhs` = `extra` index of
        /// `[before_count, before..., after_count, after...]`, each entry a
        /// `Variable.Index`. An ordering hint, not a boolean.
        solve_before = 130,
        /// `foreach (arr[i]) body` — array iteration. `lhs` = array
        /// `Variable.Index`, `rhs` = `extra` index of
        /// `[iter_var, body_count, body...]`. Roadmap: arrays are not modeled
        /// past this stub.
        foreach = 131,

        /// The family a tag belongs to. The numbering above groups tags this
        /// way already, but `class` is the authority — it is exhaustive, so a
        /// new tag cannot be added without placing itself.
        pub const Class = enum { leaf, unary, binary, cast, set, structural };

        /// Which entry point builds this tag. The C bindings use it to accept
        /// only the operators a given export is documented to take: a binary
        /// op handed to `crv_node_unary` is rejected here.
        pub fn class(tag: Tag) Class {
            return switch (tag) {
                .int_literal, .bool_literal, .var_ref => .leaf,
                .neg, .bnot, .lnot => .unary,
                .add,
                .sub,
                .mul,
                .sdiv,
                .udiv,
                .smod,
                .umod,
                .band,
                .bor,
                .bxor,
                .sll,
                .srl,
                .sra,
                .eq,
                .ne,
                .slt,
                .ult,
                .sle,
                .ule,
                .sgt,
                .ugt,
                .sge,
                .uge,
                .land,
                .lor,
                .implies,
                .iff,
                => .binary,
                .zext, .sext, .trunc => .cast,
                .range, .in, .dist, .dist_weight_eq, .dist_weight_div => .set,
                .if_else, .unique, .solve_before, .foreach => .structural,
            };
        }
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

    /// Like `Node.Tag`, the values are the on-disk encoding and are what
    /// `crv.h` declares `crv_var_kind` with.
    pub const Kind = enum(u8) {
        /// Fixed input the solver may read but not assign.
        state = 0,
        /// Randomized each solve.
        rand = 1,
        /// Randomized cyclically (`randc`): every value before repeats.
        randc = 2,
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

/// A typed integer literal from a little-endian array of 64-bit words, taken
/// modulo `ty.width` bits. Like `constBig` this is an unsigned bit pattern, but
/// it is expressed in fixed 64-bit words rather than `std.math.big` limbs, so
/// the encoding does not depend on the host's limb width — which is what the C
/// ABI is defined in terms of.
pub fn constBits(ir: *Ir, gpa: Allocator, words: []const u64, ty: Type) Allocator.Error!Node.Index {
    return ir.appendLiteral(gpa, ty, struct {
        w: []const u64,
        fn word(ctx: @This(), j: u32) u32 {
            const i = j / 2;
            if (i >= ctx.w.len) return 0;
            return @truncate(ctx.w[i] >> @intCast(32 * (j % 2)));
        }
    }{ .w = words });
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

/// Set membership — SystemVerilog `value inside { members... }`. A member is a
/// value node or a `range` node.
pub fn in(ir: *Ir, gpa: Allocator, value: Node.Index, members: []const Node.Index) Allocator.Error!Node.Index {
    const start = try ir.addCounted(gpa, Node.Index, members);
    return ir.addNode(gpa, .{ .tag = .in, .data = .{ .lhs = @intFromEnum(value), .rhs = start } });
}

/// Which weighting a `dist` item uses. Not stored — `distItem` folds it into
/// the node's tag — but the values are what `crv.h` declares `crv_dist_kind`
/// with, so the C binding decodes one instead of translating it.
pub const DistKind = enum(u8) { eq = 0, div = 1 };

/// One weighted item of a `dist`: `value := weight` (`.eq`) or `value :/ weight`
/// (`.div`, the weight split across the range). `value` is a value node or a
/// `range` node.
pub fn distItem(
    ir: *Ir,
    gpa: Allocator,
    kind: DistKind,
    value: Node.Index,
    weight: Node.Index,
) Allocator.Error!Node.Index {
    const tag: Node.Tag = switch (kind) {
        .eq => .dist_weight_eq,
        .div => .dist_weight_div,
    };
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(value), .rhs = @intFromEnum(weight) } });
}

/// Weighted distribution — `value dist { items... }`, each item from `distItem`.
pub fn dist(ir: *Ir, gpa: Allocator, value: Node.Index, items: []const Node.Index) Allocator.Error!Node.Index {
    const start = try ir.addCounted(gpa, Node.Index, items);
    return ir.addNode(gpa, .{ .tag = .dist, .data = .{ .lhs = @intFromEnum(value), .rhs = start } });
}

/// `if (cond) then_stmt else else_stmt`. Pass `.null` for `else_stmt` when
/// there is no `else` branch.
pub fn ifElse(
    ir: *Ir,
    gpa: Allocator,
    cond: Node.Index,
    then_stmt: Node.Index,
    else_stmt: Node.Index,
) Allocator.Error!Node.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.appendSlice(gpa, &.{ @intFromEnum(then_stmt), @intFromEnum(else_stmt) });
    return ir.addNode(gpa, .{ .tag = .if_else, .data = .{ .lhs = @intFromEnum(cond), .rhs = start } });
}

/// `unique { nodes... }` — every listed value must differ from the others.
pub fn unique(ir: *Ir, gpa: Allocator, nodes: []const Node.Index) Allocator.Error!Node.Index {
    const start = try ir.addCounted(gpa, Node.Index, nodes);
    return ir.addNode(gpa, .{ .tag = .unique, .data = .{ .lhs = start } });
}

/// `solve before... before after...` — a solve-ordering hint, not a boolean.
pub fn solveBefore(
    ir: *Ir,
    gpa: Allocator,
    before: []const Variable.Index,
    after: []const Variable.Index,
) Allocator.Error!Node.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.ensureUnusedCapacity(gpa, before.len + after.len + 2);
    inline for (.{ before, after }) |group| {
        ir.extra.appendAssumeCapacity(@intCast(group.len));
        for (group) |v| ir.extra.appendAssumeCapacity(@intFromEnum(v));
    }
    return ir.addNode(gpa, .{ .tag = .solve_before, .data = .{ .lhs = start } });
}

/// Append `[count, items...]` to `extra` and return the start offset — the
/// encoding every variable-arity node uses for its payload.
fn addCounted(ir: *Ir, gpa: Allocator, comptime T: type, items: []const T) Allocator.Error!u32 {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.ensureUnusedCapacity(gpa, items.len + 1);
    ir.extra.appendAssumeCapacity(@intCast(items.len));
    for (items) |item| ir.extra.appendAssumeCapacity(@intFromEnum(item));
    return start;
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
///
/// Recursion depth is the expression's depth. To type a whole IR, prefer
/// `resolveTypes`, which is linear and iterative.
pub fn typeOf(ir: *const Ir, node: Node.Index) Type {
    const i = @intFromEnum(node);
    const tag = ir.nodes.items(.tag)[i];
    const d = ir.nodes.items(.data)[i];
    return if (propagatesOperandType(tag)) ir.typeOf(@enumFromInt(d.lhs)) else ir.ownType(tag, d);
}

/// Resolve every node's type in one linear forward pass, returning a
/// caller-owned array indexed by `Node.Index`. Same answer as `typeOf` node by
/// node, but linear instead of quadratic in the tree depth, and iterative — an
/// operand always has a lower index than its user, so a single sweep suffices.
/// Requires an IR that `validate` accepts (one built through this API always
/// is).
pub fn resolveTypes(ir: *const Ir, gpa: Allocator) Allocator.Error![]Type {
    const out = try gpa.alloc(Type, ir.nodes.len);
    errdefer gpa.free(out);
    for (ir.nodes.items(.tag), ir.nodes.items(.data), out, 0..) |tag, d, *t, i| {
        t.* = ir.widthOf(tag, d, out[0..i]);
    }
    return out;
}

/// One node's type, given the already-resolved types of every lower-indexed
/// node — the non-recursive form of `typeOf`, for callers that keep such a
/// table (see `resolveTypes`).
pub fn widthOf(ir: *const Ir, tag: Node.Tag, d: Node.Data, resolved: []const Type) Type {
    return if (propagatesOperandType(tag)) resolved[d.lhs] else ir.ownType(tag, d);
}

/// Whether a node's type is simply its left operand's. These are the only tags
/// whose type cannot be read off the node itself, and the reason typing is
/// recursive at all.
fn propagatesOperandType(tag: Node.Tag) bool {
    return switch (tag) {
        .neg,
        .bnot,
        .add,
        .sub,
        .mul,
        .sdiv,
        .udiv,
        .smod,
        .umod,
        .band,
        .bor,
        .bxor,
        .sll,
        .srl,
        .sra,
        .range,
        => true,

        .int_literal,
        .bool_literal,
        .var_ref,
        .zext,
        .sext,
        .trunc,
        .lnot,
        .eq,
        .ne,
        .slt,
        .ult,
        .sle,
        .ule,
        .sgt,
        .ugt,
        .sge,
        .uge,
        .land,
        .lor,
        .implies,
        .iff,
        .in,
        .dist,
        .dist_weight_eq,
        .dist_weight_div,
        .if_else,
        .unique,
        .solve_before,
        .foreach,
        => false,
    };
}

/// The type of a node that carries it, for the tags `propagatesOperandType`
/// answers `false` for. Everything not explicitly typed is a 1-bit boolean.
fn ownType(ir: *const Ir, tag: Node.Tag, d: Node.Data) Type {
    return switch (tag) {
        .int_literal => @bitCast(d.rhs),
        .var_ref => varType(ir.vars.items(.ty)[d.lhs]),
        .zext, .sext, .trunc => .{ .width = @intCast(d.rhs) },
        else => .{ .width = 1 },
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

// -- Validation --------------------------------------------------------------

pub const ValidateError = error{
    /// The IR is not structurally sound; see `validate`.
    InvalidIr,
};

/// Check that the IR is structurally sound, so that consumers can walk it
/// without bounds checks. An IR built through this API always passes; the point
/// of the pass is untrusted input — `deserialize` runs it before handing back a
/// cache blob, and a C caller can run it on an IR it assembled by hand.
///
/// It checks that:
///
///   * every tag and variable kind is a value this build knows;
///   * every operand index refers to an existing node with a *lower* index —
///     evaluation is a single forward sweep, so this is what makes the tree
///     acyclic and in evaluation order — and every variable index exists;
///   * every `extra` payload (literal words, `in`/`dist`/`unique` members,
///     `solve_before` lists, `foreach` bodies, constraint bodies) lies inside
///     the pool and has the length its node claims;
///   * literal and cast widths are non-zero and fit a `u16`;
///   * operand widths agree wherever the evaluator assumes a single width
///     (arithmetic, comparisons, `range` bounds, `in` members). Shifts and the
///     logical operators are exempt: they read the right operand at its own
///     width, or only for truthiness.
///
/// `gpa` is used for a scratch array of resolved node types and is released
/// before returning.
pub fn validate(ir: *const Ir, gpa: Allocator) (Allocator.Error || ValidateError)!void {
    if (ir.nodes.len > std.math.maxInt(u32)) return error.InvalidIr;
    if (ir.vars.len > std.math.maxInt(u32)) return error.InvalidIr;
    if (ir.extra.items.len > std.math.maxInt(u32)) return error.InvalidIr;

    // Enum-typed bytes are checked first, as raw bytes: an out-of-range tag or
    // kind loaded as its enum type would be illegal behavior, and deserialized
    // bytes have not been vetted yet.
    for (std.mem.sliceAsBytes(ir.nodes.items(.tag))) |raw| {
        if (std.enums.fromInt(Node.Tag, raw) == null) return error.InvalidIr;
    }
    for (std.mem.sliceAsBytes(ir.vars.items(.kind))) |raw| {
        if (std.enums.fromInt(Variable.Kind, raw) == null) return error.InvalidIr;
    }

    const types = try gpa.alloc(Type, ir.nodes.len);
    defer gpa.free(types);

    const tags = ir.nodes.items(.tag);
    const datas = ir.nodes.items(.data);
    for (tags, datas, 0..) |tag, d, iu| {
        const i: u32 = @intCast(iu);
        switch (tag) {
            .int_literal => {
                const ty: Type = @bitCast(d.rhs);
                if (ty.width == 0) return error.InvalidIr;
                if ((try counted(ir, d.lhs)).len != litWords(ty.width)) return error.InvalidIr;
            },
            .bool_literal => if (d.lhs > 1) return error.InvalidIr,
            .var_ref => try below(d.lhs, ir.vars.len),

            .zext, .sext, .trunc => {
                try below(d.lhs, i);
                if (d.rhs == 0 or d.rhs > std.math.maxInt(u16)) return error.InvalidIr;
            },

            .neg, .bnot, .lnot => try below(d.lhs, i),

            .add,
            .sub,
            .mul,
            .sdiv,
            .udiv,
            .smod,
            .umod,
            .band,
            .bor,
            .bxor,
            .eq,
            .ne,
            .slt,
            .ult,
            .sle,
            .ule,
            .sgt,
            .ugt,
            .sge,
            .uge,
            .range,
            .sll,
            .srl,
            .sra,
            .land,
            .lor,
            .implies,
            .iff,
            .dist_weight_eq,
            .dist_weight_div,
            => {
                try below(d.lhs, i);
                try below(d.rhs, i);
                if (requiresEqualWidths(tag) and types[d.lhs].width != types[d.rhs].width) {
                    return error.InvalidIr;
                }
            },

            .in => {
                try below(d.lhs, i);
                for (try counted(ir, d.rhs)) |m| {
                    try below(m, i);
                    if (types[m].width != types[d.lhs].width) return error.InvalidIr;
                }
            },
            .dist => {
                try below(d.lhs, i);
                for (try counted(ir, d.rhs)) |item| {
                    try below(item, i);
                    switch (tags[item]) {
                        .dist_weight_eq, .dist_weight_div => {},
                        else => return error.InvalidIr,
                    }
                }
            },
            .if_else => {
                try below(d.lhs, i);
                const arms = try window(ir, d.rhs, 2);
                try below(arms[0], i);
                if (arms[1] != @intFromEnum(Node.Index.null)) try below(arms[1], i);
            },
            .unique => for (try counted(ir, d.lhs)) |m| try below(m, i),
            .solve_before => {
                const before = try counted(ir, d.lhs);
                const after = try counted(ir, @as(usize, d.lhs) + 1 + before.len);
                for (before) |v| try below(v, ir.vars.len);
                for (after) |v| try below(v, ir.vars.len);
            },
            .foreach => {
                try below(d.lhs, ir.vars.len);
                try below((try window(ir, d.rhs, 1))[0], ir.vars.len);
                for (try counted(ir, @as(usize, d.rhs) + 1)) |b| try below(b, i);
            },
        }
        types[i] = ir.widthOf(tag, d, types[0..i]);
    }

    for (ir.constraints.items(.body)) |body| {
        for (try window(ir, @intFromEnum(body.start), body.len)) |stmt| {
            try below(stmt, ir.nodes.len);
        }
    }
}

/// Whether a binary `tag`'s operands must have the same width. True for the
/// operators the evaluator runs at a single width; false for the shifts, which
/// read the shift amount at its own width, and for the logical operators, which
/// only test for non-zero.
pub fn requiresEqualWidths(tag: Node.Tag) bool {
    return switch (tag) {
        .add,
        .sub,
        .mul,
        .sdiv,
        .udiv,
        .smod,
        .umod,
        .band,
        .bor,
        .bxor,
        .eq,
        .ne,
        .slt,
        .ult,
        .sle,
        .ule,
        .sgt,
        .ugt,
        .sge,
        .uge,
        .range,
        => true,
        else => false,
    };
}

fn below(index: u32, limit: usize) ValidateError!void {
    if (index >= limit) return error.InvalidIr;
}

/// The `extra` window `[at, at + len)`, or `InvalidIr` if it runs off the pool.
fn window(ir: *const Ir, at: usize, len: usize) ValidateError![]const u32 {
    const pool = ir.extra.items;
    if (at > pool.len or len > pool.len - at) return error.InvalidIr;
    return pool[at..][0..len];
}

/// The items of a `[count, items...]` payload at `at`.
fn counted(ir: *const Ir, at: usize) ValidateError![]const u32 {
    const count = (try window(ir, at, 1))[0];
    return window(ir, at + 1, count);
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
/// On-disk format version. Bump on any layout change — including a change to
/// a stored enum's numeric values; readers reject mismatches.
pub const format_version: u32 = 4;

const checksum_len = Blake3.digest_length;
/// Bytes preceding the first section: `magic` + `format_version`.
const header_len = magic.len + @sizeOf(u32);

/// Exact byte length of this IR's `serialize` output, computed without
/// serializing, so a caller can size a buffer up front.
pub fn serializedSize(ir: *const Ir) usize {
    const per_node = @sizeOf(Node.Tag) + @sizeOf(Node.Data);
    const per_var = @sizeOf(Variable.Id) + @sizeOf(Type) + @sizeOf(Variable.Kind);
    const per_constraint = @sizeOf(Constraint.Id) + @sizeOf(Constraint.Flags) + @sizeOf(Extra.Slice);
    return header_len + checksum_len +
        4 + ir.nodes.len * per_node +
        4 + ir.vars.len * per_var +
        4 + ir.constraints.len * per_constraint +
        4 + ir.extra.items.len * @sizeOf(u32);
}

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
} || ValidateError || Allocator.Error;

/// Reconstruct an `Ir` from bytes produced by `serialize`. The result owns its
/// storage and must be `deinit`ed.
///
/// The checksum is verified before any length is trusted, and the reconstructed
/// IR is then run through `validate`, so a blob that survives both can be walked
/// without bounds checks even if it was hostile.
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

    try ir.validate(gpa);
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
    try std.testing.expectEqual(ir.serializedSize(), bytes.len);

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

test "validate accepts what the builders produce, including structural nodes" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(8), .kind = .rand });

    // if (x == 3) y == 4;
    const cond = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 3, Type.bit(8)));
    const then = try ir.binary(gpa, .eq, try ir.varRef(gpa, y), try ir.constInt(gpa, 4, Type.bit(8)));
    const conditional = try ir.ifElse(gpa, cond, then, .null);

    // x dist { [0:7] :/ 10 };
    const item = try ir.distItem(
        gpa,
        .eq,
        try ir.range(gpa, try ir.constInt(gpa, 0, Type.bit(8)), try ir.constInt(gpa, 7, Type.bit(8))),
        try ir.constInt(gpa, 10, Type.bit(8)),
    );
    const weighted = try ir.dist(gpa, try ir.varRef(gpa, x), &.{item});

    const distinct = try ir.unique(gpa, &.{ try ir.varRef(gpa, x), try ir.varRef(gpa, y) });
    const order = try ir.solveBefore(gpa, &.{x}, &.{y});

    _ = try ir.addConstraint(gpa, @enumFromInt(9), .{ .soft = true }, &.{ conditional, weighted, distinct, order });
    try ir.validate(gpa);

    // And they survive a cache round-trip, which validates on the way back in.
    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);
    var ir2 = try Ir.deserialize(gpa, bytes);
    defer ir2.deinit(gpa);
    try std.testing.expectEqual(ir.hash(), ir2.hash());
}

test "validate rejects malformed IRs" {
    const gpa = std.testing.allocator;

    // An operand that refers forward (here, to the node itself and past the end)
    // would make the forward-sweep evaluator read an unresolved value.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        _ = try ir.addNode(gpa, .{ .tag = .add, .data = .{ .lhs = 0, .rhs = 1 } });
        try std.testing.expectError(error.InvalidIr, ir.validate(gpa));
    }

    // A reference to a variable that was never declared.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        _ = try ir.varRef(gpa, @enumFromInt(3));
        try std.testing.expectError(error.InvalidIr, ir.validate(gpa));
    }

    // Operands of different widths, which the evaluator has no rule for.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const a = try ir.constInt(gpa, 1, Type.bit(4));
        const b = try ir.constInt(gpa, 1, Type.bit(8));
        _ = try ir.binary(gpa, .add, a, b);
        try std.testing.expectError(error.InvalidIr, ir.validate(gpa));
    }

    // A constraint body pointing at a node that does not exist.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{@enumFromInt(7)});
        try std.testing.expectError(error.InvalidIr, ir.validate(gpa));
    }
}

test "deserialize rejects a checksum-valid but malformed blob" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
    defer ir.deinit(gpa);
    _ = try ir.constInt(gpa, 1, Type.bit(8));

    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);

    const tampered = try gpa.dupe(u8, bytes);
    defer gpa.free(tampered);

    // Corrupt the first node's tag into one no build knows, then re-checksum so
    // the blob passes the integrity check — validation is what has to catch it.
    tampered[header_len + 4] = 254;
    const payload = tampered[0 .. tampered.len - checksum_len];
    var digest: Digest = undefined;
    Blake3.hash(payload, &digest, .{});
    @memcpy(tampered[tampered.len - checksum_len ..], &digest);

    try std.testing.expectError(error.InvalidIr, Ir.deserialize(gpa, tampered));
}

test "constBits and constBig agree on the same magnitude" {
    const gpa = std.testing.allocator;

    var from_big: Ir = .{};
    defer from_big.deinit(gpa);
    var limbs = [_]std.math.big.Limb{ 0xdead_beef, 1 << 36 };
    _ = try from_big.constBig(gpa, .{ .limbs = &limbs, .positive = true }, Type.bit(128));

    var from_words: Ir = .{};
    defer from_words.deinit(gpa);
    _ = try from_words.constBits(gpa, &.{ 0xdead_beef, 1 << 36 }, Type.bit(128));

    try std.testing.expectEqual(from_big.hash(), from_words.hash());
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

    // The linear sweep agrees with the recursive resolver, node for node.
    const types = try ir.resolveTypes(gpa);
    defer gpa.free(types);
    for (types, 0..) |t, i| {
        try std.testing.expectEqual(ir.typeOf(@enumFromInt(@as(u32, @intCast(i)))), t);
    }
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

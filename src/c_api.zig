//! C ABI bindings — the implementation behind `include/crv.h`.
//!
//! This file is the root of the `libcrv` C library artifact, and deliberately
//! not part of the `crv` Zig module: a Zig program that imports the module
//! therefore emits none of these symbols and cannot collide with the library.
//!
//! Nothing here reimplements the IR or the solver. Each export is argument
//! checking, a call into the Zig core, and an error-to-status mapping. Two
//! pieces of bookkeeping live in the C handles because they exist to protect C
//! callers from mistakes Zig's type system would otherwise catch:
//!
//!   * the resolved width of every node, maintained as nodes are appended, so
//!     a builder can reject mismatched operand widths in constant time; and
//!   * a generation counter, so a solver notices an IR that was mutated after
//!     it was bound to it rather than reading a stale node-type table.
//!
//! The enum values crossing the ABI are mapped explicitly, never cast, so the
//! IR's internal tag ordering can change without breaking a compiled consumer.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const Ir = @import("Ir.zig");
const Solver = @import("Solver.zig");
const RejectionSampler = @import("RejectionSampler.zig");

/// Every handle allocates from here. `crv.h` exposes no allocator hook, so a
/// future `crv_ir_new_with_allocator` can add one without disturbing anything.
const gpa: Allocator = if (builtin.link_libc) std.heap.c_allocator else std.heap.smp_allocator;

/// Keep in sync with the `CRV_VERSION_*` macros in `include/crv.h` and the
/// version in `build.zig.zon`.
const version = "0.1.0";

/// Written through an out-parameter when a builder fails: `CRV_INVALID`.
const invalid: u32 = std.math.maxInt(u32);

// -- Status ------------------------------------------------------------------

/// Mirrors `crv_status`. Negative is a hard error.
const Status = enum(c_int) {
    ok = 0,
    exhausted = 1,

    err_oom = -1,
    err_invalid_argument = -2,
    err_width_mismatch = -3,
    err_buffer_too_small = -4,
    err_invalid_ir = -5,
    err_unsupported_node = -6,
    err_stale = -7,
    err_bad_magic = -8,
    err_unsupported_version = -9,
    err_checksum_mismatch = -10,
    err_truncated = -11,
};

/// Every error the core can hand back at the ABI boundary. Naming the union
/// keeps `statusOf` exhaustive, so a new error in the core is a compile error
/// here rather than a silently mistranslated status.
const CoreError = Ir.DeserializeError || RejectionSampler.InitError;

fn statusOf(err: CoreError) Status {
    return switch (err) {
        error.OutOfMemory => .err_oom,
        error.InvalidIr => .err_invalid_ir,
        error.UnsupportedNode => .err_unsupported_node,
        error.BadMagic => .err_bad_magic,
        error.UnsupportedVersion => .err_unsupported_version,
        error.ChecksumMismatch => .err_checksum_mismatch,
        error.Truncated => .err_truncated,
    };
}

export fn crv_version_string() [*:0]const u8 {
    return version;
}

export fn crv_status_string(status: c_int) [*:0]const u8 {
    const s = std.enums.fromInt(Status, status) orelse return "unknown status";
    return switch (s) {
        .ok => "ok",
        .exhausted => "solver gave up within its attempt budget",
        .err_oom => "out of memory",
        .err_invalid_argument => "invalid argument",
        .err_width_mismatch => "operand widths disagree",
        .err_buffer_too_small => "buffer too small",
        .err_invalid_ir => "malformed IR",
        .err_unsupported_node => "IR node unsupported by this solver",
        .err_stale => "IR was modified after the solver was bound to it",
        .err_bad_magic => "not a libcrv cache blob",
        .err_unsupported_version => "unsupported cache format version",
        .err_checksum_mismatch => "cache blob failed its checksum",
        .err_truncated => "cache blob ends mid-record",
    };
}

// -- Handles -----------------------------------------------------------------

/// What a `crv_ir *` points at.
const CIr = struct {
    ir: Ir = .{},
    /// Each node's resolved type, parallel to `ir.nodes` — see the module doc.
    types: std.ArrayListUnmanaged(Ir.Type) = .empty,
    /// Bumped on every mutation.
    generation: u32 = 0,

    fn nodeIndex(c: *const CIr, n: u32) ?Ir.Node.Index {
        if (n >= c.ir.nodes.len) return null;
        return @enumFromInt(n);
    }

    fn varIndex(c: *const CIr, v: u32) ?Ir.Variable.Index {
        if (v >= c.ir.vars.len) return null;
        return @enumFromInt(v);
    }

    /// The width of an already-checked node index.
    fn width(c: *const CIr, n: Ir.Node.Index) u16 {
        return c.types.items[@intFromEnum(n)].width;
    }

    /// Room for the type of the one node the caller is about to append, taken
    /// before the append so the two arrays cannot fall out of step.
    fn reserveType(c: *CIr) Allocator.Error!void {
        try c.types.ensureUnusedCapacity(gpa, 1);
    }

    /// Record a freshly built node: cache its type, publish it, count the
    /// mutation. Pair every call with a preceding `reserveType`.
    fn emit(c: *CIr, out: ?*u32, built: Allocator.Error!Ir.Node.Index) Status {
        const node = built catch return .err_oom;
        c.generation +%= 1;

        const i = @intFromEnum(node);
        std.debug.assert(i == c.types.items.len);
        const tag = c.ir.nodes.items(.tag)[i];
        const data = c.ir.nodes.items(.data)[i];
        c.types.appendAssumeCapacity(c.ir.widthOf(tag, data, c.types.items));

        if (out) |o| o.* = i;
        return .ok;
    }
};

/// What a `crv_solver *` points at. One type per handle, whichever engine is
/// behind it — the C mirror of the `Solver` vtable.
const CSolver = struct {
    owner: *const CIr,
    /// The owner's generation when this solver was bound.
    generation: u32,
    value_words: usize,
    var_count: usize,
    engine: Engine,

    const Engine = union(enum) {
        rejection: RejectionSampler,
    };

    fn solver(s: *CSolver) Solver {
        return switch (s.engine) {
            .rejection => |*e| e.solver(),
        };
    }
};

fn setInvalid(out: ?*u32) void {
    if (out) |o| o.* = invalid;
}

/// A `[ptr, len)` pair from C as a slice, or null if it is not one. An empty
/// slice is accepted with a null pointer, which is what C callers naturally
/// pass for "no members".
fn sliceOf(comptime T: type, ptr: ?[*]const T, len: usize) ?[]const T {
    if (len == 0) return &.{};
    const p = ptr orelse return null;
    return p[0..len];
}

/// Like `sliceOf`, but every element must name an existing node.
fn nodeSlice(c: *const CIr, ptr: ?[*]const u32, len: usize) ?[]const Ir.Node.Index {
    const raw = sliceOf(u32, ptr, len) orelse return null;
    for (raw) |n| _ = c.nodeIndex(n) orelse return null;
    return @ptrCast(raw);
}

/// Like `sliceOf`, but every element must name an existing variable.
fn varSlice(c: *const CIr, ptr: ?[*]const u32, len: usize) ?[]const Ir.Variable.Index {
    const raw = sliceOf(u32, ptr, len) orelse return null;
    for (raw) |v| _ = c.varIndex(v) orelse return null;
    return @ptrCast(raw);
}

// -- ABI mappings ------------------------------------------------------------
//
// Explicit both ways: the wire values in `crv.h` are fixed, the IR's tags are
// free to move.

fn varKind(raw: c_int) ?Ir.Variable.Kind {
    return switch (raw) {
        0 => .state,
        1 => .rand,
        2 => .randc,
        else => null,
    };
}

fn varKindValue(kind: Ir.Variable.Kind) u8 {
    return switch (kind) {
        .state => 0,
        .rand => 1,
        .randc => 2,
    };
}

fn unaryTag(raw: c_int) ?Ir.Node.Tag {
    return switch (raw) {
        1 => .neg,
        2 => .bnot,
        3 => .lnot,
        else => null,
    };
}

fn binaryTag(raw: c_int) ?Ir.Node.Tag {
    return switch (raw) {
        32 => .add,
        33 => .sub,
        34 => .mul,
        35 => .sdiv,
        36 => .udiv,
        37 => .smod,
        38 => .umod,
        39 => .band,
        40 => .bor,
        41 => .bxor,
        42 => .sll,
        43 => .srl,
        44 => .sra,
        64 => .eq,
        65 => .ne,
        66 => .slt,
        67 => .ult,
        68 => .sle,
        69 => .ule,
        70 => .sgt,
        71 => .ugt,
        72 => .sge,
        73 => .uge,
        74 => .land,
        75 => .lor,
        76 => .implies,
        77 => .iff,
        else => null,
    };
}

const Cast = enum { zext, sext, trunc };

fn castKind(raw: c_int) ?Cast {
    return switch (raw) {
        1 => .zext,
        2 => .sext,
        3 => .trunc,
        else => null,
    };
}

fn distKind(raw: c_int) ?Ir.DistKind {
    return switch (raw) {
        0 => .eq,
        1 => .div,
        else => null,
    };
}

// -- IR lifetime -------------------------------------------------------------

export fn crv_ir_new() ?*CIr {
    const c = gpa.create(CIr) catch return null;
    c.* = .{};
    return c;
}

export fn crv_ir_free(handle: ?*CIr) void {
    const c = handle orelse return;
    c.types.deinit(gpa);
    c.ir.deinit(gpa);
    gpa.destroy(c);
}

export fn crv_ir_reserve(handle: ?*CIr, nodes: u32, vars: u32, extra: u32) Status {
    const c = handle orelse return .err_invalid_argument;
    c.ir.nodes.ensureTotalCapacity(gpa, nodes) catch return .err_oom;
    c.types.ensureTotalCapacity(gpa, nodes) catch return .err_oom;
    c.ir.vars.ensureTotalCapacity(gpa, vars) catch return .err_oom;
    c.ir.extra.ensureTotalCapacity(gpa, extra) catch return .err_oom;
    return .ok;
}

export fn crv_ir_validate(handle: ?*const CIr) Status {
    const c = handle orelse return .err_invalid_argument;
    c.ir.validate(gpa) catch |err| return statusOf(err);
    return .ok;
}

export fn crv_ir_node_count(handle: ?*const CIr) u32 {
    const c = handle orelse return 0;
    return @intCast(c.ir.nodes.len);
}

// -- Variables ---------------------------------------------------------------

const VarInfo = extern struct {
    id: u32,
    width: u16,
    kind: u8,
    reserved: u8,
};

export fn crv_var_add(handle: ?*CIr, id: u32, width: u16, kind: c_int, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    if (width == 0) return .err_invalid_argument;
    const k = varKind(kind) orelse return .err_invalid_argument;

    const v = c.ir.addVariable(gpa, .{
        .id = @enumFromInt(id),
        .ty = Ir.Type.bit(width),
        .kind = k,
    }) catch return .err_oom;

    c.generation +%= 1;
    if (out) |o| o.* = @intFromEnum(v);
    return .ok;
}

export fn crv_var_count(handle: ?*const CIr) u32 {
    const c = handle orelse return 0;
    return @intCast(c.ir.vars.len);
}

export fn crv_var_get(handle: ?*const CIr, v: u32, out: ?*VarInfo) Status {
    const c = handle orelse return .err_invalid_argument;
    const index = c.varIndex(v) orelse return .err_invalid_argument;
    const i = @intFromEnum(index);
    if (out) |o| o.* = .{
        .id = @intFromEnum(c.ir.vars.items(.id)[i]),
        .width = c.ir.vars.items(.ty)[i].width,
        .kind = varKindValue(c.ir.vars.items(.kind)[i]),
        .reserved = 0,
    };
    return .ok;
}

// -- Leaves ------------------------------------------------------------------

export fn crv_node_var(handle: ?*CIr, v: u32, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const index = c.varIndex(v) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.varRef(gpa, index));
}

export fn crv_node_bool(handle: ?*CIr, value: c_int, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.boolLit(gpa, value != 0));
}

export fn crv_node_const_u64(handle: ?*CIr, value: u64, width: u16, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    if (width == 0) return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.constInt(gpa, value, Ir.Type.bit(width)));
}

export fn crv_node_const_bits(
    handle: ?*CIr,
    words: ?[*]const u64,
    nwords: usize,
    width: u16,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    if (width == 0) return .err_invalid_argument;
    const magnitude = sliceOf(u64, words, nwords) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.constBits(gpa, magnitude, Ir.Type.bit(width)));
}

// -- Operators ---------------------------------------------------------------

export fn crv_node_unary(handle: ?*CIr, op: c_int, a: u32, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const tag = unaryTag(op) orelse return .err_invalid_argument;
    const operand = c.nodeIndex(a) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.unary(gpa, tag, operand));
}

export fn crv_node_binary(handle: ?*CIr, op: c_int, a: u32, b: u32, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const tag = binaryTag(op) orelse return .err_invalid_argument;
    const lhs = c.nodeIndex(a) orelse return .err_invalid_argument;
    const rhs = c.nodeIndex(b) orelse return .err_invalid_argument;
    if (Ir.requiresEqualWidths(tag) and c.width(lhs) != c.width(rhs)) {
        return .err_width_mismatch;
    }
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.binary(gpa, tag, lhs, rhs));
}

export fn crv_node_cast(handle: ?*CIr, cast: c_int, a: u32, width: u16, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const kind = castKind(cast) orelse return .err_invalid_argument;
    const operand = c.nodeIndex(a) orelse return .err_invalid_argument;
    if (width == 0) return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, switch (kind) {
        .zext => c.ir.zext(gpa, operand, width),
        .sext => c.ir.sext(gpa, operand, width),
        .trunc => c.ir.trunc(gpa, operand, width),
    });
}

export fn crv_node_width(handle: ?*const CIr, n: u32, out_width: ?*u16) Status {
    const c = handle orelse return .err_invalid_argument;
    const node = c.nodeIndex(n) orelse return .err_invalid_argument;
    if (out_width) |o| o.* = c.width(node);
    return .ok;
}

// -- Sets, distributions, structural constraints -----------------------------

export fn crv_node_range(handle: ?*CIr, lo: u32, hi: u32, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const low = c.nodeIndex(lo) orelse return .err_invalid_argument;
    const high = c.nodeIndex(hi) orelse return .err_invalid_argument;
    if (c.width(low) != c.width(high)) return .err_width_mismatch;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.range(gpa, low, high));
}

export fn crv_node_in(
    handle: ?*CIr,
    value: u32,
    members: ?[*]const u32,
    n: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const tested = c.nodeIndex(value) orelse return .err_invalid_argument;
    const set = nodeSlice(c, members, n) orelse return .err_invalid_argument;
    for (set) |m| {
        if (c.width(m) != c.width(tested)) return .err_width_mismatch;
    }
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.in(gpa, tested, set));
}

export fn crv_node_dist_item(
    handle: ?*CIr,
    kind: c_int,
    value: u32,
    weight: u32,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const k = distKind(kind) orelse return .err_invalid_argument;
    const v = c.nodeIndex(value) orelse return .err_invalid_argument;
    const w = c.nodeIndex(weight) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.distItem(gpa, k, v, w));
}

export fn crv_node_dist(
    handle: ?*CIr,
    value: u32,
    items: ?[*]const u32,
    n: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const v = c.nodeIndex(value) orelse return .err_invalid_argument;
    const list = nodeSlice(c, items, n) orelse return .err_invalid_argument;
    for (list) |item| {
        switch (c.ir.nodes.items(.tag)[@intFromEnum(item)]) {
            .dist_weight_eq, .dist_weight_div => {},
            else => return .err_invalid_argument,
        }
    }
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.dist(gpa, v, list));
}

export fn crv_node_if(handle: ?*CIr, cond: u32, then_stmt: u32, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const condition = c.nodeIndex(cond) orelse return .err_invalid_argument;
    const body = c.nodeIndex(then_stmt) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.ifElse(gpa, condition, body, .null));
}

export fn crv_node_if_else(
    handle: ?*CIr,
    cond: u32,
    then_stmt: u32,
    else_stmt: u32,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const condition = c.nodeIndex(cond) orelse return .err_invalid_argument;
    const body = c.nodeIndex(then_stmt) orelse return .err_invalid_argument;
    const alternative = c.nodeIndex(else_stmt) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.ifElse(gpa, condition, body, alternative));
}

export fn crv_node_unique(handle: ?*CIr, nodes: ?[*]const u32, n: usize, out: ?*u32) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const set = nodeSlice(c, nodes, n) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.unique(gpa, set));
}

export fn crv_node_solve_before(
    handle: ?*CIr,
    before: ?[*]const u32,
    nbefore: usize,
    after: ?[*]const u32,
    nafter: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    const first = varSlice(c, before, nbefore) orelse return .err_invalid_argument;
    const second = varSlice(c, after, nafter) orelse return .err_invalid_argument;
    c.reserveType() catch return .err_oom;
    return c.emit(out, c.ir.solveBefore(gpa, first, second));
}

// -- Constraints -------------------------------------------------------------

const ConstraintInfo = extern struct {
    id: u32,
    flags: u32,
    stmt_count: u32,
};

const constraint_soft: u32 = 0x1;

export fn crv_constraint_add(
    handle: ?*CIr,
    id: u32,
    flags: u32,
    stmts: ?[*]const u32,
    n: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const c = handle orelse return .err_invalid_argument;
    if (flags & ~constraint_soft != 0) return .err_invalid_argument;
    const body = nodeSlice(c, stmts, n) orelse return .err_invalid_argument;

    const index = c.ir.addConstraint(
        gpa,
        @enumFromInt(id),
        .{ .soft = flags & constraint_soft != 0 },
        body,
    ) catch return .err_oom;

    c.generation +%= 1;
    if (out) |o| o.* = @intFromEnum(index);
    return .ok;
}

export fn crv_constraint_count(handle: ?*const CIr) u32 {
    const c = handle orelse return 0;
    return @intCast(c.ir.constraints.len);
}

export fn crv_constraint_get(handle: ?*const CIr, constraint: u32, out: ?*ConstraintInfo) Status {
    const c = handle orelse return .err_invalid_argument;
    if (constraint >= c.ir.constraints.len) return .err_invalid_argument;
    const i: usize = constraint;
    if (out) |o| o.* = .{
        .id = @intFromEnum(c.ir.constraints.items(.id)[i]),
        .flags = if (c.ir.constraints.items(.flags)[i].soft) constraint_soft else 0,
        .stmt_count = c.ir.constraints.items(.body)[i].len,
    };
    return .ok;
}

// -- Hashing and caching -----------------------------------------------------

export fn crv_ir_hash(handle: ?*const CIr, out: ?*Ir.Digest) Status {
    const c = handle orelse return .err_invalid_argument;
    if (out) |o| o.* = c.ir.hash();
    return .ok;
}

export fn crv_ir_serialized_size(handle: ?*const CIr) usize {
    const c = handle orelse return 0;
    return c.ir.serializedSize();
}

export fn crv_ir_serialize(handle: ?*const CIr, buf: ?*anyopaque, cap: usize, written: ?*usize) Status {
    const c = handle orelse return .err_invalid_argument;
    const size = c.ir.serializedSize();
    if (written) |w| w.* = size;
    if (cap < size) return .err_buffer_too_small;
    const dst = buf orelse return .err_invalid_argument;

    const bytes = c.ir.serialize(gpa) catch return .err_oom;
    defer gpa.free(bytes);
    std.debug.assert(bytes.len == size);
    @memcpy(@as([*]u8, @ptrCast(dst))[0..bytes.len], bytes);
    return .ok;
}

export fn crv_ir_deserialize(buf: ?*const anyopaque, len: usize, out: ?*?*CIr) Status {
    const result = out orelse return .err_invalid_argument;
    result.* = null;
    const src = buf orelse return .err_invalid_argument;

    var ir = Ir.deserialize(gpa, @as([*]const u8, @ptrCast(src))[0..len]) catch |err| {
        return statusOf(err);
    };
    const types = ir.resolveTypes(gpa) catch {
        ir.deinit(gpa);
        return .err_oom;
    };
    const c = gpa.create(CIr) catch {
        gpa.free(types);
        ir.deinit(gpa);
        return .err_oom;
    };

    c.* = .{ .ir = ir, .types = std.ArrayListUnmanaged(Ir.Type).fromOwnedSlice(types) };
    result.* = c;
    return .ok;
}

// -- Solving -----------------------------------------------------------------

const RejectionOptions = extern struct {
    seed: u64,
    max_attempts: u32,
    reserved: u32,
};

const Stats = extern struct {
    attempts: u64,
    hits: u64,
};

export fn crv_value_words(handle: ?*const CIr) usize {
    const c = handle orelse return 0;
    return Solver.valueLimbs(&c.ir);
}

export fn crv_rejection_sampler_new(
    handle: ?*const CIr,
    options: ?*const RejectionOptions,
    out: ?*?*CSolver,
) Status {
    const result = out orelse return .err_invalid_argument;
    result.* = null;
    const c = handle orelse return .err_invalid_argument;

    // The engine indexes the IR's arrays unchecked, so it only ever sees one
    // that has been vetted — even if the caller assembled it by hand.
    c.ir.validate(gpa) catch |err| return statusOf(err);

    const defaults: RejectionSampler.Options = .{};
    const opts: RejectionSampler.Options = if (options) |o| .{
        .seed = o.seed,
        .max_attempts = if (o.max_attempts == 0) defaults.max_attempts else o.max_attempts,
    } else defaults;

    const engine = RejectionSampler.init(gpa, &c.ir, opts) catch |err| return statusOf(err);

    const s = gpa.create(CSolver) catch {
        var mutable = engine;
        mutable.deinit(gpa);
        return .err_oom;
    };
    s.* = .{
        .owner = c,
        .generation = c.generation,
        .value_words = Solver.valueLimbs(&c.ir),
        .var_count = c.ir.vars.len,
        .engine = .{ .rejection = engine },
    };
    result.* = s;
    return .ok;
}

export fn crv_solver_free(handle: ?*CSolver) void {
    const s = handle orelse return;
    switch (s.engine) {
        .rejection => |*e| e.deinit(gpa),
    }
    gpa.destroy(s);
}

export fn crv_solver_next(handle: ?*CSolver, values: ?[*]u64, nwords: usize) Status {
    const s = handle orelse return .err_invalid_argument;
    if (s.owner.generation != s.generation) return .err_stale;

    const needed = s.value_words * s.var_count;
    if (nwords < needed) return .err_buffer_too_small;

    var none: [0]Solver.Value = .{};
    const out: []Solver.Value = if (needed == 0)
        &none
    else
        (values orelse return .err_invalid_argument)[0..nwords];

    return if (s.solver().next(out)) .ok else .exhausted;
}

export fn crv_solver_stats(handle: ?*const CSolver, out: ?*Stats) Status {
    const s = handle orelse return .err_invalid_argument;
    if (out) |o| o.* = switch (s.engine) {
        .rejection => |*e| .{ .attempts = e.attempts, .hits = e.hits },
    };
    return .ok;
}

// -- Tests -------------------------------------------------------------------
//
// End-to-end coverage lives in `test/smoke.c`, which exercises the real header
// from real C. These cover the paths a well-behaved C program does not reach.

test "builders reject bad handles, indices, and widths" {
    const ir = crv_ir_new().?;
    defer crv_ir_free(ir);

    var node: u32 = 0;
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_bool(null, 1, &node));
    try std.testing.expectEqual(invalid, node);

    // An index no node has, an op code that decodes to nothing, and a zero width.
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(ir, 1, 7, &node));
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(ir, 999, 0, &node));
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_const_u64(ir, 1, 0, &node));

    // Arity is part of the op: a binary op is not a unary one.
    var a: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(ir, 1, 8, &a));
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(ir, 32, a, &node));

    // Widths must agree where the evaluator assumes they do.
    var b: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(ir, 1, 4, &b));
    try std.testing.expectEqual(Status.err_width_mismatch, crv_node_binary(ir, 32, a, b, &node));

    // ... and a cast is how you make them agree.
    var widened: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_cast(ir, 1, b, 8, &widened));
    try std.testing.expectEqual(Status.ok, crv_node_binary(ir, 32, a, widened, &node));

    var width: u16 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_width(ir, node, &width));
    try std.testing.expectEqual(@as(u16, 8), width);
}

test "a solver notices an IR that changed underneath it" {
    const ir = crv_ir_new().?;
    defer crv_ir_free(ir);

    var x: u32 = 0;
    var ref: u32 = 0;
    var lit: u32 = 0;
    var eq: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_var_add(ir, 1, 6, 1, &x));
    try std.testing.expectEqual(Status.ok, crv_node_var(ir, x, &ref));
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(ir, 42, 6, &lit));
    try std.testing.expectEqual(Status.ok, crv_node_binary(ir, 64, ref, lit, &eq));
    const body = [_]u32{eq};
    try std.testing.expectEqual(Status.ok, crv_constraint_add(ir, 0, 0, &body, 1, null));

    var solver: ?*CSolver = null;
    try std.testing.expectEqual(Status.ok, crv_rejection_sampler_new(ir, null, &solver));
    defer crv_solver_free(solver);

    var values: [1]u64 = undefined;
    try std.testing.expectEqual(Status.err_buffer_too_small, crv_solver_next(solver, &values, 0));
    try std.testing.expectEqual(Status.ok, crv_solver_next(solver, &values, 1));
    try std.testing.expectEqual(@as(u64, 42), values[0]);

    var stats: Stats = undefined;
    try std.testing.expectEqual(Status.ok, crv_solver_stats(solver, &stats));
    try std.testing.expectEqual(@as(u64, 1), stats.hits);

    // Appending to the IR invalidates the solver's view of it.
    var another: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(ir, 7, 6, &another));
    try std.testing.expectEqual(Status.err_stale, crv_solver_next(solver, &values, 1));
}

test "a solver refuses an IR it cannot evaluate" {
    const ir = crv_ir_new().?;
    defer crv_ir_free(ir);

    var x: u32 = 0;
    var ref: u32 = 0;
    var distinct: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_var_add(ir, 1, 4, 1, &x));
    try std.testing.expectEqual(Status.ok, crv_node_var(ir, x, &ref));
    const members = [_]u32{ref};
    try std.testing.expectEqual(Status.ok, crv_node_unique(ir, &members, 1, &distinct));
    const body = [_]u32{distinct};
    try std.testing.expectEqual(Status.ok, crv_constraint_add(ir, 0, 0, &body, 1, null));

    var solver: ?*CSolver = null;
    try std.testing.expectEqual(
        Status.err_unsupported_node,
        crv_rejection_sampler_new(ir, null, &solver),
    );
    try std.testing.expectEqual(@as(?*CSolver, null), solver);
}

test "a corrupt cache blob comes back as a status" {
    const ir = crv_ir_new().?;
    defer crv_ir_free(ir);
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(ir, 1, 8, null));

    const size = crv_ir_serialized_size(ir);
    const buf = try std.testing.allocator.alloc(u8, size);
    defer std.testing.allocator.free(buf);

    var written: usize = 0;
    try std.testing.expectEqual(
        Status.err_buffer_too_small,
        crv_ir_serialize(ir, buf.ptr, size - 1, &written),
    );
    try std.testing.expectEqual(size, written);
    try std.testing.expectEqual(Status.ok, crv_ir_serialize(ir, buf.ptr, size, &written));

    buf[size - 1] ^= 0xff;
    var loaded: ?*CIr = null;
    try std.testing.expectEqual(
        Status.err_checksum_mismatch,
        crv_ir_deserialize(buf.ptr, size, &loaded),
    );
    try std.testing.expectEqual(@as(?*CIr, null), loaded);

    const junk = [_]u8{0} ** 64;
    try std.testing.expectEqual(Status.err_bad_magic, crv_ir_deserialize(&junk, junk.len, &loaded));
    try std.testing.expectEqual(Status.err_truncated, crv_ir_deserialize(buf.ptr, 4, &loaded));
}

test {
    std.testing.refAllDecls(@This());
}

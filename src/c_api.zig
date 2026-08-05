//! C ABI bindings — the implementation behind `include/crv.h`.
//!
//! This file is the root of the `libcrv` C library artifact, and deliberately
//! not part of the `crv` Zig module: a Zig program that imports the module
//! therefore emits none of these symbols and cannot collide with the library.
//!
//! Nothing here reimplements the IR or the solver, and nothing here wraps them
//! either: `crv_ir` *is* an `Ir`, seen through a byte buffer of the right size
//! (see `ir_storage`). Every export is argument checking, a call into the Zig
//! core, and an error-to-status mapping. A C caller and a Zig caller therefore
//! get the same representation, the same costs, and the same hazards — in
//! particular, mutating an IR a solver is bound to is undefined for both.
//!
//! The enums crossing the ABI follow the same rule. `crv_op`, `crv_cast`,
//! `crv_var_kind` and `crv_dist_kind` are declared in `crv.h` with the IR's own
//! values, which are fixed for the on-disk format anyway, so an argument is
//! decoded and range-checked rather than translated through a table that could
//! drift out of step with either side.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const Ir = @import("Ir.zig");
const Solver = @import("Solver.zig");
const RejectionSampler = @import("RejectionSampler.zig");

/// Where the IR's arrays — and the solver handles — come from. `crv.h` exposes
/// no allocator hook, so a future `crv_ir_init_with_allocator` can add one
/// without disturbing anything.
const gpa: Allocator = if (builtin.link_libc) std.heap.c_allocator else std.heap.smp_allocator;

/// Keep in sync with the `CRV_VERSION_*` macros in `include/crv.h` and the
/// version in `build.zig.zon`.
const version = "0.1.0";

/// Written through an out-parameter when a builder fails: `CRV_INVALID`.
const invalid: u32 = std.math.maxInt(u32);

/// The `crv_ir` struct in `crv.h` is a raw byte buffer of exactly this size and
/// alignment, so that C code can hold an `Ir` by value without a definition of
/// it. The buffer carries slack on purpose: growing `Ir` past it is an ABI
/// break, and this assertion is the one moment it can be caught, so it fails
/// the library's own build rather than a consumer's runtime.
const ir_storage = struct {
    const size = 128;
    const alignment = 8;
};

comptime {
    std.debug.assert(@sizeOf(Ir) <= ir_storage.size);
    std.debug.assert(@alignOf(Ir) <= ir_storage.alignment);
}

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
    err_bad_magic = -7,
    err_unsupported_version = -8,
    err_checksum_mismatch = -9,
    err_truncated = -10,
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
        .err_bad_magic => "not a libcrv cache blob",
        .err_unsupported_version => "unsupported cache format version",
        .err_checksum_mismatch => "cache blob failed its checksum",
        .err_truncated => "cache blob ends mid-record",
    };
}

// -- Handles -----------------------------------------------------------------

/// What a `crv_solver *` points at. One type per handle, whichever engine is
/// behind it — the C mirror of the `Solver` vtable. Unlike the IR this stays an
/// opaque, library-allocated handle: its size depends on the engine, and a
/// header constant sized to the largest one would become an ABI liability the
/// moment a heavier backend is added.
const CSolver = struct {
    /// The shape the engine was built for, and so the shape a solution buffer
    /// must have. Cached rather than recomputed: an IR that changed underneath
    /// the solver has already invalidated the engine itself.
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

fn nodeIndex(ir: *const Ir, n: u32) ?Ir.Node.Index {
    if (n >= ir.nodes.len) return null;
    return @enumFromInt(n);
}

fn varIndex(ir: *const Ir, v: u32) ?Ir.Variable.Index {
    if (v >= ir.vars.len) return null;
    return @enumFromInt(v);
}

/// The width an already-checked node evaluates at.
fn widthOf(ir: *const Ir, n: Ir.Node.Index) u16 {
    return ir.typeOf(n).width;
}

/// Publish a freshly built node, or map the single way building can fail.
fn emit(out: ?*u32, built: Allocator.Error!Ir.Node.Index) Status {
    const node = built catch return .err_oom;
    if (out) |o| o.* = @intFromEnum(node);
    return .ok;
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
fn nodeSlice(ir: *const Ir, ptr: ?[*]const u32, len: usize) ?[]const Ir.Node.Index {
    const raw = sliceOf(u32, ptr, len) orelse return null;
    for (raw) |n| _ = nodeIndex(ir, n) orelse return null;
    return @ptrCast(raw);
}

/// Like `sliceOf`, but every element must name an existing variable.
fn varSlice(ir: *const Ir, ptr: ?[*]const u32, len: usize) ?[]const Ir.Variable.Index {
    const raw = sliceOf(u32, ptr, len) orelse return null;
    for (raw) |v| _ = varIndex(ir, v) orelse return null;
    return @ptrCast(raw);
}

// -- ABI decoding ------------------------------------------------------------
//
// There is one numbering, not two: `crv_op`, `crv_cast`, `crv_var_kind` and
// `crv_dist_kind` are declared in `crv.h` with the IR's own enum values, so
// nothing here translates. What is left is the check C cannot make — that the
// integer names a member at all, and that it names one this entry point takes.

/// Decode an operator code into the tag it is defined to equal, accepting only
/// the class the calling export is documented to build.
fn tagIn(raw: c_int, class: Ir.Node.Tag.Class) ?Ir.Node.Tag {
    const tag = std.enums.fromInt(Ir.Node.Tag, raw) orelse return null;
    return if (tag.class() == class) tag else null;
}

// -- IR lifetime -------------------------------------------------------------

export fn crv_ir_init(ir: *Ir) void {
    ir.* = .{};
}

export fn crv_ir_deinit(ir: *Ir) void {
    ir.deinit(gpa);
}

export fn crv_ir_reserve(ir: *Ir, nodes: u32, vars: u32, extra: u32) Status {
    ir.nodes.ensureTotalCapacity(gpa, nodes) catch return .err_oom;
    ir.vars.ensureTotalCapacity(gpa, vars) catch return .err_oom;
    ir.extra.ensureTotalCapacity(gpa, extra) catch return .err_oom;
    return .ok;
}

export fn crv_ir_validate(ir: *const Ir) Status {
    ir.validate(gpa) catch |err| return statusOf(err);
    return .ok;
}

export fn crv_ir_node_count(ir: *const Ir) u32 {
    return @intCast(ir.nodes.len);
}

// -- Variables ---------------------------------------------------------------

const VarInfo = extern struct {
    id: u32,
    width: u16,
    kind: u8,
    reserved: u8,
};

export fn crv_var_add(ir: *Ir, id: u32, width: u16, kind: c_int, out: ?*u32) Status {
    setInvalid(out);
    if (width == 0) return .err_invalid_argument;
    const k = std.enums.fromInt(Ir.Variable.Kind, kind) orelse return .err_invalid_argument;

    const v = ir.addVariable(gpa, .{
        .id = @enumFromInt(id),
        .ty = Ir.Type.bit(width),
        .kind = k,
    }) catch return .err_oom;

    if (out) |o| o.* = @intFromEnum(v);
    return .ok;
}

export fn crv_var_count(ir: *const Ir) u32 {
    return @intCast(ir.vars.len);
}

export fn crv_var_get(ir: *const Ir, v: u32, out: ?*VarInfo) Status {
    const index = varIndex(ir, v) orelse return .err_invalid_argument;
    const i = @intFromEnum(index);
    if (out) |o| o.* = .{
        .id = @intFromEnum(ir.vars.items(.id)[i]),
        .width = ir.vars.items(.ty)[i].width,
        .kind = @intFromEnum(ir.vars.items(.kind)[i]),
        .reserved = 0,
    };
    return .ok;
}

// -- Leaves ------------------------------------------------------------------

export fn crv_node_var(ir: *Ir, v: u32, out: ?*u32) Status {
    setInvalid(out);
    const index = varIndex(ir, v) orelse return .err_invalid_argument;
    return emit(out, ir.varRef(gpa, index));
}

export fn crv_node_bool(ir: *Ir, value: c_int, out: ?*u32) Status {
    setInvalid(out);
    return emit(out, ir.boolLit(gpa, value != 0));
}

export fn crv_node_const_u64(ir: *Ir, value: u64, width: u16, out: ?*u32) Status {
    setInvalid(out);
    if (width == 0) return .err_invalid_argument;
    return emit(out, ir.constInt(gpa, value, Ir.Type.bit(width)));
}

export fn crv_node_const_bits(
    ir: *Ir,
    words: ?[*]const u64,
    nwords: usize,
    width: u16,
    out: ?*u32,
) Status {
    setInvalid(out);
    if (width == 0) return .err_invalid_argument;
    const magnitude = sliceOf(u64, words, nwords) orelse return .err_invalid_argument;
    return emit(out, ir.constBits(gpa, magnitude, Ir.Type.bit(width)));
}

// -- Operators ---------------------------------------------------------------

export fn crv_node_unary(ir: *Ir, op: c_int, a: u32, out: ?*u32) Status {
    setInvalid(out);
    const tag = tagIn(op, .unary) orelse return .err_invalid_argument;
    const operand = nodeIndex(ir, a) orelse return .err_invalid_argument;
    return emit(out, ir.unary(gpa, tag, operand));
}

export fn crv_node_binary(ir: *Ir, op: c_int, a: u32, b: u32, out: ?*u32) Status {
    setInvalid(out);
    const tag = tagIn(op, .binary) orelse return .err_invalid_argument;
    const lhs = nodeIndex(ir, a) orelse return .err_invalid_argument;
    const rhs = nodeIndex(ir, b) orelse return .err_invalid_argument;
    if (Ir.requiresEqualWidths(tag) and widthOf(ir, lhs) != widthOf(ir, rhs)) {
        return .err_width_mismatch;
    }
    return emit(out, ir.binary(gpa, tag, lhs, rhs));
}

export fn crv_node_cast(ir: *Ir, cast: c_int, a: u32, width: u16, out: ?*u32) Status {
    setInvalid(out);
    const tag = tagIn(cast, .cast) orelse return .err_invalid_argument;
    const operand = nodeIndex(ir, a) orelse return .err_invalid_argument;
    if (width == 0) return .err_invalid_argument;
    return emit(out, switch (tag) {
        .zext => ir.zext(gpa, operand, width),
        .sext => ir.sext(gpa, operand, width),
        .trunc => ir.trunc(gpa, operand, width),
        else => unreachable, // `.cast` is exactly these three.
    });
}

export fn crv_node_width(ir: *const Ir, n: u32, out_width: ?*u16) Status {
    const node = nodeIndex(ir, n) orelse return .err_invalid_argument;
    if (out_width) |o| o.* = widthOf(ir, node);
    return .ok;
}

// -- Sets, distributions, structural constraints -----------------------------

export fn crv_node_range(ir: *Ir, lo: u32, hi: u32, out: ?*u32) Status {
    setInvalid(out);
    const low = nodeIndex(ir, lo) orelse return .err_invalid_argument;
    const high = nodeIndex(ir, hi) orelse return .err_invalid_argument;
    if (widthOf(ir, low) != widthOf(ir, high)) return .err_width_mismatch;
    return emit(out, ir.range(gpa, low, high));
}

export fn crv_node_in(
    ir: *Ir,
    value: u32,
    members: ?[*]const u32,
    n: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const tested = nodeIndex(ir, value) orelse return .err_invalid_argument;
    const set = nodeSlice(ir, members, n) orelse return .err_invalid_argument;
    const width = widthOf(ir, tested);
    for (set) |m| {
        if (widthOf(ir, m) != width) return .err_width_mismatch;
    }
    return emit(out, ir.in(gpa, tested, set));
}

export fn crv_node_dist_item(
    ir: *Ir,
    kind: c_int,
    value: u32,
    weight: u32,
    out: ?*u32,
) Status {
    setInvalid(out);
    const k = std.enums.fromInt(Ir.DistKind, kind) orelse return .err_invalid_argument;
    const v = nodeIndex(ir, value) orelse return .err_invalid_argument;
    const w = nodeIndex(ir, weight) orelse return .err_invalid_argument;
    return emit(out, ir.distItem(gpa, k, v, w));
}

export fn crv_node_dist(
    ir: *Ir,
    value: u32,
    items: ?[*]const u32,
    n: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const v = nodeIndex(ir, value) orelse return .err_invalid_argument;
    const list = nodeSlice(ir, items, n) orelse return .err_invalid_argument;
    for (list) |item| {
        switch (ir.nodes.items(.tag)[@intFromEnum(item)]) {
            .dist_weight_eq, .dist_weight_div => {},
            else => return .err_invalid_argument,
        }
    }
    return emit(out, ir.dist(gpa, v, list));
}

export fn crv_node_if(ir: *Ir, cond: u32, then_stmt: u32, out: ?*u32) Status {
    setInvalid(out);
    const condition = nodeIndex(ir, cond) orelse return .err_invalid_argument;
    const body = nodeIndex(ir, then_stmt) orelse return .err_invalid_argument;
    return emit(out, ir.ifElse(gpa, condition, body, .null));
}

export fn crv_node_if_else(
    ir: *Ir,
    cond: u32,
    then_stmt: u32,
    else_stmt: u32,
    out: ?*u32,
) Status {
    setInvalid(out);
    const condition = nodeIndex(ir, cond) orelse return .err_invalid_argument;
    const body = nodeIndex(ir, then_stmt) orelse return .err_invalid_argument;
    const alternative = nodeIndex(ir, else_stmt) orelse return .err_invalid_argument;
    return emit(out, ir.ifElse(gpa, condition, body, alternative));
}

export fn crv_node_unique(ir: *Ir, nodes: ?[*]const u32, n: usize, out: ?*u32) Status {
    setInvalid(out);
    const set = nodeSlice(ir, nodes, n) orelse return .err_invalid_argument;
    return emit(out, ir.unique(gpa, set));
}

export fn crv_node_solve_before(
    ir: *Ir,
    before: ?[*]const u32,
    nbefore: usize,
    after: ?[*]const u32,
    nafter: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    const first = varSlice(ir, before, nbefore) orelse return .err_invalid_argument;
    const second = varSlice(ir, after, nafter) orelse return .err_invalid_argument;
    return emit(out, ir.solveBefore(gpa, first, second));
}

// -- Constraints -------------------------------------------------------------

const ConstraintInfo = extern struct {
    id: u32,
    flags: u32,
    stmt_count: u32,
};

const constraint_soft: u32 = 0x1;

export fn crv_constraint_add(
    ir: *Ir,
    id: u32,
    flags: u32,
    stmts: ?[*]const u32,
    n: usize,
    out: ?*u32,
) Status {
    setInvalid(out);
    if (flags & ~constraint_soft != 0) return .err_invalid_argument;
    const body = nodeSlice(ir, stmts, n) orelse return .err_invalid_argument;

    const index = ir.addConstraint(
        gpa,
        @enumFromInt(id),
        .{ .soft = flags & constraint_soft != 0 },
        body,
    ) catch return .err_oom;

    if (out) |o| o.* = @intFromEnum(index);
    return .ok;
}

export fn crv_constraint_count(ir: *const Ir) u32 {
    return @intCast(ir.constraints.len);
}

export fn crv_constraint_get(ir: *const Ir, constraint: u32, out: ?*ConstraintInfo) Status {
    if (constraint >= ir.constraints.len) return .err_invalid_argument;
    const i: usize = constraint;
    if (out) |o| o.* = .{
        .id = @intFromEnum(ir.constraints.items(.id)[i]),
        .flags = if (ir.constraints.items(.flags)[i].soft) constraint_soft else 0,
        .stmt_count = ir.constraints.items(.body)[i].len,
    };
    return .ok;
}

// -- Hashing and caching -----------------------------------------------------

export fn crv_ir_hash(ir: *const Ir, out: ?*Ir.Digest) Status {
    if (out) |o| o.* = ir.hash();
    return .ok;
}

export fn crv_ir_serialized_size(ir: *const Ir) usize {
    return ir.serializedSize();
}

export fn crv_ir_serialize(ir: *const Ir, buf: ?*anyopaque, cap: usize, written: ?*usize) Status {
    const size = ir.serializedSize();
    if (written) |w| w.* = size;
    if (cap < size) return .err_buffer_too_small;
    const dst = buf orelse return .err_invalid_argument;

    const bytes = ir.serialize(gpa) catch return .err_oom;
    defer gpa.free(bytes);
    std.debug.assert(bytes.len == size);
    @memcpy(@as([*]u8, @ptrCast(dst))[0..bytes.len], bytes);
    return .ok;
}

export fn crv_ir_deserialize(buf: *const anyopaque, len: usize, out: *Ir) Status {
    // Initialized up front, so a rejected blob still leaves the caller's
    // storage holding an empty IR that `crv_ir_deinit` accepts.
    out.* = .{};

    out.* = Ir.deserialize(gpa, @as([*]const u8, @ptrCast(buf))[0..len]) catch |err| {
        return statusOf(err);
    };
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

export fn crv_value_words(ir: *const Ir) usize {
    return Solver.valueLimbs(ir);
}

export fn crv_rejection_sampler_new(
    ir: *const Ir,
    options: ?*const RejectionOptions,
    out: *?*CSolver,
) Status {
    out.* = null;

    // The engine indexes the IR's arrays unchecked, so it only ever sees one
    // that has been vetted — even if the caller assembled it by hand.
    ir.validate(gpa) catch |err| return statusOf(err);

    const defaults: RejectionSampler.Options = .{};
    const opts: RejectionSampler.Options = if (options) |o| .{
        .seed = o.seed,
        .max_attempts = if (o.max_attempts == 0) defaults.max_attempts else o.max_attempts,
    } else defaults;

    const engine = RejectionSampler.init(gpa, ir, opts) catch |err| return statusOf(err);

    const solver = gpa.create(CSolver) catch {
        var mutable = engine;
        mutable.deinit(gpa);
        return .err_oom;
    };
    solver.* = .{
        .value_words = Solver.valueLimbs(ir),
        .var_count = ir.vars.len,
        .engine = .{ .rejection = engine },
    };
    out.* = solver;
    return .ok;
}

/// The one export that takes a nullable handle, so that freeing the result of a
/// failed `crv_rejection_sampler_new` is a no-op — the contract `free` has.
export fn crv_solver_free(handle: ?*CSolver) void {
    const s = handle orelse return;
    switch (s.engine) {
        .rejection => |*e| e.deinit(gpa),
    }
    gpa.destroy(s);
}

export fn crv_solver_next(s: *CSolver, values: ?[*]u64, nwords: usize) Status {
    const needed = s.value_words * s.var_count;
    if (nwords < needed) return .err_buffer_too_small;

    var none: [0]Solver.Value = .{};
    const out: []Solver.Value = if (needed == 0)
        &none
    else
        (values orelse return .err_invalid_argument)[0..nwords];

    return if (s.solver().next(out)) .ok else .exhausted;
}

export fn crv_solver_stats(s: *const CSolver, out: ?*Stats) Status {
    if (out) |o| o.* = switch (s.engine) {
        .rejection => |*e| .{ .attempts = e.attempts, .hits = e.hits },
    };
    return .ok;
}

// -- Tests -------------------------------------------------------------------
//
// End-to-end coverage lives in `test/smoke.c`, which exercises the real header
// from real C. These cover the paths a well-behaved C program does not reach.

/// What a C caller writes as `CRV_OP_ADD` or `CRV_CAST_ZEXT`. The header holds
/// the same numbers; `crv_header_values` is what proves it.
fn code(tag: Ir.Node.Tag) c_int {
    return @intFromEnum(tag);
}

fn kindCode(kind: Ir.Variable.Kind) c_int {
    return @intFromEnum(kind);
}

fn shout(comptime name: []const u8) []const u8 {
    comptime {
        var out: [name.len]u8 = undefined;
        for (name, &out) |ch, *o| o.* = std.ascii.toUpper(ch);
        const frozen = out;
        return &frozen;
    }
}

// The claim the whole boundary rests on, checked against the real header
// rather than by eye. Names are built rather than listed, so a tag added to a
// class the C API exposes fails to compile here until `crv.h` declares it.
test "crv.h declares the IR's own enum values" {
    const c = @import("crv.h");

    inline for (comptime std.enums.values(Ir.Node.Tag)) |tag| {
        const prefix = switch (comptime tag.class()) {
            .unary, .binary => "CRV_OP_",
            .cast => "CRV_CAST_",
            // Everything else has a builder of its own, not an op code.
            .leaf, .set, .structural => continue,
        };
        try std.testing.expectEqual(
            @as(i64, @intFromEnum(tag)),
            @as(i64, @field(c, prefix ++ shout(@tagName(tag)))),
        );
    }

    inline for (comptime std.enums.values(Ir.Variable.Kind)) |kind| {
        try std.testing.expectEqual(
            @as(i64, @intFromEnum(kind)),
            @as(i64, @field(c, "CRV_VAR_" ++ shout(@tagName(kind)))),
        );
    }

    inline for (comptime std.enums.values(Ir.DistKind)) |kind| {
        try std.testing.expectEqual(
            @as(i64, @intFromEnum(kind)),
            @as(i64, @field(c, "CRV_DIST_" ++ shout(@tagName(kind)))),
        );
    }
}

test "builders reject bad indices, op codes, and widths" {
    var ir: Ir = undefined;
    crv_ir_init(&ir);
    defer crv_ir_deinit(&ir);

    // An index no node has, an op code that decodes to nothing, and a zero
    // width. A refused builder leaves its out-parameter invalid, never stale.
    var node: u32 = 0;
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(&ir, code(.neg), 7, &node));
    try std.testing.expectEqual(invalid, node);
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(&ir, 999, 0, &node));
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_const_u64(&ir, 1, 0, &node));

    // Class is part of the op: a binary op, or a tag no builder exposes, is
    // not a unary one.
    var a: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(&ir, 1, 8, &a));
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(&ir, code(.add), a, &node));
    try std.testing.expectEqual(Status.err_invalid_argument, crv_node_unary(&ir, code(.var_ref), a, &node));

    // Widths must agree where the evaluator assumes they do.
    var b: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(&ir, 1, 4, &b));
    try std.testing.expectEqual(Status.err_width_mismatch, crv_node_binary(&ir, code(.add), a, b, &node));

    // ... and a cast is how you make them agree.
    var widened: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_cast(&ir, code(.zext), b, 8, &widened));
    try std.testing.expectEqual(Status.ok, crv_node_binary(&ir, code(.add), a, widened, &node));

    var width: u16 = 0;
    try std.testing.expectEqual(Status.ok, crv_node_width(&ir, node, &width));
    try std.testing.expectEqual(@as(u16, 8), width);
}

test "solutions come back through a caller-sized buffer" {
    var ir: Ir = undefined;
    crv_ir_init(&ir);
    defer crv_ir_deinit(&ir);

    var x: u32 = 0;
    var ref: u32 = 0;
    var lit: u32 = 0;
    var eq: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_var_add(&ir, 1, 6, kindCode(.rand), &x));
    try std.testing.expectEqual(Status.ok, crv_node_var(&ir, x, &ref));
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(&ir, 42, 6, &lit));
    try std.testing.expectEqual(Status.ok, crv_node_binary(&ir, code(.eq), ref, lit, &eq));
    const body = [_]u32{eq};
    try std.testing.expectEqual(Status.ok, crv_constraint_add(&ir, 0, 0, &body, 1, null));

    var solver: ?*CSolver = null;
    try std.testing.expectEqual(Status.ok, crv_rejection_sampler_new(&ir, null, &solver));
    defer crv_solver_free(solver);
    const s = solver.?;

    var values: [1]u64 = undefined;
    try std.testing.expectEqual(Status.err_buffer_too_small, crv_solver_next(s, &values, 0));
    try std.testing.expectEqual(Status.ok, crv_solver_next(s, &values, 1));
    try std.testing.expectEqual(@as(u64, 42), values[0]);

    var stats: Stats = undefined;
    try std.testing.expectEqual(Status.ok, crv_solver_stats(s, &stats));
    try std.testing.expectEqual(@as(u64, 1), stats.hits);
}

test "a solver refuses an IR it cannot evaluate" {
    var ir: Ir = undefined;
    crv_ir_init(&ir);
    defer crv_ir_deinit(&ir);

    var x: u32 = 0;
    var ref: u32 = 0;
    var distinct: u32 = 0;
    try std.testing.expectEqual(Status.ok, crv_var_add(&ir, 1, 4, kindCode(.rand), &x));
    try std.testing.expectEqual(Status.ok, crv_node_var(&ir, x, &ref));
    const members = [_]u32{ref};
    try std.testing.expectEqual(Status.ok, crv_node_unique(&ir, &members, 1, &distinct));
    const body = [_]u32{distinct};
    try std.testing.expectEqual(Status.ok, crv_constraint_add(&ir, 0, 0, &body, 1, null));

    var solver: ?*CSolver = null;
    try std.testing.expectEqual(
        Status.err_unsupported_node,
        crv_rejection_sampler_new(&ir, null, &solver),
    );
    try std.testing.expectEqual(@as(?*CSolver, null), solver);
}

test "a corrupt cache blob comes back as a status" {
    var ir: Ir = undefined;
    crv_ir_init(&ir);
    defer crv_ir_deinit(&ir);
    try std.testing.expectEqual(Status.ok, crv_node_const_u64(&ir, 1, 8, null));

    const size = crv_ir_serialized_size(&ir);
    const buf = try std.testing.allocator.alloc(u8, size);
    defer std.testing.allocator.free(buf);

    var written: usize = 0;
    try std.testing.expectEqual(
        Status.err_buffer_too_small,
        crv_ir_serialize(&ir, buf.ptr, size - 1, &written),
    );
    try std.testing.expectEqual(size, written);
    try std.testing.expectEqual(Status.ok, crv_ir_serialize(&ir, buf.ptr, size, &written));

    // A rejected blob still leaves an empty IR behind, so the caller's `defer`
    // stays correct however the call turned out.
    buf[size - 1] ^= 0xff;
    var loaded: Ir = undefined;
    defer crv_ir_deinit(&loaded);
    try std.testing.expectEqual(
        Status.err_checksum_mismatch,
        crv_ir_deserialize(buf.ptr, size, &loaded),
    );
    try std.testing.expectEqual(@as(u32, 0), crv_ir_node_count(&loaded));

    const junk = [_]u8{0} ** 64;
    try std.testing.expectEqual(Status.err_bad_magic, crv_ir_deserialize(&junk, junk.len, &loaded));
    try std.testing.expectEqual(Status.err_truncated, crv_ir_deserialize(buf.ptr, 4, &loaded));
}

test {
    std.testing.refAllDecls(@This());
}

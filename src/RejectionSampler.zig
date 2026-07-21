//! Rejection-sampling constraint solver.
//!
//! The simplest randomizing engine: draw a fresh random value for every
//! variable, evaluate the whole constraint set over that assignment, and accept
//! the first draw that satisfies every constraint (retrying up to
//! `max_attempts`). It is *incomplete* — a `false` result means "no assignment
//! found within the attempt budget", not "unsatisfiable" — but it is simple,
//! embarrassingly fast per attempt, and a solid baseline.
//!
//! Evaluation is strict and exact: `ir.typeOf` gives every node one width, and
//! each operation is evaluated at that width. Types carry no signedness — the
//! operators do (`slt`/`ult`, `sdiv`/`udiv`, `sra`/`srl`, `sext`/`zext`).
//!
//! Storage is two parallel vectors, always allocated: a `u64` `narrow` vector
//! and a `std.math.big.int` `wide` vector. Each node lives in one, chosen by its
//! width — `<= 64` bits narrow, otherwise wide — so a mostly-narrow IR keeps its
//! narrow nodes on the fast integer path even when a few nodes are wide. The
//! evaluator reads each operand from the vector its width selects. The wide
//! vector's ops run over a preallocated limb pool, so the hot loop never
//! allocates.
//!
//! Values are little-endian limb (`u64`) vectors: each variable occupies
//! `Solver.valueLimbs(ir)` limbs in `out` (1 in the common ≤64-bit case, so
//! `out[i]` is just variable `i`'s value).
//!
//! Scope: the scalar expression subset plus `in`, `dist`, `if_else`, `unique`,
//! and `solve_before`. `dist` is enforced as membership (an item's value is
//! allowed when its weight is nonzero); the relative weights bias only the
//! distribution and are not yet honored. `solve_before` is an ordering hint
//! with no effect on a simultaneous draw, so it always holds. `foreach` needs
//! array-typed variables, which the IR does not model yet, so it still panics.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");
const Solver = @import("Solver.zig");
const Value = Solver.Value;

const big = std.math.big.int;
const Limb = std.math.big.Limb;
const Mutable = big.Mutable;
const Const = big.Const;
const Order = std.math.Order;
const Signedness = std.builtin.Signedness;

const RejectionSampler = @This();

ir: *const Ir,
/// Each node's width, resolved once at `init` via `ir.typeOf`.
types: []Ir.Type,
prng: std.Random.DefaultPrng,
max_attempts: u32,
value_limbs: usize,
attempts: u64 = 0,
hits: u64 = 0,

/// Per-variable draw info, indexed by `Ir.Variable.Index`.
vars: []VarInfo,
/// Working copy of the drawn variable values (`value_limbs` per variable),
/// copied to `out` on success.
vbuf: []Limb,
vlen: []usize,

/// Node values live in one of two vectors, chosen by width. Narrow node values
/// (width <= 64) are masked to the node's width; wide values (width > 64) are
/// non-negative bit patterns over the limb pool.
narrow: []u64,
wide: []Limb,
wlen: []usize,
wlimbs: usize,

/// Scratch for the wide big.int operations.
t0: []Limb,
t1: []Limb,
t2: []Limb,
rem: []Limb,
div_buf: []Limb,

const VarInfo = struct { width: u16, used: u32, top_mask: Limb };

pub const Options = struct {
    seed: u64 = 0,
    max_attempts: u32 = 10_000,
};

pub fn init(gpa: Allocator, ir: *const Ir, options: Options) Allocator.Error!RejectionSampler {
    const n = ir.nodes.len;

    const types = try gpa.alloc(Ir.Type, n);
    errdefer gpa.free(types);
    var max_width: u16 = 1;
    for (types, 0..) |*t, i| {
        t.* = ir.typeOf(@enumFromInt(@as(u32, @intCast(i))));
        max_width = @max(max_width, t.width);
    }

    const value_limbs = Solver.valueLimbs(ir);

    const vars = try gpa.alloc(VarInfo, ir.vars.len);
    errdefer gpa.free(vars);
    for (ir.vars.items(.ty), vars) |ty, *vi| {
        const width: u16 = if (ty.width == 0) Ir.default_width else ty.width;
        const used = big.calcTwosCompLimbCount(width);
        const top_bits = width - (used - 1) * @bitSizeOf(Limb);
        vi.* = .{
            .width = width,
            .used = @intCast(used),
            .top_mask = if (top_bits == @bitSizeOf(Limb)) ~@as(Limb, 0) else (@as(Limb, 1) << @intCast(top_bits)) - 1,
        };
    }

    const wlimbs = big.calcTwosCompLimbCount(max_width) + 1;

    var self: RejectionSampler = .{
        .ir = ir,
        .types = types,
        .prng = .init(options.seed),
        .max_attempts = options.max_attempts,
        .value_limbs = value_limbs,
        .vars = vars,
        .vbuf = &.{},
        .vlen = &.{},
        .narrow = &.{},
        .wide = &.{},
        .wlen = &.{},
        .wlimbs = wlimbs,
        .t0 = &.{},
        .t1 = &.{},
        .t2 = &.{},
        .rem = &.{},
        .div_buf = &.{},
    };
    errdefer self.freeBuffers(gpa);

    self.vbuf = try gpa.alloc(Limb, value_limbs * ir.vars.len);
    self.vlen = try gpa.alloc(usize, ir.vars.len);
    self.narrow = try gpa.alloc(u64, n);
    self.wide = try gpa.alloc(Limb, wlimbs * n);
    self.wlen = try gpa.alloc(usize, n);
    self.t0 = try gpa.alloc(Limb, 2 * wlimbs + 2);
    self.t1 = try gpa.alloc(Limb, 2 * wlimbs + 2);
    self.t2 = try gpa.alloc(Limb, 2 * wlimbs + 2);
    self.rem = try gpa.alloc(Limb, wlimbs + 1);
    self.div_buf = try gpa.alloc(Limb, big.calcDivLimbsBufferLen(wlimbs, wlimbs));

    return self;
}

fn freeBuffers(self: *RejectionSampler, gpa: Allocator) void {
    gpa.free(self.vbuf);
    gpa.free(self.vlen);
    gpa.free(self.narrow);
    gpa.free(self.wide);
    gpa.free(self.wlen);
    gpa.free(self.t0);
    gpa.free(self.t1);
    gpa.free(self.t2);
    gpa.free(self.rem);
    gpa.free(self.div_buf);
}

pub fn deinit(self: *RejectionSampler, gpa: Allocator) void {
    gpa.free(self.types);
    gpa.free(self.vars);
    self.freeBuffers(gpa);
}

pub fn solver(self: *RejectionSampler) Solver {
    return .{ .ptr = self, .vtable = &vtable };
}

const vtable: Solver.VTable = .{ .next = nextErased };

fn nextErased(ptr: *anyopaque, out: []Value) bool {
    const self: *RejectionSampler = @ptrCast(@alignCast(ptr));
    return self.next(out);
}

/// Draw one satisfying assignment into `out`, which must hold at least
/// `Solver.valueLimbs(ir) * ir.vars.len` limbs. Returns `true` and fills `out`
/// on success, or `false` if no draw satisfied the constraints within
/// `max_attempts`.
pub fn next(self: *RejectionSampler, out: []Value) bool {
    const tags = self.ir.nodes.items(.tag);
    const datas = self.ir.nodes.items(.data);
    const total = self.value_limbs * self.ir.vars.len;
    std.debug.assert(out.len >= total);

    for (0..self.max_attempts) |tries| {
        self.draw();
        if (self.evaluate(tags, datas)) {
            for (self.vbuf[0..total], out[0..total]) |src, *dst| dst.* = @intCast(src);
            self.attempts += @as(u64, tries) + 1;
            self.hits += 1;
            return true;
        }
    }
    self.attempts += self.max_attempts;
    return false;
}

fn draw(self: *RejectionSampler) void {
    const vl = self.value_limbs;
    for (self.vars, 0..) |vi, v| {
        const region = self.vbuf[v * vl ..][0..vl];
        for (region[0..vi.used]) |*limb| limb.* = @intCast(self.prng.next());
        region[vi.used - 1] &= vi.top_mask;
        for (region[vi.used..]) |*limb| limb.* = 0;
        var len = vi.used;
        while (len > 1 and region[len - 1] == 0) len -= 1;
        self.vlen[v] = len;
    }
}

/// Store an `int_literal` into node `i`'s slot. The literal is `[nwords, ...]`
/// in `extra`; narrow literals read the low words directly, wide ones assemble
/// the full magnitude and truncate it into the wide vector.
fn litVal(self: *RejectionSampler, i: u32, w: u16, d: Ir.Node.Data) void {
    if (w <= 64) {
        self.setVal(i, w, self.litValue(d));
    } else {
        const c = self.litConst(d, self.t1);
        var r = self.wm(i);
        r.truncate(c, .unsigned, w);
        self.storeW(i, r);
    }
}

/// The low 64 bits of the literal at `d` (its top two `u32` words).
inline fn litValue(self: *const RejectionSampler, d: Ir.Node.Data) u64 {
    const e = self.ir.extra.items;
    const n = e[d.lhs];
    const lo: u64 = e[d.lhs + 1];
    const hi: u64 = if (n >= 2) e[d.lhs + 2] else 0;
    return lo | (hi << 32);
}

/// Assemble the full magnitude of the literal at `d` into `buf` (little-endian
/// `u32` words repacked into `Limb`s), returning it as a `Const`.
fn litConst(self: *const RejectionSampler, d: Ir.Node.Data, buf: []Limb) Const {
    const e = self.ir.extra.items;
    const words = e[d.lhs + 1 ..][0..e[d.lhs]];
    const per_limb = @bitSizeOf(Limb) / 32;
    const nlimbs = (words.len + per_limb - 1) / per_limb;
    for (buf[0..nlimbs]) |*l| l.* = 0;
    for (words, 0..) |word, j| {
        buf[j / per_limb] |= @as(Limb, word) << @intCast(32 * (j % per_limb));
    }
    var len = nlimbs;
    while (len > 1 and buf[len - 1] == 0) len -= 1;
    return .{ .limbs = buf[0..len], .positive = true };
}

fn evaluate(self: *RejectionSampler, tags: []const Ir.Node.Tag, datas: []const Ir.Node.Data) bool {
    const ty = self.types;
    for (tags, datas, 0..) |tag, d, iu| {
        const i: u32 = @intCast(iu);
        const w = ty[i].width;
        switch (tag) {
            .int_literal => self.litVal(i, w, d),
            .bool_literal => self.setVal(i, w, d.lhs),
            .var_ref => self.varRef(i, w, d.lhs),

            .zext => self.zext(i, d, w),
            .sext => self.sext(i, d, w),
            .trunc => self.trunc(i, d, w),

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
            => if (w <= 64) {
                self.narrow[i] = self.arithNarrow(tag, d, w);
            } else {
                self.arithWide(i, tag, d, w);
            },

            .eq, .ne, .slt, .ult, .sle, .ule, .sgt, .ugt, .sge, .uge => self.narrow[i] = @intFromBool(self.compare(tag, d)),
            .lnot, .land, .lor, .implies, .iff => self.narrow[i] = @intFromBool(self.logical(tag, d)),

            // Members consumed by their parent `in`/`dist`, not booleans on
            // their own — evaluate to a placeholder.
            .range, .dist_weight_eq, .dist_weight_div => self.setVal(i, w, 0),

            .in => self.narrow[i] = @intFromBool(self.inEval(tags, datas, d)),
            .dist => self.narrow[i] = @intFromBool(self.distEval(tags, datas, d)),
            .unique => self.narrow[i] = @intFromBool(self.uniqueEval(d)),
            .if_else => self.narrow[i] = @intFromBool(self.ifElseEval(d)),
            // An ordering hint: it never changes which assignments are valid,
            // and a simultaneous draw can't honor its distribution, so it holds.
            .solve_before => self.narrow[i] = 1,

            // `foreach` needs array-typed variables, which the IR does not model
            // yet; there is nothing to iterate over.
            .foreach => @panic("RejectionSampler: foreach requires array support (unimplemented)"),
        }
    }

    const extra = self.ir.extra.items;
    for (self.ir.constraints.items(.body)) |body| {
        const stmts = extra[@intFromEnum(body.start)..][0..body.len];
        for (stmts) |st| if (!self.truthy(st)) return false;
    }
    return true;
}

/// True when node `i`'s value is non-zero.
fn truthy(self: *RejectionSampler, i: u32) bool {
    return if (self.types[i].width <= 64) self.narrow[i] != 0 else !self.wc(i).eqlZero();
}

fn setVal(self: *RejectionSampler, i: u32, w: u16, v: u64) void {
    if (w <= 64) {
        self.narrow[i] = maskW(v, w);
    } else {
        var t = tmp(self.t1);
        t.set(v);
        var r = self.wm(i);
        r.truncate(t.toConst(), .unsigned, w);
        self.storeW(i, r);
    }
}

fn varRef(self: *RejectionSampler, i: u32, w: u16, v: u32) void {
    if (w <= 64) {
        self.narrow[i] = @intCast(self.vbuf[v * self.value_limbs]);
    } else {
        var r = self.wm(i);
        r.truncate(self.varConst(v), .unsigned, self.vars[v].width);
        self.storeW(i, r);
    }
}

fn zext(self: *RejectionSampler, i: u32, d: Ir.Node.Data, w: u16) void {
    const aw = self.types[d.lhs].width;
    if (w <= 64) {
        self.narrow[i] = self.narrow[d.lhs]; // value unchanged, just wider
    } else if (aw <= 64) {
        var r = self.wm(i);
        r.set(self.narrow[d.lhs]);
        self.storeW(i, r);
    } else {
        var r = self.wm(i);
        r.copy(self.wc(d.lhs));
        self.storeW(i, r);
    }
}

fn sext(self: *RejectionSampler, i: u32, d: Ir.Node.Data, w: u16) void {
    const aw = self.types[d.lhs].width;
    if (w <= 64) {
        self.narrow[i] = maskW(sextW(self.narrow[d.lhs], aw), w);
        return;
    }
    var signed = tmp(self.t1);
    if (aw <= 64) {
        signed.set(asI64(self.narrow[d.lhs], aw));
    } else {
        signed.truncate(self.wc(d.lhs), .signed, aw);
    }
    var r = self.wm(i);
    r.truncate(signed.toConst(), .unsigned, w);
    self.storeW(i, r);
}

fn trunc(self: *RejectionSampler, i: u32, d: Ir.Node.Data, w: u16) void {
    const aw = self.types[d.lhs].width;
    if (aw <= 64) {
        self.narrow[i] = maskW(self.narrow[d.lhs], w);
    } else if (w <= 64) {
        var t = tmp(self.t1);
        t.truncate(self.wc(d.lhs), .unsigned, w);
        self.narrow[i] = t.toConst().toInt(u64) catch 0;
    } else {
        var r = self.wm(i);
        r.truncate(self.wc(d.lhs), .unsigned, w);
        self.storeW(i, r);
    }
}

fn arithNarrow(self: *RejectionSampler, tag: Ir.Node.Tag, d: Ir.Node.Data, w: u16) u64 {
    const s = self.narrow;
    return switch (tag) {
        .neg => maskW(0 -% s[d.lhs], w),
        .bnot => maskW(~s[d.lhs], w),
        .add => maskW(s[d.lhs] +% s[d.rhs], w),
        .sub => maskW(s[d.lhs] -% s[d.rhs], w),
        .mul => maskW(s[d.lhs] *% s[d.rhs], w),
        .sdiv => sdivmod(s[d.lhs], s[d.rhs], w, .div),
        .udiv => if (s[d.rhs] == 0) 0 else s[d.lhs] / s[d.rhs],
        .smod => sdivmod(s[d.lhs], s[d.rhs], w, .mod),
        .umod => if (s[d.rhs] == 0) 0 else s[d.lhs] % s[d.rhs],
        .band => s[d.lhs] & s[d.rhs],
        .bor => s[d.lhs] | s[d.rhs],
        .bxor => s[d.lhs] ^ s[d.rhs],
        .sll => maskW(shl(s[d.lhs], self.shamt(d.rhs, w)), w),
        .srl => srl(s[d.lhs], self.shamt(d.rhs, w)),
        .sra => sra(s[d.lhs], self.shamt(d.rhs, w), w),
        else => unreachable,
    };
}

fn arithWide(self: *RejectionSampler, i: u32, tag: Ir.Node.Tag, d: Ir.Node.Data, w: u16) void {
    var r = self.wm(i);
    switch (tag) {
        .neg => _ = r.subWrap(zero, self.wc(d.lhs), .unsigned, w),
        .bnot => r.bitNotWrap(self.wc(d.lhs), .unsigned, w),
        .add => _ = r.addWrap(self.wc(d.lhs), self.wc(d.rhs), .unsigned, w),
        .sub => _ = r.subWrap(self.wc(d.lhs), self.wc(d.rhs), .unsigned, w),
        .mul => {
            var t = tmp(self.t0);
            t.mulWrap(self.wc(d.lhs), self.wc(d.rhs), .unsigned, w, &.{}, null);
            r.truncate(t.toConst(), .unsigned, w);
        },
        .sdiv => self.divWide(&r, d, w, .signed, .div),
        .udiv => self.divWide(&r, d, w, .unsigned, .div),
        .smod => self.divWide(&r, d, w, .signed, .mod),
        .umod => self.divWide(&r, d, w, .unsigned, .mod),
        .band => bitWide(&r, self.t0, self.wc(d.lhs), self.wc(d.rhs), .band, w),
        .bor => bitWide(&r, self.t0, self.wc(d.lhs), self.wc(d.rhs), .bor, w),
        .bxor => bitWide(&r, self.t0, self.wc(d.lhs), self.wc(d.rhs), .bxor, w),
        .sll => {
            var t = tmp(self.t0);
            t.shiftLeft(self.wc(d.lhs), self.shamt(d.rhs, w));
            r.truncate(t.toConst(), .unsigned, w);
        },
        .srl => {
            var t = tmp(self.t0);
            t.shiftRight(self.wc(d.lhs), self.shamt(d.rhs, w));
            r.truncate(t.toConst(), .unsigned, w);
        },
        .sra => {
            var v = tmp(self.t1);
            v.truncate(self.wc(d.lhs), .signed, w);
            var t = tmp(self.t0);
            t.shiftRight(v.toConst(), self.shamt(d.rhs, w));
            r.truncate(t.toConst(), .unsigned, w);
        },
        else => unreachable,
    }
    self.storeW(i, r);
}

/// Shift amount: the operand's value, clamped to the result width `w` (a larger
/// shift yields 0/sign after masking and would overflow the shift temporary).
fn shamt(self: *RejectionSampler, i: u32, w: u16) usize {
    if (self.types[i].width <= 64) return @intCast(@min(self.narrow[i], @as(u64, w)));
    var t = tmp(self.t2);
    t.truncate(self.wc(i), .unsigned, self.types[i].width);
    return @min(t.toConst().toInt(usize) catch @as(usize, w), @as(usize, w));
}

fn compare(self: *RejectionSampler, tag: Ir.Node.Tag, d: Ir.Node.Data) bool {
    const aw = self.types[d.lhs].width;
    if (aw <= 64) {
        const a = self.narrow[d.lhs];
        const b = self.narrow[d.rhs];
        return switch (tag) {
            .eq => a == b,
            .ne => a != b,
            .ult => a < b,
            .ule => a <= b,
            .ugt => a > b,
            .uge => a >= b,
            .slt => asI64(a, aw) < asI64(b, aw),
            .sle => asI64(a, aw) <= asI64(b, aw),
            .sgt => asI64(a, aw) > asI64(b, aw),
            .sge => asI64(a, aw) >= asI64(b, aw),
            else => unreachable,
        };
    }
    const a = self.wc(d.lhs);
    const b = self.wc(d.rhs);
    return switch (tag) {
        .eq => a.eql(b),
        .ne => !a.eql(b),
        .ult => a.order(b) == .lt,
        .ule => a.order(b) != .gt,
        .ugt => a.order(b) == .gt,
        .uge => a.order(b) != .lt,
        .slt => self.scmpWide(d, aw) == .lt,
        .sle => self.scmpWide(d, aw) != .gt,
        .sgt => self.scmpWide(d, aw) == .gt,
        .sge => self.scmpWide(d, aw) != .lt,
        else => unreachable,
    };
}

fn scmpWide(self: *RejectionSampler, d: Ir.Node.Data, aw: u16) Order {
    var a = tmp(self.t1);
    a.truncate(self.wc(d.lhs), .signed, aw);
    var b = tmp(self.t2);
    b.truncate(self.wc(d.rhs), .signed, aw);
    return a.toConst().order(b.toConst());
}

fn logical(self: *RejectionSampler, tag: Ir.Node.Tag, d: Ir.Node.Data) bool {
    return switch (tag) {
        .lnot => !self.truthy(d.lhs),
        .land => self.truthy(d.lhs) and self.truthy(d.rhs),
        .lor => self.truthy(d.lhs) or self.truthy(d.rhs),
        .implies => !self.truthy(d.lhs) or self.truthy(d.rhs),
        .iff => self.truthy(d.lhs) == self.truthy(d.rhs),
        else => unreachable,
    };
}

/// `value inside { members... }`: true if `value` hits any member.
fn inEval(self: *RejectionSampler, tags: []const Ir.Node.Tag, datas: []const Ir.Node.Data, d: Ir.Node.Data) bool {
    const members = self.ir.extra.items[d.rhs + 1 ..][0..self.ir.extra.items[d.rhs]];
    for (members) |m| if (self.memberHit(tags, datas, d.lhs, m)) return true;
    return false;
}

/// `value dist { items... }`: true if `value` hits any item with nonzero weight
/// (the relative weights bias distribution only and are not yet honored).
fn distEval(self: *RejectionSampler, tags: []const Ir.Node.Tag, datas: []const Ir.Node.Data, d: Ir.Node.Data) bool {
    const items = self.ir.extra.items[d.rhs + 1 ..][0..self.ir.extra.items[d.rhs]];
    for (items) |it| {
        const item = datas[it]; // dist_weight_*: lhs = value/range, rhs = weight
        if (self.truthy(item.rhs) and self.memberHit(tags, datas, d.lhs, item.lhs)) return true;
    }
    return false;
}

/// `unique { nodes... }`: true if the listed values are pairwise distinct.
fn uniqueEval(self: *RejectionSampler, d: Ir.Node.Data) bool {
    const nodes = self.ir.extra.items[d.lhs + 1 ..][0..self.ir.extra.items[d.lhs]];
    for (nodes, 0..) |a, k| {
        for (nodes[k + 1 ..]) |b| if (self.valEqual(a, b)) return false;
    }
    return true;
}

/// `if (cond) then; else else_node;`: `(cond -> then) and (!cond -> else)`.
fn ifElseEval(self: *RejectionSampler, d: Ir.Node.Data) bool {
    const extra = self.ir.extra.items;
    if (self.truthy(d.lhs)) return self.truthy(extra[d.rhs]);
    const else_node = extra[d.rhs + 1];
    return else_node == @intFromEnum(Ir.Node.Index.null) or self.truthy(else_node);
}

/// True if `value` matches `member`: within it when `member` is a `range`,
/// else equal to it. Widths need not match — the comparison is numeric.
fn memberHit(self: *RejectionSampler, tags: []const Ir.Node.Tag, datas: []const Ir.Node.Data, value: u32, member: u32) bool {
    if (tags[member] == .range) {
        const rd = datas[member];
        return self.valLe(rd.lhs, value) and self.valLe(value, rd.rhs);
    }
    return self.valEqual(value, member);
}

/// Numeric equality of nodes `i` and `j`, regardless of which vector holds them.
fn valEqual(self: *RejectionSampler, i: u32, j: u32) bool {
    if (self.types[i].width <= 64 and self.types[j].width <= 64) return self.narrow[i] == self.narrow[j];
    return self.constOf(i, self.t1).eql(self.constOf(j, self.t2));
}

/// Unsigned `node[i] <= node[j]`, regardless of which vector holds them.
fn valLe(self: *RejectionSampler, i: u32, j: u32) bool {
    if (self.types[i].width <= 64 and self.types[j].width <= 64) return self.narrow[i] <= self.narrow[j];
    return self.constOf(i, self.t1).order(self.constOf(j, self.t2)) != .gt;
}

/// Node `i` as a `Const`, promoting a narrow value into `buf` when needed.
fn constOf(self: *RejectionSampler, i: u32, buf: []Limb) Const {
    if (self.types[i].width > 64) return self.wc(i);
    var m = tmp(buf);
    m.set(self.narrow[i]);
    return m.toConst();
}

fn wc(self: *const RejectionSampler, i: u32) Const {
    return .{ .limbs = self.wide[i * self.wlimbs ..][0..self.wlen[i]], .positive = true };
}

fn wm(self: *RejectionSampler, i: u32) Mutable {
    return .{ .limbs = self.wide[i * self.wlimbs ..][0..self.wlimbs], .len = 1, .positive = true };
}

fn storeW(self: *RejectionSampler, i: u32, m: Mutable) void {
    self.wlen[i] = m.len;
}

fn varConst(self: *const RejectionSampler, v: u32) Const {
    return .{ .limbs = self.vbuf[v * self.value_limbs ..][0..self.vlen[v]], .positive = true };
}

fn tmp(buf: []Limb) Mutable {
    return .{ .limbs = buf, .len = 1, .positive = true };
}

const zero: Const = .{ .limbs = &.{0}, .positive = true };

fn bitWide(r: *Mutable, buf: []Limb, a: Const, b: Const, comptime op: enum { band, bor, bxor }, w: u16) void {
    var t = tmp(buf);
    switch (op) {
        .band => t.bitAnd(a, b),
        .bor => t.bitOr(a, b),
        .bxor => t.bitXor(a, b),
    }
    r.truncate(t.toConst(), .unsigned, w);
}

fn divWide(self: *RejectionSampler, r: *Mutable, d: Ir.Node.Data, w: u16, comptime s: Signedness, comptime op: enum { div, mod }) void {
    var b = tmp(self.t1);
    b.truncate(self.wc(d.rhs), s, w);
    if (b.eqlZero()) {
        r.set(0);
        return;
    }
    var a = tmp(self.t2);
    a.truncate(self.wc(d.lhs), s, w);
    var q = tmp(self.t0);
    var rm = tmp(self.rem);
    Mutable.divTrunc(&q, &rm, a.toConst(), b.toConst(), self.div_buf);
    const result = switch (op) {
        .div => &q,
        .mod => &rm,
    };
    r.truncate(result.toConst(), .unsigned, w);
}

fn maskW(v: u64, w: u16) u64 {
    return if (w >= 64) v else v & ((@as(u64, 1) << @intCast(w)) - 1);
}

fn sextW(v: u64, w: u16) u64 {
    if (w >= 64) return v;
    const m = maskW(v, w);
    return if ((m >> @intCast(w - 1)) & 1 == 1) m | (~@as(u64, 0) << @intCast(w)) else m;
}

fn asI64(v: u64, w: u16) i64 {
    return @bitCast(sextW(v, w));
}

fn shl(v: u64, sh: usize) u64 {
    return if (sh >= 64) 0 else v << @intCast(sh);
}

fn srl(v: u64, sh: usize) u64 {
    return if (sh >= 64) 0 else v >> @intCast(sh);
}

fn sra(a: u64, sh: usize, w: u16) u64 {
    const v = asI64(a, w);
    if (sh >= 64) return maskW(@bitCast(@as(i64, if (v < 0) -1 else 0)), w);
    return maskW(@bitCast(v >> @intCast(sh)), w);
}

fn sdivmod(a: u64, b: u64, w: u16, comptime op: enum { div, mod }) u64 {
    const x = asI64(a, w);
    const y = asI64(b, w);
    if (y == 0) return 0;
    const r: i64 = if (y == -1)
        (if (op == .div) 0 -% x else 0)
    else switch (op) {
        .div => @divTrunc(x, y),
        .mod => @rem(x, y),
    };
    return maskW(@bitCast(r), w);
}

const Type = Ir.Type;

fn constraintOne(gpa: Allocator, ir: *Ir, node: Ir.Node.Index) !void {
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{node});
}

test "in-range constraint" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const membership = try ir.in(gpa, try ir.varRef(gpa, x), &.{
        try ir.range(gpa, try ir.constInt(gpa, 3, Type.bit(4)), try ir.constInt(gpa, 7, Type.bit(4))),
    });
    try constraintOne(gpa, &ir, membership);

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 0x1234 });
    defer sampler.deinit(gpa);

    var out: [1]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[0] >= 3 and out[0] <= 7);
    }
}

test "signed vs unsigned comparison" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const slt = try ir.binary(gpa, .slt, try ir.varRef(gpa, x), try ir.constInt(gpa, 0, Type.bit(8)));
    try constraintOne(gpa, &ir, slt);

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 3 });
    defer sampler.deinit(gpa);

    var out: [1]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(@as(i8, @bitCast(@as(u8, @intCast(out[0])))) < 0);
    }
}

test "mixed widths: zext a 64-bit variable to 128 and compare" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 64-bit x (narrow node); zext to 128 (wide); constraint { ugt(x_128, 1<<63) }.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(64), .kind = .rand });
    const wide = try ir.zext(gpa, try ir.varRef(gpa, x), 128);
    const bound = try ir.zext(gpa, try ir.constInt(gpa, 1 << 63, Type.bit(64)), 128);
    try constraintOne(gpa, &ir, try ir.binary(gpa, .ugt, wide, bound));

    try std.testing.expect(Solver.valueLimbs(&ir) == 1); // widest variable is 64-bit
    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 6 });
    defer sampler.deinit(gpa);

    var out: [1]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[0] > (1 << 63)); // top bit set
    }
}

test "mixed widths: truncate a wide value back to narrow" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 100-bit x; low 8 bits (trunc to 8) must equal 42.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(100), .kind = .rand });
    const low = try ir.trunc(gpa, try ir.varRef(gpa, x), 8);
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, low, try ir.constInt(gpa, 42, Type.bit(8))));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 9 });
    defer sampler.deinit(gpa);

    const vl = Solver.valueLimbs(&ir);
    const out = try gpa.alloc(Value, vl);
    defer gpa.free(out);
    for (0..200) |_| {
        try std.testing.expect(sampler.next(out));
        try std.testing.expectEqual(@as(u64, 42), out[0] & 0xff);
    }
}

test "wide (>64-bit) signed comparison" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(72), .kind = .rand });
    try constraintOne(gpa, &ir, try ir.binary(gpa, .slt, try ir.varRef(gpa, x), try ir.constInt(gpa, 0, Type.bit(72))));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 11 });
    defer sampler.deinit(gpa);

    var out: [2]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[1] >> 7 != 0);
    }
}

test "wide (>64-bit) unsigned modulo" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(100), .kind = .rand });
    const rem = try ir.binary(gpa, .umod, try ir.varRef(gpa, x), try ir.constInt(gpa, 100, Type.bit(100)));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, rem, try ir.constInt(gpa, 7, Type.bit(100))));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 4 });
    defer sampler.deinit(gpa);

    const vl = Solver.valueLimbs(&ir);
    const out = try gpa.alloc(Value, vl);
    defer gpa.free(out);
    for (0..50) |_| {
        try std.testing.expect(sampler.next(out));
        const x_val = (@as(u128, out[1]) << 64) | out[0];
        try std.testing.expectEqual(@as(u128, 7), x_val % 100);
    }
}

test "wide (>64-bit) literal in a constraint" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 128-bit x; constraint { x >= (1 << 100) } — a bound whose only set bit is
    // above word 1, so it exercises the wide-literal storage.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(128), .kind = .rand });
    var limbs = [_]Limb{ 0, 1 << 36 }; // (1 << 36) << 64 == 1 << 100
    const bound = try ir.constBig(gpa, .{ .limbs = &limbs, .positive = true }, Type.bit(128));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .uge, try ir.varRef(gpa, x), bound));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 5 });
    defer sampler.deinit(gpa);

    const vl = Solver.valueLimbs(&ir);
    const out = try gpa.alloc(Value, vl);
    defer gpa.free(out);
    for (0..200) |_| {
        try std.testing.expect(sampler.next(out));
        try std.testing.expect(out[1] >= (1 << 36)); // high limb clears the bound
    }
}

test "dist enforces membership and excludes zero-weight items" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // x dist { 5 := 0, 7 := 2, [20:22] := 1 } — 5 is excluded (weight 0), so the
    // allowed set is {7, 20, 21, 22}.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const x_ref = try ir.varRef(gpa, x);
    const excluded = try ir.distItem(gpa, .dist_weight_eq, try ir.constInt(gpa, 5, Type.bit(8)), try ir.constInt(gpa, 0, Type.bit(8)));
    const single = try ir.distItem(gpa, .dist_weight_eq, try ir.constInt(gpa, 7, Type.bit(8)), try ir.constInt(gpa, 2, Type.bit(8)));
    const span = try ir.distItem(gpa, .dist_weight_div, try ir.range(gpa, try ir.constInt(gpa, 20, Type.bit(8)), try ir.constInt(gpa, 22, Type.bit(8))), try ir.constInt(gpa, 1, Type.bit(8)));
    try constraintOne(gpa, &ir, try ir.dist(gpa, x_ref, &.{ excluded, single, span }));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 2 });
    defer sampler.deinit(gpa);

    var out: [1]Value = undefined;
    for (0..300) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[0] == 7 or (out[0] >= 20 and out[0] <= 22));
    }
}

test "dist membership over a wide variable" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 72-bit x dist { [0 : 1<<71] := 1 } — x must be <= 2^71.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(72), .kind = .rand });
    var hi_limbs = [_]Limb{ 0, 1 << 7 }; // (1 << 7) << 64 == 1 << 71
    const hi = try ir.constBig(gpa, .{ .limbs = &hi_limbs, .positive = true }, Type.bit(72));
    const span = try ir.range(gpa, try ir.constInt(gpa, 0, Type.bit(72)), hi);
    const item = try ir.distItem(gpa, .dist_weight_eq, span, try ir.constInt(gpa, 1, Type.bit(72)));
    try constraintOne(gpa, &ir, try ir.dist(gpa, try ir.varRef(gpa, x), &.{item}));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 8 });
    defer sampler.deinit(gpa);

    var out: [2]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        const x_val = (@as(u128, out[1]) << 64) | out[0];
        try std.testing.expect(x_val <= (@as(u128, 1) << 71));
    }
}

test "unique forces distinct values" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // Three 2-bit variables, all pairwise distinct (a permutation of 3 of {0..3}).
    const a = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(2), .kind = .rand });
    const b = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(2), .kind = .rand });
    const c = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(2), .kind = .rand });
    try constraintOne(gpa, &ir, try ir.unique(gpa, &.{ try ir.varRef(gpa, a), try ir.varRef(gpa, b), try ir.varRef(gpa, c) }));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 7 });
    defer sampler.deinit(gpa);

    var out: [3]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[0] != out[1] and out[0] != out[2] and out[1] != out[2]);
    }
}

test "if_else selects the active branch" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // if (x >= 128) x == 200;  (no else) — accepts x < 128 or x == 200.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const x_ref = try ir.varRef(gpa, x);
    const cond = try ir.binary(gpa, .uge, x_ref, try ir.constInt(gpa, 128, Type.bit(8)));
    const then_c = try ir.binary(gpa, .eq, x_ref, try ir.constInt(gpa, 200, Type.bit(8)));
    try constraintOne(gpa, &ir, try ir.ifElse(gpa, cond, then_c, .null));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 3 });
    defer sampler.deinit(gpa);

    var out: [1]Value = undefined;
    var saw_low = false;
    var saw_high = false;
    for (0..400) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[0] < 128 or out[0] == 200);
        saw_low = saw_low or out[0] < 128;
        saw_high = saw_high or out[0] == 200;
    }
    try std.testing.expect(saw_low and saw_high); // both branches are reachable
}

test "solve_before is a benign ordering hint" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // { a < b; solve a before b; } — the hint must not change the solution set.
    const a = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const b = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(8), .kind = .rand });
    const lt = try ir.binary(gpa, .ult, try ir.varRef(gpa, a), try ir.varRef(gpa, b));
    const order = try ir.solveBefore(gpa, &.{a}, &.{b});
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ lt, order });

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 10 });
    defer sampler.deinit(gpa);

    var out: [2]Value = undefined;
    for (0..200) |_| {
        try std.testing.expect(sampler.next(&out));
        try std.testing.expect(out[0] < out[1]);
    }
}

test "solves through the Solver interface" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const v = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(6), .kind = .rand });
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, try ir.varRef(gpa, v), try ir.constInt(gpa, 42, Type.bit(6))));

    var sampler = try RejectionSampler.init(gpa, &ir, .{ .seed = 1 });
    defer sampler.deinit(gpa);
    const s: Solver = sampler.solver();

    var out: [1]Value = undefined;
    try std.testing.expect(s.next(&out));
    try std.testing.expectEqual(@as(Value, 42), out[0]);
}

test {
    std.testing.refAllDecls(@This());
}

//! Solver-independent static analyses over the IR.
//!
//! Two results, both computed by a single pass over the flat node arrays and
//! both useful to any engine that has to decide *which bits actually matter*:
//!
//!   * **Demanded bits** — for each variable, a mask of the bits some
//!     constraint can observe. A clear bit is **free**: no constraint reads it,
//!     so an engine may draw it straight from its RNG. This matters more than
//!     it looks. A 64-bit variable whose low 8 bits are constrained
//!     (`x & 0xff == 0x42`) contributes 8 bits to a decision procedure instead
//!     of 64, and the other 56 cost nothing. Because the solution set factors
//!     as (assignments to demanded bits) × (free bits), drawing the free bits
//!     independently at random is *exactly* uniform — but only if they are
//!     redrawn per sample rather than read back from one solver model, which
//!     would repeat whatever the solver happened to pick.
//!
//!   * **Defined variables** — variables functionally determined by the
//!     others, found from top-level `v == expr` conjuncts. Two solutions
//!     agreeing on the undefined variables agree everywhere, so a decision
//!     procedure only has to branch on (or hash over) the rest.
//!
//! Both are **sound under-approximations**: the analysis never claims a bit is
//! free when a constraint can see it, and never claims a variable is defined
//! when it is not. Both may miss opportunities — a semantic pass (e.g. an SMT
//! definability check) can refine them further, and is expected to.
//!
//! ## Why a reverse sweep is valid
//!
//! The IR is a flattened tree in dependency order: a node's operands are always
//! at lower indices, because builders construct children before parents. So
//! demand — which flows from constraint roots *down* toward the leaves —
//! propagates correctly in one backward pass, accumulating with `or` where a
//! node feeds several parents.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");

const big = std.math.big.int;
const Limb = std.math.big.Limb;
const limb_bits = @bitSizeOf(Limb);

const Analysis = @This();

/// Bits of each variable that some constraint can observe: `limbs` limbs per
/// variable, indexed by `Ir.Variable.Index`, little-endian. A clear bit is free.
demanded: []Limb,
/// Set when the variable is functionally determined by the variables that are
/// not, and so need not be branched on. Indexed by `Ir.Variable.Index`.
defined: []bool,
/// Limbs per variable mask in `demanded`.
limbs: usize,
/// False when the demand analysis bailed out and conservatively marked every
/// bit demanded (see `Options.max_mask_bytes`, or an unmodeled node tag). The
/// result is still sound, just not informative.
precise: bool,

pub const Options = struct {
    /// Ceiling on the scratch allocation for per-node demand masks. The IR
    /// permits widths up to 65535 bits, so a wide IR with many nodes could ask
    /// for an unreasonable amount; past this the analysis degrades to
    /// "everything demanded" rather than allocating without bound.
    max_mask_bytes: usize = 16 << 20,
};

pub fn deinit(a: *Analysis, gpa: Allocator) void {
    gpa.free(a.demanded);
    gpa.free(a.defined);
    a.* = undefined;
}

/// The mask of observable bits for one variable.
pub fn demandedBits(a: *const Analysis, v: Ir.Variable.Index) []const Limb {
    return a.demanded[@intFromEnum(v) * a.limbs ..][0..a.limbs];
}

/// Whether bit `bit` of variable `v` is free — unobservable by any constraint,
/// and therefore safe to draw at random.
pub fn isFree(a: *const Analysis, v: Ir.Variable.Index, bit: u16) bool {
    return !getBit(a.demandedBits(v), bit);
}

/// How many of `v`'s bits are free.
pub fn freeBitCount(a: *const Analysis, ir: *const Ir, v: Ir.Variable.Index) u16 {
    const width = varWidth(ir, v);
    const mask = a.demandedBits(v);
    var n: u16 = 0;
    for (0..width) |b| {
        if (!getBit(mask, @intCast(b))) n += 1;
    }
    return n;
}

pub fn run(gpa: Allocator, ir: *const Ir, options: Options) Allocator.Error!Analysis {
    var max_width: u16 = 1;
    for (0..ir.nodes.len) |i| {
        max_width = @max(max_width, ir.typeOf(@enumFromInt(@as(u32, @intCast(i)))).width);
    }
    for (ir.vars.items(.ty)) |ty| max_width = @max(max_width, varTypeWidth(ty));
    const limbs = big.calcTwosCompLimbCount(max_width);

    const demanded = try gpa.alloc(Limb, limbs * ir.vars.len);
    errdefer gpa.free(demanded);
    @memset(demanded, 0);

    const defined = try gpa.alloc(bool, ir.vars.len);
    errdefer gpa.free(defined);
    @memset(defined, false);

    var a: Analysis = .{
        .demanded = demanded,
        .defined = defined,
        .limbs = limbs,
        .precise = true,
    };

    // The per-node masks are scratch, not part of the result. Bail to the
    // conservative answer rather than allocating without bound.
    const scratch_bytes = std.math.mul(usize, limbs * ir.nodes.len, @sizeOf(Limb)) catch
        std.math.maxInt(usize);
    if (scratch_bytes > options.max_mask_bytes) {
        a.markAllDemanded(ir);
        return a;
    }

    const node_demand = try gpa.alloc(Limb, limbs * ir.nodes.len);
    defer gpa.free(node_demand);
    @memset(node_demand, 0);

    const scratch = try gpa.alloc(Limb, 2 * limbs);
    defer gpa.free(scratch);

    var w: Worker = .{
        .ir = ir,
        .a = &a,
        .demand = node_demand,
        .limbs = limbs,
        .tmp = scratch[0..limbs],
        .tmp2 = scratch[limbs..][0..limbs],
    };

    w.seedRoots();
    w.sweep();
    if (!a.precise) a.markAllDemanded(ir);

    try a.findDefined(ir, gpa);
    return a;
}

/// The sound fallback: every bit of every variable is observable.
fn markAllDemanded(a: *Analysis, ir: *const Ir) void {
    a.precise = false;
    for (0..ir.vars.len) |v| {
        const mask = a.demanded[v * a.limbs ..][0..a.limbs];
        setLow(mask, varWidth(ir, @enumFromInt(@as(u32, @intCast(v)))));
    }
}

// -- Demanded bits -----------------------------------------------------------

const Worker = struct {
    ir: *const Ir,
    a: *Analysis,
    /// `limbs` limbs of demand per node, indexed by `Ir.Node.Index`.
    demand: []Limb,
    limbs: usize,
    tmp: []Limb,
    tmp2: []Limb,

    fn maskOf(w: *Worker, node: u32) []Limb {
        return w.demand[@as(usize, node) * w.limbs ..][0..w.limbs];
    }

    fn widthOf(w: *const Worker, node: u32) u16 {
        return w.ir.typeOf(@enumFromInt(node)).width;
    }

    /// Demand every bit of `node`.
    fn demandAll(w: *Worker, node: u32) void {
        setLow(w.maskOf(node), w.widthOf(node));
    }

    /// Constraint statements are read for truth, which inspects every bit.
    fn seedRoots(w: *Worker) void {
        const extra = w.ir.extra.items;
        for (w.ir.constraints.items(.body)) |body| {
            for (extra[@intFromEnum(body.start)..][0..body.len]) |stmt| w.demandAll(stmt);
        }
    }

    fn sweep(w: *Worker) void {
        const tags = w.ir.nodes.items(.tag);
        const datas = w.ir.nodes.items(.data);
        var i = w.ir.nodes.len;
        while (i > 0) {
            i -= 1;
            const node: u32 = @intCast(i);
            if (isZero(w.maskOf(node))) continue;
            w.transfer(node, tags[i], datas[i]);
        }
    }

    /// Push node `i`'s demand down onto its operands.
    fn transfer(w: *Worker, i: u32, tag: Ir.Node.Tag, d: Ir.Node.Data) void {
        const width = w.widthOf(i);
        switch (tag) {
            .int_literal, .bool_literal => {},

            .var_ref => {
                const v: usize = d.lhs;
                orInto(w.a.demanded[v * w.limbs ..][0..w.limbs], w.maskOf(i));
            },

            // Bitwise: bit k of the result depends only on bit k of the
            // operands, so demand passes through position-for-position. A
            // literal operand narrows it further: `a & 0xff` cannot observe
            // anything above bit 7 of `a`.
            .bnot => orInto(w.maskOf(d.lhs), w.maskOf(i)),
            .band, .bor, .bxor => w.transferBitwise(i, tag, d),

            // Carries travel toward the most significant end only, so a
            // result bit depends on operand bits at or below it.
            .neg => w.demandBelowTop(d.lhs, w.maskOf(i)),
            .add, .sub, .mul => {
                w.demandBelowTop(d.lhs, w.maskOf(i));
                w.demandBelowTop(d.rhs, w.maskOf(i));
            },

            // Any result bit can depend on any operand bit.
            .sdiv, .udiv, .srem, .urem => {
                w.demandAll(d.lhs);
                w.demandAll(d.rhs);
            },

            .sll, .srl, .sra => w.transferShift(i, tag, d, width),

            .zext => {
                // Bits at or above the operand's width are a constant zero.
                const aw = w.widthOf(d.lhs);
                copy(w.tmp, w.maskOf(i));
                clearFrom(w.tmp, aw);
                orInto(w.maskOf(d.lhs), w.tmp);
            },
            .sext => {
                // Below the sign bit, demand passes through; the sign bit
                // itself is demanded if any replicated copy of it is.
                const aw = w.widthOf(d.lhs);
                copy(w.tmp, w.maskOf(i));
                clearFrom(w.tmp, aw - 1);
                if (anyFrom(w.maskOf(i), aw - 1)) setBit(w.tmp, aw - 1);
                orInto(w.maskOf(d.lhs), w.tmp);
            },
            .trunc => orInto(w.maskOf(d.lhs), w.maskOf(i)), // demand is already narrow

            // A one-bit verdict over the operands' full values.
            .eq, .ne, .slt, .ult, .sle, .ule, .sgt, .ugt, .sge, .uge => {
                w.demandAll(d.lhs);
                w.demandAll(d.rhs);
            },
            .lnot => w.demandAll(d.lhs),
            .land, .lor, .implies, .iff => {
                w.demandAll(d.lhs);
                w.demandAll(d.rhs);
            },

            .range => {
                w.demandAll(d.lhs);
                w.demandAll(d.rhs);
            },
            .in => {
                w.demandAll(d.lhs);
                w.demandAllExtraList(d.rhs);
            },
            .dist => {
                w.demandAll(d.lhs);
                w.demandAllExtraList(d.rhs);
            },
            .dist_weight_eq, .dist_weight_div => {
                w.demandAll(d.lhs);
                w.demandAll(d.rhs);
            },
            .if_else => {
                w.demandAll(d.lhs);
                const extra = w.ir.extra.items;
                for (extra[d.rhs..][0..2]) |branch| {
                    if (@as(Ir.Node.Index, @enumFromInt(branch)) != .null) w.demandAll(branch);
                }
            },
            .unique => w.demandAllExtraList(d.lhs),

            // An ordering hint over variables, not a constraint on values:
            // it makes nothing observable.
            .solve_before => {},

            // Arrays are staked out but not modeled. Rather than guess at the
            // encoding, give up on precision entirely.
            .foreach => w.a.precise = false,
        }
    }

    /// Demand every node in a `[count, node0, ...]` run in `extra`.
    fn demandAllExtraList(w: *Worker, at: u32) void {
        const extra = w.ir.extra.items;
        for (extra[at + 1 ..][0..extra[at]]) |node| w.demandAll(node);
    }

    fn transferBitwise(w: *Worker, i: u32, tag: Ir.Node.Tag, d: Ir.Node.Data) void {
        // `and` with a zero bit, or `or` with a one bit, pins the result
        // regardless of the other operand — so that position is unobservable.
        w.transferBitwiseSide(i, tag, d.lhs, d.rhs);
        w.transferBitwiseSide(i, tag, d.rhs, d.lhs);
    }

    fn transferBitwiseSide(w: *Worker, i: u32, tag: Ir.Node.Tag, target: u32, other: u32) void {
        copy(w.tmp, w.maskOf(i));
        if (tag != .bxor and w.ir.nodes.items(.tag)[other] == .int_literal) {
            w.literalMask(w.tmp2, other);
            if (tag == .bor) notInto(w.tmp2, w.widthOf(other));
            andInto(w.tmp, w.tmp2);
        }
        orInto(w.maskOf(target), w.tmp);
    }

    /// Demand bits `0..=k` of `node`, where `k` is the highest bit set in `m`.
    fn demandBelowTop(w: *Worker, node: u32, m: []const Limb) void {
        const top = highestSetBit(m) orelse return;
        const n = @min(@as(u32, top) + 1, @as(u32, w.widthOf(node)));
        setLow(w.maskOf(node), @intCast(n));
    }

    fn transferShift(w: *Worker, i: u32, tag: Ir.Node.Tag, d: Ir.Node.Data, width: u16) void {
        // A variable shift amount can move any operand bit into any demanded
        // position, so only a literal amount lets us narrow anything.
        const s = w.constShift(d.rhs, width) orelse {
            w.demandAll(d.lhs);
            w.demandAll(d.rhs);
            return;
        };
        w.demandAll(d.rhs); // literal; harmless, and keeps the rule uniform

        switch (tag) {
            .sll => {
                // result bit k comes from operand bit k - s
                copy(w.tmp, w.maskOf(i));
                shrInto(w.tmp, s);
                orInto(w.maskOf(d.lhs), w.tmp);
            },
            .srl => {
                // result bit k comes from operand bit k + s
                copy(w.tmp, w.maskOf(i));
                shlInto(w.tmp, s, width);
                orInto(w.maskOf(d.lhs), w.tmp);
            },
            .sra => {
                // as `srl`, except the vacated high positions all read the
                // sign bit, so demanding any of them demands the sign bit
                copy(w.tmp, w.maskOf(i));
                shlInto(w.tmp, s, width);
                if (width > s and anyFrom(w.maskOf(i), width - s)) setBit(w.tmp, width - 1);
                orInto(w.maskOf(d.lhs), w.tmp);
            },
            else => unreachable,
        }
    }

    /// A shift amount that is a literal, clamped to `width` (any larger amount
    /// shifts everything out, and clamping keeps the mask shifts in range).
    fn constShift(w: *const Worker, node: u32, width: u16) ?u16 {
        if (w.ir.nodes.items(.tag)[node] != .int_literal) return null;
        const e = w.ir.extra.items;
        const d = w.ir.nodes.items(.data)[node];
        // Any word above the first two puts the amount far beyond any width.
        const words = e[d.lhs];
        var j: u32 = 2;
        while (j < words) : (j += 1) {
            if (e[d.lhs + 1 + j] != 0) return width;
        }
        return @intCast(@min(w.ir.intValue(@enumFromInt(node)), @as(u64, width)));
    }

    /// Expand an `int_literal`'s magnitude into a limb mask.
    fn literalMask(w: *const Worker, dst: []Limb, node: u32) void {
        @memset(dst, 0);
        const e = w.ir.extra.items;
        const d = w.ir.nodes.items(.data)[node];
        const words = e[d.lhs + 1 ..][0..e[d.lhs]];
        const per_limb = limb_bits / 32;
        for (words, 0..) |word, j| {
            const limb = j / per_limb;
            if (limb >= dst.len) break;
            dst[limb] |= @as(Limb, word) << @intCast(32 * (j % per_limb));
        }
    }
};

// -- Defined variables -------------------------------------------------------

/// Find variables pinned by a top-level `v == expr`.
///
/// A variable is only accepted once every *candidate* it depends on has been
/// accepted, which both admits chains (`v == u + 1`, `u == w * 2` defines both
/// `v` and `u` in terms of `w`) and rules out cycles (`a == b`, `b == a`
/// defines neither, since neither can go first). State variables are already
/// fixed, so they are neither candidates nor dependencies.
fn findDefined(a: *Analysis, ir: *const Ir, gpa: Allocator) Allocator.Error!void {
    const n = ir.vars.len;
    if (n == 0) return;

    // `def[v]` is the expression pinning `v`, if a top-level equality does.
    const def = try gpa.alloc(Ir.Node.Index, n);
    defer gpa.free(def);
    @memset(def, .null);

    const kinds = ir.vars.items(.kind);
    var stmts: StmtWalker = .{ .ir = ir, .def = def, .kinds = kinds };
    stmts.collect();

    // Fixpoint: accept a candidate once none of its free variables is a
    // still-unaccepted candidate.
    const uses = try gpa.alloc(bool, n);
    defer gpa.free(uses);

    var changed = true;
    while (changed) {
        changed = false;
        for (0..n) |v| {
            if (a.defined[v] or def[v] == .null) continue;
            @memset(uses, false);
            collectVars(ir, def[v], uses);
            if (uses[v]) continue; // self-referential; pins nothing
            var ready = true;
            for (0..n) |u| {
                if (uses[u] and !a.defined[u] and def[u] != .null) {
                    ready = false;
                    break;
                }
            }
            if (!ready) continue;
            a.defined[v] = true;
            changed = true;
        }
    }
}

const StmtWalker = struct {
    ir: *const Ir,
    def: []Ir.Node.Index,
    kinds: []const Ir.Variable.Kind,

    fn collect(s: *StmtWalker) void {
        const extra = s.ir.extra.items;
        for (s.ir.constraints.items(.body), s.ir.constraints.items(.flags)) |body, flags| {
            // A soft constraint may be dropped, so it cannot pin anything.
            if (flags.soft) continue;
            for (extra[@intFromEnum(body.start)..][0..body.len]) |stmt| s.visit(stmt);
        }
    }

    /// Walk the conjunctive spine. `land` is the only connective that keeps a
    /// child unconditionally true; `lor`, `implies`, and `if_else` do not.
    fn visit(s: *StmtWalker, node: u32) void {
        const tag = s.ir.nodes.items(.tag)[node];
        const d = s.ir.nodes.items(.data)[node];
        switch (tag) {
            .land => {
                s.visit(d.lhs);
                s.visit(d.rhs);
            },
            .eq => {
                s.record(d.lhs, d.rhs);
                s.record(d.rhs, d.lhs);
            },
            else => {},
        }
    }

    /// Record `expr` as pinning `node`, when `node` is a randomized variable
    /// that nothing has pinned yet.
    fn record(s: *StmtWalker, node: u32, expr: u32) void {
        if (s.ir.nodes.items(.tag)[node] != .var_ref) return;
        const v = s.ir.nodes.items(.data)[node].lhs;
        if (s.kinds[v] == .state) return;
        if (s.def[v] != .null) return;
        s.def[v] = @enumFromInt(expr);
    }
};

/// Mark every variable reachable from `node`.
fn collectVars(ir: *const Ir, node: Ir.Node.Index, out: []bool) void {
    if (node == .null) return;
    const i = @intFromEnum(node);
    const d = ir.nodes.items(.data)[i];
    const extra = ir.extra.items;
    switch (ir.nodes.items(.tag)[i]) {
        .int_literal, .bool_literal => {},
        .var_ref => out[d.lhs] = true,

        .neg, .bnot, .lnot, .zext, .sext, .trunc => collectVars(ir, @enumFromInt(d.lhs), out),

        .add,
        .sub,
        .mul,
        .sdiv,
        .udiv,
        .srem,
        .urem,
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
        .range,
        .dist_weight_eq,
        .dist_weight_div,
        => {
            collectVars(ir, @enumFromInt(d.lhs), out);
            collectVars(ir, @enumFromInt(d.rhs), out);
        },

        .in, .dist => {
            collectVars(ir, @enumFromInt(d.lhs), out);
            for (extra[d.rhs + 1 ..][0..extra[d.rhs]]) |m| collectVars(ir, @enumFromInt(m), out);
        },
        .unique => {
            for (extra[d.lhs + 1 ..][0..extra[d.lhs]]) |m| collectVars(ir, @enumFromInt(m), out);
        },
        .if_else => {
            collectVars(ir, @enumFromInt(d.lhs), out);
            for (extra[d.rhs..][0..2]) |branch| collectVars(ir, @enumFromInt(branch), out);
        },

        // Conservative: assume they can involve anything, so no equality
        // sitting above one of these gets mistaken for a definition.
        .solve_before, .foreach => @memset(out, true),
    }
}

// -- Width helpers -----------------------------------------------------------

fn varTypeWidth(ty: Ir.Type) u16 {
    return if (ty.width == 0) Ir.default_width else ty.width;
}

fn varWidth(ir: *const Ir, v: Ir.Variable.Index) u16 {
    return varTypeWidth(ir.vars.items(.ty)[@intFromEnum(v)]);
}

// -- Bit-mask primitives -----------------------------------------------------
//
// Masks are little-endian limb vectors of a fixed length, holding an unsigned
// bit set. Widths reach 65535 bits, so these never assume a mask fits a word.

fn getBit(m: []const Limb, bit: u16) bool {
    const limb = @as(usize, bit) / limb_bits;
    if (limb >= m.len) return false;
    return (m[limb] >> @intCast(bit % limb_bits)) & 1 != 0;
}

fn setBit(m: []Limb, bit: u16) void {
    const limb = @as(usize, bit) / limb_bits;
    if (limb >= m.len) return;
    m[limb] |= @as(Limb, 1) << @intCast(bit % limb_bits);
}

/// Set bits `0..n`.
fn setLow(m: []Limb, n: u16) void {
    const full = @as(usize, n) / limb_bits;
    for (m[0..@min(full, m.len)]) |*l| l.* = ~@as(Limb, 0);
    const rest: u6 = @intCast(n % limb_bits);
    if (rest != 0 and full < m.len) m[full] |= (@as(Limb, 1) << rest) - 1;
}

/// Clear bits `n..`.
fn clearFrom(m: []Limb, n: u16) void {
    const full = @as(usize, n) / limb_bits;
    const rest: u6 = @intCast(n % limb_bits);
    if (full < m.len) {
        m[full] &= if (rest == 0) 0 else (@as(Limb, 1) << rest) - 1;
        for (m[full + 1 ..]) |*l| l.* = 0;
    }
}

/// Whether any bit at or above `n` is set.
fn anyFrom(m: []const Limb, n: u16) bool {
    const full = @as(usize, n) / limb_bits;
    if (full >= m.len) return false;
    const rest: u6 = @intCast(n % limb_bits);
    if (m[full] >> rest != 0) return true;
    for (m[full + 1 ..]) |l| {
        if (l != 0) return true;
    }
    return false;
}

fn isZero(m: []const Limb) bool {
    for (m) |l| {
        if (l != 0) return false;
    }
    return true;
}

fn highestSetBit(m: []const Limb) ?u16 {
    var i = m.len;
    while (i > 0) {
        i -= 1;
        if (m[i] != 0) {
            const lead: usize = @clz(m[i]);
            return @intCast(i * limb_bits + (limb_bits - 1 - lead));
        }
    }
    return null;
}

fn copy(dst: []Limb, src: []const Limb) void {
    @memcpy(dst, src);
}

fn clear(m: []Limb) void {
    @memset(m, 0);
}

fn orInto(dst: []Limb, src: []const Limb) void {
    for (dst, src) |*d, s| d.* |= s;
}

fn andInto(dst: []Limb, src: []const Limb) void {
    for (dst, src) |*d, s| d.* &= s;
}

/// Complement within `width` bits.
fn notInto(m: []Limb, width: u16) void {
    for (m) |*l| l.* = ~l.*;
    clearFrom(m, width);
}

/// Shift right in place by `s` bits.
fn shrInto(m: []Limb, s: u16) void {
    if (s == 0) return;
    const words = @as(usize, s) / limb_bits;
    const bits: u6 = @intCast(s % limb_bits);
    if (words >= m.len) {
        clear(m);
        return;
    }
    var i: usize = 0;
    while (i + words < m.len) : (i += 1) {
        var v = m[i + words] >> bits;
        if (bits != 0 and i + words + 1 < m.len) {
            v |= m[i + words + 1] << @intCast(limb_bits - @as(u32, bits));
        }
        m[i] = v;
    }
    while (i < m.len) : (i += 1) m[i] = 0;
}

/// Shift left in place by `s` bits, discarding anything at or above `width`.
fn shlInto(m: []Limb, s: u16, width: u16) void {
    if (s != 0) {
        const words = @as(usize, s) / limb_bits;
        const bits: u6 = @intCast(s % limb_bits);
        if (words >= m.len) {
            clear(m);
            return;
        }
        var i = m.len;
        while (i > words) {
            i -= 1;
            var v = m[i - words] << bits;
            if (bits != 0 and i - words >= 1) {
                v |= m[i - words - 1] >> @intCast(limb_bits - @as(u32, bits));
            }
            m[i] = v;
        }
        while (i > 0) {
            i -= 1;
            m[i] = 0;
        }
    }
    clearFrom(m, width);
}

// -- Tests -------------------------------------------------------------------

const Type = Ir.Type;

fn constraintOne(gpa: Allocator, ir: *Ir, node: Ir.Node.Index) !void {
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{node});
}

test "masking frees the bits it clears" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 32-bit x; constraint { (x & 0xff) == 0x42 }.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(32), .kind = .rand });
    const low = try ir.binary(gpa, .band, try ir.varRef(gpa, x), try ir.constInt(gpa, 0xff, Type.bit(32)));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, low, try ir.constInt(gpa, 0x42, Type.bit(32))));

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expect(a.precise);
    // Only the low byte is observable; the top 24 bits are free.
    try std.testing.expectEqual(@as(u16, 24), a.freeBitCount(&ir, x));
    for (0..8) |b| try std.testing.expect(!a.isFree(x, @intCast(b)));
    for (8..32) |b| try std.testing.expect(a.isFree(x, @intCast(b)));
}

test "truncation frees the bits it drops" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 100-bit x; constraint { trunc(x, 8) == 42 } -- the wide path.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(100), .kind = .rand });
    const low = try ir.trunc(gpa, try ir.varRef(gpa, x), 8);
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, low, try ir.constInt(gpa, 42, Type.bit(8))));

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expectEqual(@as(u16, 92), a.freeBitCount(&ir, x));
    try std.testing.expect(!a.isFree(x, 7));
    try std.testing.expect(a.isFree(x, 8));
    try std.testing.expect(a.isFree(x, 99));
}

test "a shifted mask moves the demanded window" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 32-bit x; constraint { ((x >> 8) & 0xf) == 3 } -- observes bits 8..11.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(32), .kind = .rand });
    const shifted = try ir.binary(gpa, .srl, try ir.varRef(gpa, x), try ir.constInt(gpa, 8, Type.bit(32)));
    const field = try ir.binary(gpa, .band, shifted, try ir.constInt(gpa, 0xf, Type.bit(32)));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, field, try ir.constInt(gpa, 3, Type.bit(32))));

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    for (0..8) |b| try std.testing.expect(a.isFree(x, @intCast(b)));
    for (8..12) |b| try std.testing.expect(!a.isFree(x, @intCast(b)));
    for (12..32) |b| try std.testing.expect(a.isFree(x, @intCast(b)));
}

test "an unconstrained variable is entirely free" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(16), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(16), .kind = .rand });
    try constraintOne(gpa, &ir, try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.constInt(gpa, 100, Type.bit(16))));

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expectEqual(@as(u16, 0), a.freeBitCount(&ir, x));
    try std.testing.expectEqual(@as(u16, 16), a.freeBitCount(&ir, y));
}

test "arithmetic demands everything at or below the demanded bits" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // 32-bit x, y; constraint { ((x + y) & 0xff) == 0 }: carries only travel
    // up, so bits 8.. of the operands cannot affect the low byte.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(32), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(32), .kind = .rand });
    const sum = try ir.binary(gpa, .add, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    const low = try ir.binary(gpa, .band, sum, try ir.constInt(gpa, 0xff, Type.bit(32)));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, low, try ir.constInt(gpa, 0, Type.bit(32))));

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expectEqual(@as(u16, 24), a.freeBitCount(&ir, x));
    try std.testing.expectEqual(@as(u16, 24), a.freeBitCount(&ir, y));
}

test "a top-level equality defines a variable, through a chain" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // w free; u == w * 2; v == u + 1.
    const w = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const u = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(8), .kind = .rand });
    const v = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(8), .kind = .rand });

    const u_def = try ir.binary(gpa, .eq, try ir.varRef(gpa, u), try ir.binary(
        gpa,
        .mul,
        try ir.varRef(gpa, w),
        try ir.constInt(gpa, 2, Type.bit(8)),
    ));
    const v_def = try ir.binary(gpa, .eq, try ir.varRef(gpa, v), try ir.binary(
        gpa,
        .add,
        try ir.varRef(gpa, u),
        try ir.constInt(gpa, 1, Type.bit(8)),
    ));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ u_def, v_def });

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expect(!a.defined[@intFromEnum(w)]);
    try std.testing.expect(a.defined[@intFromEnum(u)]);
    try std.testing.expect(a.defined[@intFromEnum(v)]);
}

test "mutually referential equalities define nothing" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // a == b; b == a. Neither can be ordered first, so neither is pinned.
    const p = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const q = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(8), .kind = .rand });
    const e1 = try ir.binary(gpa, .eq, try ir.varRef(gpa, p), try ir.varRef(gpa, q));
    const e2 = try ir.binary(gpa, .eq, try ir.varRef(gpa, q), try ir.varRef(gpa, p));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ e1, e2 });

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    // Each is a candidate pinned only by the other, so neither can be ordered
    // first and neither is accepted. Sound, if not maximal: `p` really is a
    // function of `q`, but claiming both would be circular.
    try std.testing.expect(!a.defined[@intFromEnum(p)]);
    try std.testing.expect(!a.defined[@intFromEnum(q)]);
}

test "a soft constraint pins nothing" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const eq = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 7, Type.bit(8)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{ .soft = true }, &.{eq});

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expect(!a.defined[@intFromEnum(x)]);
    // It is still observable: a soft constraint that survives does constrain.
    try std.testing.expectEqual(@as(u16, 0), a.freeBitCount(&ir, x));
}

test "disjunction is not a definition site" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // (x == 1) || (x == 2) pins nothing, even though both arms are equalities.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const a1 = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 1, Type.bit(8)));
    const a2 = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 2, Type.bit(8)));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .lor, a1, a2));

    var a = try Analysis.run(gpa, &ir, .{});
    defer a.deinit(gpa);

    try std.testing.expect(!a.defined[@intFromEnum(x)]);
}

test {
    std.testing.refAllDecls(@This());
}

//! Universal hashing over the independent support: counting (ApproxMC) and
//! cell enumeration, which together give almost-uniform sampling.
//!
//! The idea is to sample from a set you cannot enumerate by chopping it into
//! random cells of roughly equal size and enumerating one cell. A random parity
//! constraint over a subset of the support bits — each bit included with
//! probability p, against a random parity — keeps each solution with
//! probability 1/2, and, crucially, keeps any *pair* of solutions
//! independently. That pairwise independence is what bounds the variance of a
//! cell's size regardless of how the solutions are distributed, so uniformity
//! comes from the hash family rather than from anything the SAT solver does.
//! Solver bias — the thing that makes "call an SMT solver with a fresh seed"
//! a bad sampler — never enters the argument.
//!
//! Counts are kept as `mantissa * 2^exp`, never as a big integer: a cell size
//! is bounded by the pivot (a small constant) and the exponent is just the
//! number of parity constraints, so the pair is exact and fits in two words.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("../Ir.zig");
const Encoder = @import("Encoder.zig");
const Support = @import("Support.zig");
const bw = @import("bitwuzla");

const Limb = std.math.big.Limb;
const limb_bits = @bitSizeOf(Limb);

/// A solution count as `mantissa * 2^exp`. Exact — the mantissa is a cell size
/// and the exponent is a parity-constraint count — and free of any big-integer
/// arithmetic, which matters because the support can run to thousands of bits.
pub const Count = struct {
    mantissa: u64,
    exp: u16,

    pub const zero: Count = .{ .mantissa = 0, .exp = 0 };

    /// `floor(log2(value))`, or null for a zero count.
    pub fn log2Floor(c: Count) ?u32 {
        if (c.mantissa == 0) return null;
        const lead: u32 = @clz(c.mantissa);
        return @as(u32, c.exp) + (63 - lead);
    }

    fn normalized(c: Count) struct { m: u64, e: i64 } {
        if (c.mantissa == 0) return .{ .m = 0, .e = std.math.minInt(i64) };
        const lead: u6 = @intCast(@clz(c.mantissa));
        return .{ .m = c.mantissa << lead, .e = @as(i64, c.exp) - lead };
    }

    pub fn order(a: Count, b: Count) std.math.Order {
        const na = a.normalized();
        const nb = b.normalized();
        if (na.e != nb.e) return std.math.order(na.e, nb.e);
        return std.math.order(na.m, nb.m);
    }
};

/// How an enumeration finished, which is the difference between "the cell holds
/// exactly this many" and "the cell holds at least this many".
pub const Status = enum {
    /// The solver proved there are no further solutions: the count is exact.
    exhausted,
    /// The caller's limit was reached first: the count is a lower bound.
    truncated,
    /// The solver gave up (budget, termination callback). Nothing is proven.
    unknown,
};

pub const Enumeration = struct {
    found: usize,
    status: Status,
};

/// Everything the hashing routines need to talk to one solver session.
pub const Ctx = struct {
    ir: *const Ir,
    enc: *const Encoder,
    session: bw.Session,
    tm: bw.TermManager,
    support: []const Support.Bit,
    /// Limbs per variable in a solution buffer.
    value_limbs: usize,
    /// Limbs per whole solution (`value_limbs * ir.vars.len`).
    stride: usize,
    /// Scratch of `support.len` terms, for building concatenations.
    terms: []bw.Term,
    bv1: bw.Sort,

    /// Copy the current model into `out`, one variable's value per
    /// `value_limbs` limbs.
    pub fn readModel(ctx: *const Ctx, out: []Limb) void {
        @memset(out, 0);
        for (0..ctx.ir.vars.len) |vi| {
            const v: Ir.Variable.Index = @enumFromInt(@as(u32, @intCast(vi)));
            const value = ctx.session.getValue(ctx.enc.varTerm(v));
            bw.readBits(value, out[vi * ctx.value_limbs ..][0..ctx.value_limbs], ctx.enc.varWidth(v));
        }
    }

    /// Forbid every solution whose support bits match `sol`'s.
    ///
    /// Blocking the *projection* rather than the whole assignment is what makes
    /// the enumeration count distinct support assignments, which is the
    /// quantity the hashing argument is about. Blocking full assignments would
    /// count free-bit permutations too and blow the cell budget instantly.
    pub fn blockProjection(ctx: *const Ctx, sol: []const Limb) void {
        if (ctx.support.len == 0) {
            // Nothing to distinguish solutions by: there is exactly one.
            ctx.session.assert(ctx.tm.mkFalse());
            return;
        }
        var clause: ?bw.Term = null;
        for (ctx.support) |b| {
            const actual = ctx.bitOf(sol, b);
            const lit = ctx.tm.term2(
                .distinct,
                ctx.tm.extract(ctx.enc.varTerm(b.v), b.bit, b.bit),
                ctx.tm.bvValueU64(ctx.bv1, actual),
            );
            clause = if (clause) |prev| ctx.tm.term2(.or_, prev, lit) else lit;
        }
        ctx.session.assert(clause.?);
    }

    fn bitOf(ctx: *const Ctx, sol: []const Limb, b: Support.Bit) u64 {
        const base = @intFromEnum(b.v) * ctx.value_limbs;
        const limb = base + @as(usize, b.bit) / limb_bits;
        return (sol[limb] >> @intCast(b.bit % limb_bits)) & 1;
    }
};

/// Enumerate up to `limit` distinct support assignments under the current
/// assertion stack, writing each into `out`.
///
/// Blocking clauses accumulate on the stack, so callers wrap this in a
/// `push`/`pop` pair — which also discards the parity constraints for the cell.
pub fn enumerate(ctx: *const Ctx, limit: usize, out: []Limb) Enumeration {
    var found: usize = 0;
    while (found < limit) {
        switch (ctx.session.checkSat()) {
            .unsat => return .{ .found = found, .status = .exhausted },
            .unknown => return .{ .found = found, .status = .unknown },
            .sat => {},
        }
        const slot = out[found * ctx.stride ..][0..ctx.stride];
        ctx.readModel(slot);
        ctx.blockProjection(slot);
        found += 1;
    }
    return .{ .found = found, .status = .truncated };
}

/// Assert `m` independent random parity constraints over the support.
///
/// Each is `xor(selected bits) == parity`, built as a reduction over a
/// concatenation so that Bitwuzla's rewriter owns the chaining. `density` is
/// the per-bit inclusion probability: 0.5 gives the 2-universal family the
/// guarantees are stated for, and lower values trade some of that for shorter,
/// cheaper constraints.
pub fn addXors(ctx: *const Ctx, rand: std.Random, m: u16, density: f64) void {
    var added: u16 = 0;
    while (added < m) {
        var k: usize = 0;
        for (ctx.support) |b| {
            if (rand.float(f64) >= density) continue;
            ctx.terms[k] = ctx.tm.extract(ctx.enc.varTerm(b.v), b.bit, b.bit);
            k += 1;
        }
        // An empty selection constrains nothing (or everything, if the parity
        // is 1). Redraw rather than waste a level.
        if (k == 0) continue;

        const joined = if (k == 1) ctx.terms[0] else ctx.tm.termN(.bv_concat, ctx.terms[0..k]);
        const parity: u64 = rand.int(u1);
        ctx.session.assert(ctx.tm.term2(
            .equal,
            ctx.tm.term1(.bv_redxor, joined),
            ctx.tm.bvValueU64(ctx.bv1, parity),
        ));
        added += 1;
    }
}

/// One cell: push a fresh hash of `m` constraints, enumerate it, pop.
pub fn boundedSat(
    ctx: *const Ctx,
    rand: std.Random,
    m: u16,
    density: f64,
    limit: usize,
    out: []Limb,
) Enumeration {
    ctx.session.push(1);
    defer ctx.session.pop(1);
    if (m > 0) addXors(ctx, rand, m, density);
    return enumerate(ctx, limit, out);
}

// -- Counting ----------------------------------------------------------------

pub const Params = struct {
    /// Cell-size threshold, derived from epsilon (see `pivotFor`).
    pivot: u32,
    /// How far the zero-hash step will enumerate before giving up on counting
    /// the set outright. Independent of `pivot`: a solution set small enough to
    /// list is worth listing even when that takes far more than a cell's worth
    /// of models, because the payoff is exact uniformity forever after.
    exact_limit: u32,
    density: f64,
};

/// ApproxMC's cell-size threshold for a given epsilon:
/// `ceil(9.84 (1 + eps/(1+eps)) (1 + 1/eps)^2)` — 72 at the default eps = 0.8.
pub fn pivotFor(epsilon: f64) u32 {
    const a = 1.0 + epsilon / (1.0 + epsilon);
    const b = 1.0 + 1.0 / epsilon;
    return @intFromFloat(@ceil(9.84 * a * b * b));
}

/// Rounds needed for confidence `1 - delta`: `ceil(17 log2(3/delta))`.
pub fn roundsFor(delta: f64) u32 {
    return @intFromFloat(@ceil(17.0 * @log2(3.0 / delta)));
}

/// The confidence actually delivered by `rounds` rounds — the inverse of
/// `roundsFor`, so a capped run can report what it really proved instead of
/// what was asked for.
///
/// Clamped at 1.0, and that clamp is the common case: the bound only becomes
/// non-vacuous past 27 rounds, so the default cap of 17 proves *nothing* and
/// must say so rather than report a "probability" above one. Measured accuracy
/// at 17 rounds is far better than the theory bounds — estimates land inside
/// the epsilon band every time in testing — but "unproven and good in practice"
/// is a different claim from "proven", and only the latter is a guarantee.
pub fn deltaFor(rounds: u32) f64 {
    const raw = 3.0 / std.math.pow(f64, 2.0, @as(f64, @floatFromInt(rounds)) / 17.0);
    return @min(1.0, raw);
}

pub const CountResult = union(enum) {
    /// The solution set was small enough to enumerate outright, and `out`
    /// holds all of it. Sampling from here is exactly uniform.
    exact: usize,
    /// An estimate within the requested factor, with the number of parity
    /// constraints that produced it.
    approx: Count,
    /// The solver could not resolve the instance within budget.
    indeterminate,
    /// Proven to have no solutions at all.
    unsat,
};

/// Estimate the number of distinct support assignments.
///
/// The zero-hash step is exhaustive enumeration, so a small solution set falls
/// out of the counting phase already enumerated — the "just enumerate it" case
/// needs no separate code path, and when it fires the result is exact rather
/// than approximate.
pub fn count(
    ctx: *const Ctx,
    rand: std.Random,
    params: Params,
    scratch: []Limb,
    estimates: []Count,
) CountResult {
    ctx.session.push(1);
    const base = enumerate(ctx, params.exact_limit, scratch);
    ctx.session.pop(1);
    switch (base.status) {
        .unknown => return .indeterminate,
        .exhausted => return if (base.found == 0) .unsat else .{ .exact = base.found },
        .truncated => {},
    }

    const nbits: u16 = @intCast(@min(ctx.support.len, std.math.maxInt(u16)));
    var previous: u16 = 1;
    var taken: usize = 0;

    for (0..estimates.len) |_| {
        // A round that cannot resolve is dropped rather than recorded, so the
        // median is taken over the rounds that actually produced an estimate.
        const found = logSatSearch(ctx, rand, params, previous, nbits, scratch) orelse continue;
        previous = found.m;
        estimates[taken] = .{ .mantissa = found.cell, .exp = found.m };
        taken += 1;
    }
    if (taken == 0) return .indeterminate;

    const sample = estimates[0..taken];
    std.mem.sort(Count, sample, {}, lessThan);
    return .{ .approx = sample[sample.len / 2] };
}

fn lessThan(_: void, a: Count, b: Count) bool {
    return a.order(b) == .lt;
}

const Found = struct { m: u16, cell: u64 };

/// Find the fewest parity constraints whose cell holds at most `pivot`
/// solutions, then report that cell's size.
///
/// Cell size falls as constraints are added, so this is a binary search. It is
/// warm-started from the previous round's answer because `log2|S|` does not
/// move between rounds of the same instance — the probe usually lands on the
/// right side immediately and halves the remaining interval.
fn logSatSearch(
    ctx: *const Ctx,
    rand: std.Random,
    params: Params,
    warm: u16,
    nbits: u16,
    scratch: []Limb,
) ?Found {
    const limit = params.pivot + 1;
    var lo: u16 = 0;
    var hi: u16 = nbits;

    const start = std.math.clamp(warm, 1, @max(nbits, 1));
    const probe = boundedSat(ctx, rand, start, params.density, limit, scratch);
    switch (probe.status) {
        .unknown => return null,
        .truncated => lo = start + 1,
        .exhausted => hi = start,
    }

    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = boundedSat(ctx, rand, mid, params.density, limit, scratch);
        switch (r.status) {
            .unknown => return null,
            .truncated => lo = mid + 1,
            .exhausted => hi = mid,
        }
    }
    if (lo > nbits) return null;

    const final = boundedSat(ctx, rand, lo, params.density, limit, scratch);
    if (final.status != .exhausted) return null;
    return .{ .m = lo, .cell = final.found };
}

/// The hash width to aim at so that a cell holds roughly `target` solutions.
pub fn aimFor(c: Count, target: u32) u16 {
    const total = c.log2Floor() orelse return 0;
    const lead: u32 = @clz(@as(u64, @max(target, 1)));
    const want: u32 = 63 - lead;
    if (total <= want) return 0;
    return @intCast(@min(total - want, @as(u32, std.math.maxInt(u16))));
}

//! Model counting and uniform sampling.
//!
//! This is where a BDD stops being a decision procedure and starts being a
//! *generator*. Counting the satisfying assignments below every node — one
//! bottom-up pass — turns the diagram into a sampler: walk from the root and at
//! each node take the high branch with probability proportional to how many
//! solutions lie under it. The result is **exactly uniform over the solution
//! set**, in one pass, with no search and no rejection. It cannot fail, and it
//! cannot be slow, which is precisely what rejection sampling cannot promise.
//!
//! **Freezing.** The build-time diagram carries machinery the sampler has no
//! use for: a unique table, a memo cache, complement edges. So a finished root
//! is copied out into a compact read-only graph, in reverse-topological order,
//! with complement edges resolved away. That costs at most twice the reachable
//! node count — on the structure that is already the small one — and buys a
//! sampler that is a tight loop over a flat array with no indirection.
//!
//! **Counting in log space.** A variable may be 65535 bits wide, so counts
//! reach `2^65535`: far outside `f64`'s range, and exact big-integer counting
//! would allocate in a hot loop. Counts are therefore kept as `log2`, combined
//! with a max-subtracted `logAdd2` that stays stable across the whole range.
//! `f64` carries 53 mantissa bits, so the resulting bias against true uniform
//! is around `2^-53` — orders of magnitude below the PRNG's own artifacts.
//!
//! **Branch thresholds.** Each node's branch probability is converted once, at
//! freeze time, into an integer threshold, so the sampling loop touches no
//! floating point at all. The threshold is compared against *63* bits of
//! randomness rather than 64, which is not an accident: with a full 64-bit
//! compare, a probability of exactly 1 would need a threshold of `2^64`, and
//! clamping it to `maxInt` would leave a `2^-64` chance of stepping into a
//! zero-count branch and emitting an assignment that does not satisfy the
//! constraints. Against a threshold in `[0, 2^63]`, both endpoints are exact.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Manager = @import("Manager.zig");

const Ref = Manager.Ref;

/// A frozen, read-only decision graph: everything sampling needs, nothing else.
///
/// Node 0 is the false terminal and node 1 the true terminal; both sit at
/// `levels` so that the gap arithmetic below works without special cases.
/// Internal nodes appear after their children.
pub const Graph = struct {
    /// Number of variable levels this graph decides over.
    levels: u32,
    /// Level each node tests; `levels` for the two terminals.
    level: []u32,
    low: []u32,
    high: []u32,
    /// Probability of taking `high`, scaled to `[0, 2^63]`.
    threshold: []u64,
    /// `log2` of the number of satisfying assignments below each node,
    /// `-inf` for the false terminal.
    log2_count: []f64,
    root: u32,

    pub fn deinit(g: *Graph, gpa: Allocator) void {
        gpa.free(g.level);
        gpa.free(g.low);
        gpa.free(g.high);
        gpa.free(g.threshold);
        gpa.free(g.log2_count);
        g.* = undefined;
    }

    /// True when no assignment satisfies this graph.
    pub fn isEmpty(g: *const Graph) bool {
        return g.root == 0;
    }

    /// `log2` of the total number of satisfying assignments over all `levels`
    /// variables, `-inf` if there are none. The root may sit below level 0, so
    /// the levels it skips are folded back in here.
    pub fn log2Count(g: *const Graph) f64 {
        if (g.isEmpty()) return -std.math.inf(f64);
        return g.log2_count[g.root] + @as(f64, @floatFromInt(g.level[g.root]));
    }
};

/// Half of `2^64` — the scale for `threshold`, and the largest value a 63-bit
/// draw can never reach, which is what makes probability 1 exact.
const scale: f64 = 9223372036854775808.0;

/// Copy the graph reachable from `root` out of `m` into a standalone `Graph`,
/// annotated with the counts and thresholds sampling needs.
pub fn freeze(gpa: Allocator, m: *const Manager, root: Ref, levels: u32) Allocator.Error!Graph {
    var b: Builder = .{
        .gpa = gpa,
        .m = m,
        .levels = levels,
    };
    defer b.seen.deinit(gpa);
    errdefer b.deinitLists();

    // Terminals first, so they occupy the well-known indices 0 and 1.
    try b.emit(.zero, levels, 0, 0);
    try b.emit(.one, levels, 0, 0);
    try b.seen.put(gpa, .zero, 0);
    try b.seen.put(gpa, .one, 1);

    const root_index = try b.visit(root);

    const level = try b.level.toOwnedSlice(gpa);
    errdefer gpa.free(level);
    const low = try b.low.toOwnedSlice(gpa);
    errdefer gpa.free(low);
    const high = try b.high.toOwnedSlice(gpa);
    errdefer gpa.free(high);
    const threshold = try b.threshold.toOwnedSlice(gpa);
    errdefer gpa.free(threshold);
    const log2_count = try b.log2_count.toOwnedSlice(gpa);

    var g: Graph = .{
        .levels = levels,
        .level = level,
        .low = low,
        .high = high,
        .threshold = threshold,
        .log2_count = log2_count,
        .root = root_index,
    };

    count(&g);
    return g;
}

const Builder = struct {
    gpa: Allocator,
    m: *const Manager,
    levels: u32,
    seen: std.AutoHashMapUnmanaged(Ref, u32) = .empty,

    level: std.ArrayListUnmanaged(u32) = .empty,
    low: std.ArrayListUnmanaged(u32) = .empty,
    high: std.ArrayListUnmanaged(u32) = .empty,
    threshold: std.ArrayListUnmanaged(u64) = .empty,
    log2_count: std.ArrayListUnmanaged(f64) = .empty,

    fn deinitLists(b: *Builder) void {
        b.level.deinit(b.gpa);
        b.low.deinit(b.gpa);
        b.high.deinit(b.gpa);
        b.threshold.deinit(b.gpa);
        b.log2_count.deinit(b.gpa);
    }

    fn emit(b: *Builder, r: Ref, lvl: u32, lo: u32, hi: u32) Allocator.Error!void {
        _ = r;
        try b.level.append(b.gpa, lvl);
        try b.low.append(b.gpa, lo);
        try b.high.append(b.gpa, hi);
        try b.threshold.append(b.gpa, 0);
        try b.log2_count.append(b.gpa, 0);
    }

    /// Emit `r` and everything below it, children first. `Ref` already encodes
    /// the node *and* its complement bit, so it is the natural memo key: the
    /// two polarities of one build node become two frozen nodes, which is what
    /// removes complement edges from the result.
    fn visit(b: *Builder, r: Ref) Allocator.Error!u32 {
        if (b.seen.get(r)) |index| return index;

        const node = r.node();
        const level = b.m.nodes.items(.level)[node];
        var lo = b.m.nodes.items(.low)[node];
        var hi = b.m.nodes.items(.high)[node];
        if (r.isComplemented()) {
            lo = lo.not();
            hi = hi.not();
        }

        const lo_index = try b.visit(lo);
        const hi_index = try b.visit(hi);

        const index: u32 = @intCast(b.level.items.len);
        try b.emit(r, level, lo_index, hi_index);
        try b.seen.put(b.gpa, r, index);
        return index;
    }
};

/// Fill in `log2_count` and `threshold`. Children precede parents, so one
/// forward pass suffices.
fn count(g: *Graph) void {
    g.log2_count[0] = -std.math.inf(f64); // false: no assignments
    g.log2_count[1] = 0; // true: exactly one, over zero remaining levels

    for (2..g.log2_count.len) |i| {
        const lo = branchLog2(g, @intCast(i), g.low[i]);
        const hi = branchLog2(g, @intCast(i), g.high[i]);
        const total = logAdd2(lo, hi);
        g.log2_count[i] = total;

        // Probability of the high branch. Both endpoints land exactly: an
        // impossible high branch gives 0, an impossible low branch gives 1.
        const p: f64 = if (hi == -std.math.inf(f64)) 0.0 else @exp2(hi - total);
        g.threshold[i] = @intFromFloat(@min(scale, @max(0.0, @round(p * scale))));
    }
}

/// `log2` of the assignments reachable through one branch of `node`, including
/// the `2^gap` free assignments for the levels the branch skips.
fn branchLog2(g: *const Graph, node: u32, child: u32) f64 {
    const gap: f64 = @floatFromInt(g.level[child] - g.level[node] - 1);
    return g.log2_count[child] + gap;
}

fn logAdd2(a: f64, b: f64) f64 {
    const neg_inf = -std.math.inf(f64);
    if (a == neg_inf) return b;
    if (b == neg_inf) return a;
    const hi = @max(a, b);
    return hi + @log2(1.0 + @exp2(@min(a, b) - hi));
}

/// Draw one assignment, uniformly over the solution set.
///
/// `setBit(ctx, level, value)` receives only the levels the graph actually
/// decides. Every other level is unconstrained along this path, so its value
/// must already be uniformly random — callers pre-fill their output buffer,
/// which makes the walk cost proportional to the path length rather than to the
/// total number of bits.
///
/// Must not be called on an empty graph.
pub fn walk(
    g: *const Graph,
    prng: *std.Random.DefaultPrng,
    ctx: anytype,
    comptime setBit: fn (@TypeOf(ctx), level: u32, value: bool) void,
) void {
    std.debug.assert(!g.isEmpty());
    var node = g.root;
    while (node > 1) {
        const take_high = (prng.next() >> 1) < g.threshold[node];
        setBit(ctx, g.level[node], take_high);
        node = if (take_high) g.high[node] else g.low[node];
    }
    // Thresholds are exact at both endpoints, so a zero-count branch is
    // unreachable and the walk always lands on the true terminal.
    std.debug.assert(node == 1);
}

// -- Tests -------------------------------------------------------------------

const testing = std.testing;

/// Collect a sample into a bitmask, for the tests below.
const Collector = struct {
    bits: u64 = 0,

    fn set(c: *Collector, level: u32, value: bool) void {
        const mask = @as(u64, 1) << @intCast(level);
        // Levels are pre-filled with random bits, so a decided level must be
        // able to clear as well as set.
        if (value) c.bits |= mask else c.bits &= ~mask;
    }
};

test "counting matches brute force" {
    const gpa = testing.allocator;
    var m = try Manager.init(gpa, .{});
    defer m.deinit();

    const levels = 4;
    var vars: [levels]Ref = undefined;
    for (&vars, 0..) |*v, i| v.* = try m.variable(@intCast(i));

    // A few functions with hand-checkable counts over 4 variables.
    const cases = [_]struct { f: Ref, expect: f64 }{
        .{ .f = .one, .expect = 16 },
        .{ .f = .zero, .expect = 0 },
        .{ .f = vars[0], .expect = 8 },
        .{ .f = try m.conj(vars[0], vars[1]), .expect = 4 },
        .{ .f = try m.disj(vars[0], vars[1]), .expect = 12 },
        .{ .f = try m.xor(vars[0], try m.xor(vars[1], vars[2])), .expect = 8 },
    };

    for (cases) |c| {
        var g = try freeze(gpa, &m, c.f, levels);
        defer g.deinit(gpa);
        if (c.expect == 0) {
            try testing.expect(g.isEmpty());
            try testing.expect(g.log2Count() == -std.math.inf(f64));
        } else {
            try testing.expectApproxEqAbs(@log2(c.expect), g.log2Count(), 1e-9);
        }
    }
}

test "every drawn assignment satisfies the function" {
    const gpa = testing.allocator;
    var m = try Manager.init(gpa, .{});
    defer m.deinit();

    const levels = 6;
    var vars: [levels]Ref = undefined;
    for (&vars, 0..) |*v, i| v.* = try m.variable(@intCast(i));

    // (x0 & !x1) | (x2 & x3 & !x4) — an awkward shape with unequal branches,
    // and x5 left entirely free.
    const f = try m.disj(
        try m.conj(vars[0], vars[1].not()),
        try m.conj(vars[2], try m.conj(vars[3], vars[4].not())),
    );

    var g = try freeze(gpa, &m, f, levels);
    defer g.deinit(gpa);

    var prng: std.Random.DefaultPrng = .init(7);
    for (0..2000) |_| {
        // Unconstrained levels come pre-filled; the walk overwrites the rest.
        var c: Collector = .{ .bits = prng.next() & 0x3f };
        walk(&g, &prng, &c, Collector.set);
        try testing.expect(m.eval(f, c.bits));
    }
}

test "sampling is uniform over the solution set" {
    const gpa = testing.allocator;
    var m = try Manager.init(gpa, .{});
    defer m.deinit();

    const levels = 5;
    var vars: [levels]Ref = undefined;
    for (&vars, 0..) |*v, i| v.* = try m.variable(@intCast(i));

    // An asymmetric function, so a naive "coin flip per level" sampler would
    // visibly skew: x0 -> the rest are free; !x0 -> only one assignment.
    const rest_free = vars[0];
    const pinned = try m.conj(
        vars[0].not(),
        try m.conj(vars[1], try m.conj(vars[2], try m.conj(vars[3], vars[4]))),
    );
    const f = try m.disj(rest_free, pinned);

    var g = try freeze(gpa, &m, f, levels);
    defer g.deinit(gpa);

    // 16 assignments with x0 set, plus exactly one without: 17 solutions.
    try testing.expectApproxEqAbs(@log2(17.0), g.log2Count(), 1e-9);

    var histogram: [32]u32 = .{0} ** 32;
    var prng: std.Random.DefaultPrng = .init(99);
    const draws = 170_000;
    for (0..draws) |_| {
        var c: Collector = .{ .bits = prng.next() & 0x1f };
        walk(&g, &prng, &c, Collector.set);
        try testing.expect(m.eval(f, c.bits));
        histogram[@intCast(c.bits)] += 1;
    }

    // Each of the 17 solutions should get ~1/17 of the draws; nothing else may
    // appear at all.
    const expected: f64 = @as(f64, draws) / 17.0;
    var solutions: u32 = 0;
    for (histogram, 0..) |hits, bits| {
        if (!m.eval(f, @intCast(bits))) {
            try testing.expectEqual(@as(u32, 0), hits);
            continue;
        }
        solutions += 1;
        const ratio = @as(f64, @floatFromInt(hits)) / expected;
        try testing.expect(ratio > 0.9 and ratio < 1.1);
    }
    try testing.expectEqual(@as(u32, 17), solutions);
}

test "a probability-one branch is never mis-taken" {
    const gpa = testing.allocator;
    var m = try Manager.init(gpa, .{});
    defer m.deinit();

    // x0 & x1 & x2: at every node one branch has zero solutions, so every
    // threshold sits at an endpoint. A sloppy scaling would leak here.
    const f = try m.conj(try m.variable(0), try m.conj(try m.variable(1), try m.variable(2)));
    var g = try freeze(gpa, &m, f, 3);
    defer g.deinit(gpa);

    try testing.expectApproxEqAbs(@as(f64, 0), g.log2Count(), 1e-9);

    var prng: std.Random.DefaultPrng = .init(1234);
    for (0..10_000) |_| {
        var c: Collector = .{};
        walk(&g, &prng, &c, Collector.set);
        try testing.expectEqual(@as(u64, 0b111), c.bits);
    }
}

test {
    testing.refAllDecls(@This());
}

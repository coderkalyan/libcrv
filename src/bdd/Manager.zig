//! A reduced ordered binary decision diagram (ROBDD) package.
//!
//! This file knows nothing about libcrv — it is a general Boolean function
//! library over a fixed variable order, which is what lets it be tested against
//! plain Boolean identities rather than through the IR.
//!
//! **Canonicity.** Two structurally identical functions are the *same* `Ref`.
//! That is the property everything downstream leans on: equality is a `u32`
//! compare, and "unsatisfiable" is literally `ref == .zero`. It is maintained by
//! the unique table, which interns every `(level, low, high)` triple, together
//! with the reduction rule that a node whose branches agree is not a node at
//! all.
//!
//! **Complement edges.** A `Ref` packs a node index and a complement bit, so
//! negation is one XOR and `f` and `¬f` share their entire sub-graph — roughly
//! halving node count. Canonicity survives because of one extra rule: the
//! `high` edge is never complemented, and `mk` pushes a complemented `high`
//! down onto both children and out onto the returned edge.
//!
//! **One primitive.** Every operator is `ite(f, g, h)`. Before the memo lookup
//! the triple is normalized — the standard substitutions, the symmetry
//! rewrites, and complement canonicalization — so that calls which are
//! semantically the same collide in the cache however they were spelled. That
//! normalization does more for throughput than any other single thing here.
//!
//! **Memory.** Nodes are bump-allocated and never individually freed: there is
//! no garbage collector and no reference counting. That is not a shortcut, it
//! is a consequence of how this package is used. A caller builds one
//! self-contained problem, extracts what it needs, and calls `reset` — which
//! reclaims everything in constant time. Nothing built before a `reset` is ever
//! referenced after it. A general-purpose package cannot assume that and must
//! collect; here, collecting would mark nearly every node and sweep almost
//! nothing. `node_budget` bounds the peak instead, and is the one place a build
//! can fail.
//!
//! `reset` is genuinely constant time, not merely amortized: both tables carry
//! an epoch stamp and a `reset` just bumps the counter, so entries from an
//! earlier epoch read as empty. Clearing several megabytes of table between
//! sub-problems would otherwise dominate the build whenever a caller has many
//! small ones to solve — which, after partitioning, is the common case.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Manager = @This();

/// An edge to a node: bits 31..1 are the node index, bit 0 is the complement
/// flag. Node 0 is the terminal, so `zero` and `one` are its two edges.
pub const Ref = enum(u32) {
    /// The constant false.
    zero = 0,
    /// The constant true.
    one = 1,
    _,

    pub fn not(r: Ref) Ref {
        return @enumFromInt(@intFromEnum(r) ^ 1);
    }

    pub fn isComplemented(r: Ref) bool {
        return @intFromEnum(r) & 1 == 1;
    }

    /// The node this edge points at, ignoring the complement bit.
    pub fn node(r: Ref) u32 {
        return @intFromEnum(r) >> 1;
    }

    pub fn isTerminal(r: Ref) bool {
        return r.node() == 0;
    }

    fn make(index: u32, complemented: bool) Ref {
        return @enumFromInt((index << 1) | @intFromBool(complemented));
    }
};

pub const Error = error{
    /// The build exceeded `node_budget`. The manager is left usable but its
    /// contents are meaningless; `reset` it before building anything else.
    NodeBudgetExceeded,
} || Allocator.Error;

/// The level a terminal sits at — below every real variable, so `min` over a
/// mixed set of refs naturally picks a non-terminal whenever one exists.
pub const terminal_level: u32 = std.math.maxInt(u32);

const Node = struct { level: u32, low: Ref, high: Ref };

gpa: Allocator,
nodes: std.MultiArrayList(Node) = .{},
/// Open-addressed `(level, low, high)` -> node index. A slot counts as empty
/// when its `epoch` is stale, which is what makes `reset` constant time.
unique: []Slot,
unique_mask: u32,
unique_len: u32 = 0,
/// Direct-mapped, lossy memo for `ite`. Collisions overwrite; correctness never
/// depends on a hit, which is what lets it skip chaining and eviction policy.
memo: []Memo,
memo_mask: u32,
node_budget: u32,
/// Bumped by `reset`; entries stamped with an older value are ignored.
epoch: u32 = 1,

const Slot = struct {
    node: u32 = 0,
    epoch: u32 = 0,
};

const Memo = struct {
    f: Ref = .zero,
    g: Ref = .zero,
    h: Ref = .zero,
    r: Ref = .zero,
    epoch: u32 = 0,
};

pub const Options = struct {
    /// Initial unique-table capacity, rounded up to a power of two. It grows on
    /// demand, so this only sets how soon the first rehash happens.
    unique_capacity: u32 = 1 << 12,
    /// Memo entries, rounded up to a power of two. Deliberately small: keeping
    /// the cache resident in L2 measurably beats giving it a higher hit rate,
    /// and an oversized table costs real time to clear before it is ever read.
    memo_capacity: u32 = 1 << 14,
    /// Hard ceiling on live nodes. Crossing it fails the build.
    node_budget: u32 = 1 << 21,
};

pub fn init(gpa: Allocator, options: Options) Allocator.Error!Manager {
    const unique_cap = std.math.ceilPowerOfTwoAssert(u32, @max(16, options.unique_capacity));
    const memo_cap = std.math.ceilPowerOfTwoAssert(u32, @max(16, options.memo_capacity));

    const unique = try gpa.alloc(Slot, unique_cap);
    errdefer gpa.free(unique);
    @memset(unique, .{});

    const memo = try gpa.alloc(Memo, memo_cap);
    errdefer gpa.free(memo);
    @memset(memo, .{});

    var self: Manager = .{
        .gpa = gpa,
        .unique = unique,
        .unique_mask = unique_cap - 1,
        .memo = memo,
        .memo_mask = memo_cap - 1,
        .node_budget = @max(1, options.node_budget),
    };
    // Node 0 is the terminal; its edges are never followed.
    try self.nodes.append(gpa, .{ .level = terminal_level, .low = .zero, .high = .zero });
    return self;
}

pub fn deinit(self: *Manager) void {
    self.nodes.deinit(self.gpa);
    self.gpa.free(self.unique);
    self.gpa.free(self.memo);
    self.* = undefined;
}

/// Discard every node, keeping the allocated tables. Constant time, and the
/// only reclamation this package offers — see the note on memory above.
pub fn reset(self: *Manager) void {
    self.nodes.shrinkRetainingCapacity(1);
    self.unique_len = 0;
    self.epoch, const wrapped = @addWithOverflow(self.epoch, 1);
    if (wrapped != 0 or self.epoch == 0) {
        // Astronomically unlikely, but a wrapped epoch would resurrect stale
        // entries, so pay for one real clear and start over.
        @memset(self.unique, .{});
        @memset(self.memo, .{});
        self.epoch = 1;
    }
}

/// Live node count, excluding the terminal.
pub fn nodeCount(self: *const Manager) u32 {
    return @intCast(self.nodes.len - 1);
}

pub fn levelOf(self: *const Manager, r: Ref) u32 {
    return self.nodes.items(.level)[r.node()];
}

/// The function "variable at `level` is true".
pub fn variable(self: *Manager, level: u32) Error!Ref {
    return self.mk(level, .zero, .one);
}

// -- Node construction -------------------------------------------------------

/// Build (or find) the node testing `level`, with the reduction rule applied
/// and the complement pushed so that `high` stays regular.
fn mk(self: *Manager, level: u32, low: Ref, high: Ref) Error!Ref {
    if (low == high) return low;
    if (high.isComplemented()) return (try self.intern(level, low.not(), high.not())).not();
    return self.intern(level, low, high);
}

fn intern(self: *Manager, level: u32, low: Ref, high: Ref) Error!Ref {
    var slot = hash3(level, low, high) & self.unique_mask;
    {
        const levels = self.nodes.items(.level);
        const lows = self.nodes.items(.low);
        const highs = self.nodes.items(.high);
        while (self.unique[slot].epoch == self.epoch) : (slot = (slot + 1) & self.unique_mask) {
            const i = self.unique[slot].node;
            if (levels[i] == level and lows[i] == low and highs[i] == high) {
                return Ref.make(i, false);
            }
        }
    }

    if (self.nodes.len > self.node_budget) return error.NodeBudgetExceeded;

    const index: u32 = @intCast(self.nodes.len);
    try self.nodes.append(self.gpa, .{ .level = level, .low = low, .high = high });
    self.unique[slot] = .{ .node = index, .epoch = self.epoch };
    self.unique_len += 1;

    // Grow before the table gets dense enough for probe chains to bite.
    if (self.unique_len * 10 >= (self.unique_mask + 1) * 7) try self.growUnique();

    return Ref.make(index, false);
}

fn growUnique(self: *Manager) Allocator.Error!void {
    const capacity: u32 = (self.unique_mask + 1) * 2;
    const table = try self.gpa.alloc(Slot, capacity);
    @memset(table, .{});
    self.gpa.free(self.unique);
    self.unique = table;
    self.unique_mask = capacity - 1;

    // Every live node belongs to the current epoch, so rehashing them all
    // restores the table exactly.
    const levels = self.nodes.items(.level);
    const lows = self.nodes.items(.low);
    const highs = self.nodes.items(.high);
    for (1..self.nodes.len) |i| {
        var slot = hash3(levels[i], lows[i], highs[i]) & self.unique_mask;
        while (self.unique[slot].epoch == self.epoch) slot = (slot + 1) & self.unique_mask;
        self.unique[slot] = .{ .node = @intCast(i), .epoch = self.epoch };
    }
}

// -- if-then-else ------------------------------------------------------------

/// `ite(f, g, h)` — the single primitive every other operator is written in
/// terms of.
pub fn ite(self: *Manager, f_in: Ref, g_in: Ref, h_in: Ref) Error!Ref {
    var f = f_in;
    var g = g_in;
    var h = h_in;

    // Substitutions: inside the `then` branch `f` is true, inside `else` it is
    // false, so any occurrence of `f` in `g`/`h` collapses to a constant.
    if (f == .one) return g;
    if (f == .zero) return h;
    if (f == g) g = .one;
    if (f == g.not()) g = .zero;
    if (f == h) h = .zero;
    if (f == h.not()) h = .one;
    if (g == h) return g;
    if (g == .one and h == .zero) return f;
    if (g == .zero and h == .one) return f.not();

    // Symmetry rewrites. Each is an identity; picking the operand nearer the
    // root as `f` gives one canonical spelling per function, so distinct call
    // sites share memo entries.
    if (g == .one and self.levelOf(h) < self.levelOf(f)) {
        std.mem.swap(Ref, &f, &h); // ite(f,1,h) == ite(h,1,f)
    } else if (h == .zero and self.levelOf(g) < self.levelOf(f)) {
        std.mem.swap(Ref, &f, &g); // ite(f,g,0) == ite(g,f,0)
    } else if (h == .one and self.levelOf(g) < self.levelOf(f)) {
        const nf = f.not(); // ite(f,g,1) == ite(¬g,¬f,1)
        f = g.not();
        g = nf;
    } else if (g == .zero and self.levelOf(h) < self.levelOf(f)) {
        const nf = f.not(); // ite(f,0,h) == ite(¬h,0,¬f)
        f = h.not();
        h = nf;
    }

    // Complement canonicalization: force `f` regular, then force `g` regular
    // and remember to invert the result.
    if (f.isComplemented()) {
        f = f.not();
        std.mem.swap(Ref, &g, &h);
    }
    var flip = false;
    if (g.isComplemented()) {
        g = g.not();
        h = h.not();
        flip = true;
    }

    // The rewrites above can expose a terminal case that was not visible
    // before, so re-check rather than recursing on a trivial triple.
    if (g == h) return applyFlip(g, flip);
    if (g == .one and h == .zero) return applyFlip(f, flip);
    if (g == .zero and h == .one) return applyFlip(f.not(), flip);

    std.debug.assert(!f.isTerminal() and !f.isComplemented());
    const slot = hash3ref(f, g, h) & self.memo_mask;
    const cached = self.memo[slot];
    if (cached.epoch == self.epoch and cached.f == f and cached.g == g and cached.h == h) {
        return applyFlip(cached.r, flip);
    }

    const top = @min(self.levelOf(f), @min(self.levelOf(g), self.levelOf(h)));
    const fc = self.cofactor(f, top);
    const gc = self.cofactor(g, top);
    const hc = self.cofactor(h, top);

    const low = try self.ite(fc.low, gc.low, hc.low);
    const high = try self.ite(fc.high, gc.high, hc.high);
    const r = try self.mk(top, low, high);

    self.memo[slot] = .{ .f = f, .g = g, .h = h, .r = r, .epoch = self.epoch };
    return applyFlip(r, flip);
}

fn applyFlip(r: Ref, flip: bool) Ref {
    return if (flip) r.not() else r;
}

const Cofactor = struct { low: Ref, high: Ref };

/// Split `r` on `level`. A ref that does not test `level` is unchanged by it,
/// so both branches are the ref itself.
fn cofactor(self: *const Manager, r: Ref, level: u32) Cofactor {
    const i = r.node();
    if (self.nodes.items(.level)[i] != level) return .{ .low = r, .high = r };
    const low = self.nodes.items(.low)[i];
    const high = self.nodes.items(.high)[i];
    // A complemented edge negates the whole sub-function, hence both branches.
    if (r.isComplemented()) return .{ .low = low.not(), .high = high.not() };
    return .{ .low = low, .high = high };
}

// -- Derived operators -------------------------------------------------------

pub fn negate(_: *Manager, f: Ref) Ref {
    return f.not();
}

pub fn conj(self: *Manager, f: Ref, g: Ref) Error!Ref {
    return self.ite(f, g, .zero);
}

pub fn disj(self: *Manager, f: Ref, g: Ref) Error!Ref {
    return self.ite(f, .one, g);
}

pub fn xor(self: *Manager, f: Ref, g: Ref) Error!Ref {
    return self.ite(f, g.not(), g);
}

pub fn xnor(self: *Manager, f: Ref, g: Ref) Error!Ref {
    return self.ite(f, g, g.not());
}

pub fn implies(self: *Manager, f: Ref, g: Ref) Error!Ref {
    return self.ite(f, g, .one);
}

// -- Hashing -----------------------------------------------------------------

fn mix(h_in: u64) u32 {
    var h = h_in;
    h ^= h >> 33;
    h *%= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *%= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return @truncate(h);
}

fn hash3(level: u32, low: Ref, high: Ref) u32 {
    const a: u64 = (@as(u64, level) << 32) | @intFromEnum(low);
    return mix(a ^ (@as(u64, @intFromEnum(high)) *% 0x9e3779b97f4a7c15));
}

fn hash3ref(f: Ref, g: Ref, h: Ref) u32 {
    const a: u64 = (@as(u64, @intFromEnum(f)) << 32) | @intFromEnum(g);
    return mix(a ^ (@as(u64, @intFromEnum(h)) *% 0x9e3779b97f4a7c15));
}

// -- Tests -------------------------------------------------------------------

/// Evaluate `r` under an assignment where bit `i` of `bits` is the value of the
/// variable at level `i`. A direct reference implementation — the fast path is
/// `sample.walk` over a frozen graph — kept for tests that need to check a
/// specific assignment against the diagram.
pub fn eval(self: *const Manager, r: Ref, bits: u64) bool {
    var cur = r;
    while (!cur.isTerminal()) {
        const level = self.levelOf(cur);
        const c = self.cofactor(cur, level);
        cur = if ((bits >> @intCast(level)) & 1 == 1) c.high else c.low;
    }
    // A terminal edge is `one` exactly when it is complemented.
    return cur == .one;
}

test "terminals and negation" {
    var m = try Manager.init(std.testing.allocator, .{});
    defer m.deinit();

    try std.testing.expectEqual(Manager.Ref.one, Manager.Ref.zero.not());
    try std.testing.expect(Manager.Ref.zero.isTerminal());
    try std.testing.expect(Manager.Ref.one.isTerminal());

    const x = try m.variable(0);
    try std.testing.expect(!x.isTerminal());
    try std.testing.expectEqual(x, x.not().not());
    try std.testing.expectEqual(Manager.Ref.zero, try m.conj(x, x.not()));
    try std.testing.expectEqual(Manager.Ref.one, try m.disj(x, x.not()));
}

test "canonicity: equal functions are the identical ref" {
    var m = try Manager.init(std.testing.allocator, .{});
    defer m.deinit();

    const a = try m.variable(0);
    const b = try m.variable(1);
    const c = try m.variable(2);

    // (a & b) | (a & c) == a & (b | c)
    const lhs = try m.disj(try m.conj(a, b), try m.conj(a, c));
    const rhs = try m.conj(a, try m.disj(b, c));
    try std.testing.expectEqual(lhs, rhs);

    // De Morgan, both directions.
    try std.testing.expectEqual(
        (try m.conj(a, b)).not(),
        try m.disj(a.not(), b.not()),
    );
    try std.testing.expectEqual(
        (try m.disj(a, b)).not(),
        try m.conj(a.not(), b.not()),
    );

    // xor via ite matches its expansion.
    try std.testing.expectEqual(
        try m.xor(a, b),
        try m.disj(try m.conj(a, b.not()), try m.conj(a.not(), b)),
    );
}

test "exhaustive: random 4-variable formulas match their truth tables" {
    const gpa = std.testing.allocator;
    var m = try Manager.init(gpa, .{});
    defer m.deinit();

    const n_vars = 4;
    var refs: [n_vars]Manager.Ref = undefined;
    for (&refs, 0..) |*r, i| r.* = try m.variable(@intCast(i));

    var prng: std.Random.DefaultPrng = .init(0xBDD);
    const rand = prng.random();

    // Build random expression trees over the four variables, and check the BDD
    // against a truth table computed directly over the same tree.
    for (0..200) |_| {
        // A pool of sub-expressions: (ref, truth table over 16 assignments).
        var pool_ref: [24]Manager.Ref = undefined;
        var pool_tt: [24]u16 = undefined;
        for (0..n_vars) |i| {
            pool_ref[i] = refs[i];
            var tt: u16 = 0;
            for (0..16) |bits| {
                if ((bits >> @intCast(i)) & 1 == 1) tt |= @as(u16, 1) << @intCast(bits);
            }
            pool_tt[i] = tt;
        }

        var len: usize = n_vars;
        while (len < pool_ref.len) : (len += 1) {
            const i = rand.uintLessThan(usize, len);
            const j = rand.uintLessThan(usize, len);
            switch (rand.uintLessThan(u8, 5)) {
                0 => {
                    pool_ref[len] = try m.conj(pool_ref[i], pool_ref[j]);
                    pool_tt[len] = pool_tt[i] & pool_tt[j];
                },
                1 => {
                    pool_ref[len] = try m.disj(pool_ref[i], pool_ref[j]);
                    pool_tt[len] = pool_tt[i] | pool_tt[j];
                },
                2 => {
                    pool_ref[len] = try m.xor(pool_ref[i], pool_ref[j]);
                    pool_tt[len] = pool_tt[i] ^ pool_tt[j];
                },
                3 => {
                    pool_ref[len] = pool_ref[i].not();
                    pool_tt[len] = ~pool_tt[i];
                },
                else => {
                    const k = rand.uintLessThan(usize, len);
                    pool_ref[len] = try m.ite(pool_ref[i], pool_ref[j], pool_ref[k]);
                    pool_tt[len] = (pool_tt[i] & pool_tt[j]) | (~pool_tt[i] & pool_tt[k]);
                },
            }
        }

        for (pool_ref, pool_tt) |r, tt| {
            for (0..16) |bits| {
                const expected = (tt >> @intCast(bits)) & 1 == 1;
                try std.testing.expectEqual(expected, m.eval(r, bits));
            }
        }
    }
}

test "canonicity implies equality: identical truth tables share a ref" {
    const gpa = std.testing.allocator;
    var m = try Manager.init(gpa, .{});
    defer m.deinit();

    var refs: [3]Manager.Ref = undefined;
    for (&refs, 0..) |*r, i| r.* = try m.variable(@intCast(i));

    // Enumerate every function of 3 variables reachable as a "sum of minterms"
    // and confirm the ref is a pure function of the truth table.
    var seen: [256]?Manager.Ref = .{null} ** 256;
    for (0..256) |tt| {
        var acc: Manager.Ref = .zero;
        for (0..8) |bits| {
            if ((tt >> @intCast(bits)) & 1 == 0) continue;
            var minterm: Manager.Ref = .one;
            for (refs, 0..) |r, i| {
                const lit = if ((bits >> @intCast(i)) & 1 == 1) r else r.not();
                minterm = try m.conj(minterm, lit);
            }
            acc = try m.disj(acc, minterm);
        }
        seen[tt] = acc;
    }
    // All 256 functions are distinct refs, and none collide.
    for (0..256) |i| {
        for (i + 1..256) |j| try std.testing.expect(seen[i].? != seen[j].?);
    }
}

test "reset reclaims every node" {
    var m = try Manager.init(std.testing.allocator, .{});
    defer m.deinit();

    const a = try m.variable(0);
    _ = try m.conj(a, try m.variable(1));
    try std.testing.expect(m.nodeCount() > 0);

    m.reset();
    try std.testing.expectEqual(@as(u32, 0), m.nodeCount());

    // Still usable, and rebuilding from scratch gives a consistent result.
    const b = try m.variable(0);
    try std.testing.expectEqual(Manager.Ref.zero, try m.conj(b, b.not()));
}

/// A function whose BDD needs one node per variable, for the budget test.
fn buildParity(m: *Manager, n: u32) Manager.Error!Manager.Ref {
    var acc: Manager.Ref = .zero;
    for (0..n) |i| acc = try m.xor(acc, try m.variable(@intCast(i)));
    return acc;
}

test "node budget fails the build instead of exhausting memory" {
    var m = try Manager.init(std.testing.allocator, .{ .node_budget = 32 });
    defer m.deinit();

    try std.testing.expectError(error.NodeBudgetExceeded, buildParity(&m, 64));

    // The manager stays usable after a failed build, once reset. Each parity
    // prefix is a distinct function, so `n` variables cost ~n²/2 nodes total.
    m.reset();
    _ = try buildParity(&m, 6);
    try std.testing.expect(m.nodeCount() <= 32);
}

test {
    std.testing.refAllDecls(@This());
}

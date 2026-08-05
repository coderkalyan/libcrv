//! A BDD-based constraint solver.
//!
//! Where `RejectionSampler` guesses until it gets lucky, this engine builds a
//! canonical representation of the *entire solution set* once, and then draws
//! from it directly. The cost model inverts:
//!
//!   * Setup is expensive — bit-blasting the constraints into a reduced ordered
//!     binary decision diagram.
//!   * Every draw afterwards is a single root-to-leaf walk. It is proportional
//!     to the path length, allocates nothing, and **cannot fail**.
//!
//! That trade is the right one for constrained random verification, where the
//! same constraint set is randomized thousands of times. It also buys three
//! things rejection sampling cannot offer at any price:
//!
//!   * **Exactly uniform** sampling over the solution set, however sparse it
//!     is. A rejection sampler that accepts one draw in `2^32` is not slow, it
//!     is broken; this engine does not notice the difference.
//!   * **Unsatisfiability is decided.** `next` returning `false` means "no
//!     assignment exists", not "I gave up" — see `isUnsat`.
//!   * **The solution count is known exactly** (`log2Count`), which is a
//!     coverage measurement, not just a solver statistic.
//!
//! The pipeline at `init`:
//!
//!   1. `Partition` splits the constraints into independent components.
//!      Sampling them separately is exact, and it keeps the expensive step
//!      scoped to the largest island rather than the whole set.
//!   2. Per component: `order` assigns BDD levels to variable bits, `blast`
//!      compiles the statements, and `sample.freeze` copies the finished
//!      diagram out with its model counts. The shared `Manager` is then reset
//!      in constant time — which is why this engine has no garbage collector.
//!
//! **Failure is a hard error, by design.** A BDD can blow up: variable-by-
//! variable multiplication is provably exponential for every variable order, so
//! no ordering heuristic can save it. When a build exceeds its budget, `init`
//! fails and says why. It does not silently fall back to another engine —
//! choosing an engine is the caller's decision, and a hybrid that degrades
//! gracefully belongs behind its own `Solver` implementation, not hidden
//! inside this one.
//!
//! Scope: the scalar expression subset, plus `if_else` and `unique`. `dist`,
//! `solve_before`, and `foreach` are rejected with `error.UnsupportedNode`
//! rather than silently ignored. Matching `RejectionSampler`, `soft` constraint
//! flags are treated as hard and `randc` variables are treated as `rand`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");
const Partition = @import("Partition.zig");
const Solver = @import("Solver.zig");
const Manager = @import("bdd/Manager.zig");
const Order = @import("bdd/order.zig");
const blast = @import("bdd/blast.zig");
const sample = @import("bdd/sample.zig");

const Value = Solver.Value;
const BddSolver = @This();

pub const Options = struct {
    seed: u64 = 0,
    /// Hard ceiling on live BDD nodes during one component's build. The
    /// backstop against an exponential blow-up; crossing it fails `init`.
    node_budget: u32 = 1 << 21,
    /// Most random bits one component may span. Bounds diagram depth, and with
    /// it the recursion depth inside the BDD package.
    max_levels: u32 = 4096,
    /// Widest variable-by-variable multiply or divide to attempt.
    max_mul_width: u16 = 12,
    /// Initial BDD unique-table capacity; it grows on demand.
    unique_capacity: u32 = 1 << 12,
    /// BDD memo-cache entries. Small on purpose — see `Manager.Options`.
    memo_capacity: u32 = 1 << 14,
};

pub const Error = Order.Error || blast.Error;

ir: *const Ir,
prng: std.Random.DefaultPrng,
value_limbs: usize,
/// Per-variable draw info, indexed by `Ir.Variable.Index`.
vars: []VarInfo,
components: []Component,
/// `log2` of the number of assignments satisfying every constraint.
log2_count: f64,
/// Set when some component has no solution, which makes the whole set unsat.
unsat: bool,

const VarInfo = struct {
    /// Limbs this variable occupies, and the mask for its most significant one.
    used: u32,
    top_mask: u64,
};

/// One independent sub-problem: a frozen decision graph plus the map from its
/// levels back to variable bits.
const Component = struct {
    graph: sample.Graph,
    /// Level -> the output bit it decides.
    bits: []Slot,

    const Slot = struct { v: u32, bit: u16 };
};

pub fn init(gpa: Allocator, ir: *const Ir, options: Options) Error!BddSolver {
    var self: BddSolver = .{
        .ir = ir,
        .prng = .init(options.seed),
        .value_limbs = Solver.valueLimbs(ir),
        .vars = &.{},
        .components = &.{},
        .log2_count = 0,
        .unsat = false,
    };
    // Each resource carries its own `errdefer` rather than deferring to
    // `deinit`, which would walk components that have not been built yet.
    self.vars = try gpa.alloc(VarInfo, ir.vars.len);
    errdefer gpa.free(self.vars);

    for (self.vars, 0..) |*info, v| {
        const width = Order.widthOf(ir, @enumFromInt(@as(u32, @intCast(v))));
        const used = std.math.big.int.calcTwosCompLimbCount(width);
        const top_bits = width - (used - 1) * 64;
        info.* = .{
            .used = @intCast(used),
            .top_mask = if (top_bits == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(top_bits)) - 1,
        };
        // Every bit starts out free; each component subtracts the ones it owns.
        self.log2_count += @floatFromInt(width);
    }

    // Node widths, resolved once — `ir.typeOf` walks operands recursively, and
    // the blaster needs a width for every node it touches.
    const types = try gpa.alloc(Ir.Type, ir.nodes.len);
    defer gpa.free(types);
    for (types, 0..) |*t, i| t.* = ir.typeOf(@enumFromInt(@as(u32, @intCast(i))));

    var partition = try Partition.init(gpa, ir);
    defer partition.deinit(gpa);

    // Variable index -> slot within its component, rebuilt per component.
    const slot_of_var = try gpa.alloc(u32, ir.vars.len);
    defer gpa.free(slot_of_var);
    @memset(slot_of_var, blast.no_slot);

    var manager = try Manager.init(gpa, .{
        .unique_capacity = options.unique_capacity,
        .memo_capacity = options.memo_capacity,
        .node_budget = options.node_budget,
    });
    defer manager.deinit();

    const components = try gpa.alloc(Component, partition.components.len);
    errdefer gpa.free(components);
    // Only the first `built` entries are initialized, so unwinding must stop
    // there rather than walking the whole allocation.
    var built: usize = 0;
    errdefer for (components[0..built]) |*c| {
        c.graph.deinit(gpa);
        gpa.free(c.bits);
    };

    for (partition.components) |component| {
        var ord = try Order.init(gpa, ir, component.vars, options.max_levels);
        defer ord.deinit(gpa);

        for (component.vars, 0..) |v, slot| slot_of_var[@intFromEnum(v)] = @intCast(slot);
        defer for (component.vars) |v| {
            slot_of_var[@intFromEnum(v)] = blast.no_slot;
        };

        const root = try blast.blast(
            gpa,
            &manager,
            ir,
            types,
            &ord,
            slot_of_var,
            component.stmts,
            .{ .max_mul_width = options.max_mul_width },
        );

        var graph = try sample.freeze(gpa, &manager, root, ord.levels);
        errdefer graph.deinit(gpa);
        // Everything this component built is now dead; reclaim it wholesale
        // before the next one starts. This is what stands in for a collector.
        manager.reset();

        const bits = try gpa.alloc(Component.Slot, ord.levels);
        for (bits, ord.bits) |*slot, b| slot.* = .{
            .v = @intFromEnum(component.vars[b.slot]),
            .bit = b.bit,
        };

        components[built] = .{ .graph = graph, .bits = bits };
        built += 1;

        // The component replaces its variables' free bits with its own count.
        self.log2_count -= @floatFromInt(ord.levels);
        self.log2_count += graph.log2Count();
        if (graph.isEmpty()) self.unsat = true;
    }

    self.components = components;
    if (self.unsat) self.log2_count = -std.math.inf(f64);
    return self;
}

pub fn deinit(self: *BddSolver, gpa: Allocator) void {
    for (self.components) |*c| {
        c.graph.deinit(gpa);
        gpa.free(c.bits);
    }
    gpa.free(self.components);
    gpa.free(self.vars);
    self.* = undefined;
}

pub fn solver(self: *BddSolver) Solver {
    return .{ .ptr = self, .vtable = &vtable };
}

const vtable: Solver.VTable = .{ .next = nextErased };

fn nextErased(ptr: *anyopaque, out: []Value) bool {
    const self: *BddSolver = @ptrCast(@alignCast(ptr));
    return self.next(out);
}

/// True when the constraints admit no assignment at all. Unlike an incomplete
/// engine, this is a decided answer, not a budget having run out.
pub fn isUnsat(self: *const BddSolver) bool {
    return self.unsat;
}

/// `log2` of the exact number of satisfying assignments, or `-inf` if there are
/// none. Free from the build — the same counts drive sampling.
pub fn log2Count(self: *const BddSolver) f64 {
    return self.log2_count;
}

/// Draw one satisfying assignment into `out`, which must hold at least
/// `Solver.valueLimbs(ir) * ir.vars.len` limbs.
///
/// Returns `false` only when the constraints are unsatisfiable. There is no
/// attempt budget and no failure mode in between.
pub fn next(self: *BddSolver, out: []Value) bool {
    const total = self.value_limbs * self.ir.vars.len;
    std.debug.assert(out.len >= total);
    if (self.unsat) return false;

    // Fill everything with uniform bits first. Unconstrained variables are then
    // already correct, and so is every level a component's diagram skips — so
    // the walks below only need to touch the bits their path actually decides.
    for (self.vars, 0..) |info, v| {
        const region = out[v * self.value_limbs ..][0..self.value_limbs];
        for (region[0..info.used]) |*limb| limb.* = self.prng.next();
        region[info.used - 1] &= info.top_mask;
        @memset(region[info.used..], 0);
    }

    for (self.components) |*c| {
        if (c.graph.levels == 0) continue; // constant statements: nothing to draw
        var writer: Writer = .{ .out = out, .limbs = self.value_limbs, .bits = c.bits };
        sample.walk(&c.graph, &self.prng, &writer, Writer.setBit);
    }
    return true;
}

/// Routes a sampled level back to the output limb holding that variable bit.
const Writer = struct {
    out: []Value,
    limbs: usize,
    bits: []const Component.Slot,

    fn setBit(w: *Writer, level: u32, value: bool) void {
        const slot = w.bits[level];
        const index = slot.v * w.limbs + slot.bit / 64;
        const mask = @as(u64, 1) << @intCast(slot.bit % 64);
        if (value) w.out[index] |= mask else w.out[index] &= ~mask;
    }
};

// -- Tests -------------------------------------------------------------------

const testing = std.testing;
const Type = Ir.Type;
const RejectionSampler = @import("RejectionSampler.zig");

fn constrain(gpa: Allocator, ir: *Ir, node: Ir.Node.Index) !void {
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{node});
}

test "range membership draws only in-range values" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const membership = try ir.in(gpa, try ir.varRef(gpa, x), &.{
        try ir.range(gpa, try ir.constInt(gpa, 3, Type.bit(4)), try ir.constInt(gpa, 7, Type.bit(4))),
    });
    try constrain(gpa, &ir, membership);

    var s = try BddSolver.init(gpa, &ir, .{ .seed = 0x1234 });
    defer s.deinit(gpa);

    // 5 values in [3,7].
    try testing.expectApproxEqAbs(@log2(5.0), s.log2Count(), 1e-9);

    var seen = [_]bool{false} ** 16;
    var out: [1]Value = undefined;
    for (0..500) |_| {
        try testing.expect(s.next(&out));
        try testing.expect(out[0] >= 3 and out[0] <= 7);
        seen[out[0]] = true;
    }
    for (3..8) |v| try testing.expect(seen[v]);
}

test "signed comparison agrees with two's complement" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    try constrain(gpa, &ir, try ir.binary(
        gpa,
        .slt,
        try ir.varRef(gpa, x),
        try ir.constInt(gpa, 0, Type.bit(8)),
    ));

    var s = try BddSolver.init(gpa, &ir, .{ .seed = 3 });
    defer s.deinit(gpa);

    try testing.expectApproxEqAbs(@log2(128.0), s.log2Count(), 1e-9);

    var out: [1]Value = undefined;
    for (0..500) |_| {
        try testing.expect(s.next(&out));
        try testing.expect(@as(i8, @bitCast(@as(u8, @intCast(out[0])))) < 0);
    }
}

test "addition relates three variables" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // x + y == z, all 5-bit, wrapping.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(5), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(5), .kind = .rand });
    const z = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(5), .kind = .rand });
    const sum = try ir.binary(gpa, .add, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    try constrain(gpa, &ir, try ir.binary(gpa, .eq, sum, try ir.varRef(gpa, z)));

    var s = try BddSolver.init(gpa, &ir, .{ .seed = 5 });
    defer s.deinit(gpa);

    // z is determined by x and y: 32*32 solutions.
    try testing.expectApproxEqAbs(@log2(1024.0), s.log2Count(), 1e-9);

    var out: [3]Value = undefined;
    for (0..500) |_| {
        try testing.expect(s.next(&out));
        try testing.expectEqual((out[0] + out[1]) & 31, out[2]);
    }
}

test "unsatisfiable constraints are decided, not merely unlucky" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(8), .kind = .rand });
    const gt = try ir.binary(gpa, .ugt, try ir.varRef(gpa, x), try ir.constInt(gpa, 200, Type.bit(8)));
    const lt = try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.constInt(gpa, 100, Type.bit(8)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ gt, lt });

    var s = try BddSolver.init(gpa, &ir, .{});
    defer s.deinit(gpa);

    try testing.expect(s.isUnsat());
    try testing.expect(s.log2Count() == -std.math.inf(f64));
    var out: [1]Value = undefined;
    try testing.expect(!s.next(&out));
}

test "a needle in a 32-bit haystack, where rejection sampling gives up" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // A 32-bit address in a 4 KiB window, 64-byte aligned: 64 solutions out of
    // 2^32. Rejection sampling accepts about one draw in 67 million.
    const addr = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(32), .kind = .rand });
    const window = try ir.in(gpa, try ir.varRef(gpa, addr), &.{
        try ir.range(
            gpa,
            try ir.constInt(gpa, 0x1000, Type.bit(32)),
            try ir.constInt(gpa, 0x1fff, Type.bit(32)),
        ),
    });
    const aligned = try ir.binary(
        gpa,
        .eq,
        try ir.binary(gpa, .band, try ir.varRef(gpa, addr), try ir.constInt(gpa, 63, Type.bit(32))),
        try ir.constInt(gpa, 0, Type.bit(32)),
    );
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ window, aligned });

    var s = try BddSolver.init(gpa, &ir, .{ .seed = 77 });
    defer s.deinit(gpa);

    // 4096 bytes / 64-byte alignment = 64 addresses.
    try testing.expectApproxEqAbs(@log2(64.0), s.log2Count(), 1e-9);

    var out: [1]Value = undefined;
    var seen = std.AutoHashMap(u64, void).init(gpa);
    defer seen.deinit();
    for (0..2000) |_| {
        try testing.expect(s.next(&out));
        try testing.expect(out[0] >= 0x1000 and out[0] <= 0x1fff);
        try testing.expectEqual(@as(u64, 0), out[0] & 63);
        try seen.put(out[0], {});
    }
    // Uniform sampling reaches every one of the 64 solutions.
    try testing.expectEqual(@as(usize, 64), seen.count());

    // The same constraints defeat the rejection sampler outright.
    var rs = try RejectionSampler.init(gpa, &ir, .{ .seed = 77, .max_attempts = 10_000 });
    defer rs.deinit(gpa);
    try testing.expect(!rs.next(&out));
}

test "independent components are sampled independently" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // Three unrelated constraints plus one entirely free variable.
    const a = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(6), .kind = .rand });
    const b = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(6), .kind = .rand });
    const c = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(6), .kind = .rand });
    _ = try ir.addVariable(gpa, .{ .id = @enumFromInt(3), .ty = Type.bit(6), .kind = .rand });

    const ca = try ir.binary(gpa, .ult, try ir.varRef(gpa, a), try ir.constInt(gpa, 4, Type.bit(6)));
    const cb = try ir.binary(gpa, .ugt, try ir.varRef(gpa, b), try ir.constInt(gpa, 60, Type.bit(6)));
    const cc = try ir.binary(gpa, .eq, try ir.varRef(gpa, c), try ir.constInt(gpa, 42, Type.bit(6)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ ca, cb, cc });

    var s = try BddSolver.init(gpa, &ir, .{ .seed = 11 });
    defer s.deinit(gpa);

    // 4 * 3 * 1 * 64 — the free variable contributes its full range.
    try testing.expectApproxEqAbs(@log2(4.0 * 3.0 * 1.0 * 64.0), s.log2Count(), 1e-9);

    var out: [4]Value = undefined;
    var free_seen = [_]bool{false} ** 64;
    for (0..1000) |_| {
        try testing.expect(s.next(&out));
        try testing.expect(out[0] < 4);
        try testing.expect(out[1] > 60);
        try testing.expectEqual(@as(Value, 42), out[2]);
        free_seen[@intCast(out[3])] = true;
    }
    for (free_seen) |hit| try testing.expect(hit);
}

test "solves through the Solver interface" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const v = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(6), .kind = .rand });
    try constrain(gpa, &ir, try ir.binary(
        gpa,
        .eq,
        try ir.varRef(gpa, v),
        try ir.constInt(gpa, 42, Type.bit(6)),
    ));

    var s = try BddSolver.init(gpa, &ir, .{ .seed = 1 });
    defer s.deinit(gpa);
    const erased: Solver = s.solver();

    var out: [1]Value = undefined;
    try testing.expect(erased.next(&out));
    try testing.expectEqual(@as(Value, 42), out[0]);
}

test "a wide variable-by-variable multiply is refused, a narrow one is not" {
    const gpa = testing.allocator;

    // 12-bit multiply: within the default guard.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(6), .kind = .rand });
        const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(6), .kind = .rand });
        const product = try ir.binary(gpa, .mul, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
        try constrain(gpa, &ir, try ir.binary(gpa, .eq, product, try ir.constInt(gpa, 12, Type.bit(6))));

        var s = try BddSolver.init(gpa, &ir, .{ .seed = 2 });
        defer s.deinit(gpa);

        var out: [2]Value = undefined;
        for (0..200) |_| {
            try testing.expect(s.next(&out));
            try testing.expectEqual(@as(Value, 12), (out[0] * out[1]) & 63);
        }
    }

    // 32-bit multiply: provably hopeless for any variable order, so refused.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(32), .kind = .rand });
        const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(32), .kind = .rand });
        const product = try ir.binary(gpa, .mul, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
        try constrain(gpa, &ir, try ir.binary(gpa, .eq, product, try ir.constInt(gpa, 12, Type.bit(32))));

        try testing.expectError(error.OperandTooWide, BddSolver.init(gpa, &ir, .{}));
    }
}

test "unsupported nodes are rejected rather than silently ignored" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const start: u32 = @intCast(ir.extra.items.len);
    _ = try ir.addExtra(gpa, &.{ 1, @intFromEnum(try ir.varRef(gpa, x)) });
    const dist = try ir.addNode(gpa, .{
        .tag = .dist,
        .data = .{ .lhs = @intFromEnum(try ir.varRef(gpa, x)), .rhs = start },
    });
    try constrain(gpa, &ir, dist);

    try testing.expectError(error.UnsupportedNode, BddSolver.init(gpa, &ir, .{}));
}

test "the node budget fails the build instead of exhausting memory" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(12), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(12), .kind = .rand });
    const product = try ir.binary(gpa, .mul, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    try constrain(gpa, &ir, try ir.binary(gpa, .eq, product, try ir.constInt(gpa, 1, Type.bit(12))));

    try testing.expectError(error.NodeBudgetExceeded, BddSolver.init(gpa, &ir, .{ .node_budget = 256 }));
}

test "an oversized component is rejected up front" {
    const gpa = testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(64), .kind = .rand });
    try constrain(gpa, &ir, try ir.binary(
        gpa,
        .ult,
        try ir.varRef(gpa, x),
        try ir.constInt(gpa, 5, Type.bit(64)),
    ));

    try testing.expectError(error.TooManyBits, BddSolver.init(gpa, &ir, .{ .max_levels = 32 }));
}

// -- Differential testing against RejectionSampler ---------------------------
//
// The two engines are completely different — one interprets the IR over a
// concrete draw, the other compiles it into a decision diagram — so agreeing on
// a solution set is strong evidence both are right. It is also the only
// practical way to pin down the operator edge cases (division by zero, `sra`'s
// sign width, saturating shift amounts, truncating signed division) without
// re-implementing those semantics a third time in the test itself.
//
// Both engines sample uniformly over the solution set, so with enough draws
// each recovers it in full and the sets can be compared exactly.

const Solutions = std.AutoHashMap(u64, void);

/// Pack an assignment of `n` variables, each at most 8 bits wide, into a key.
fn assignmentKey(out: []const Value, n: usize) u64 {
    var key: u64 = 0;
    for (0..n) |v| key |= (out[v] & 0xff) << @intCast(8 * v);
    return key;
}

/// Every solution both engines can find, checked for exact agreement against
/// each other and against the BDD's own model count.
fn expectSameSolutions(gpa: Allocator, ir: *const Ir, n: usize) !void {
    var bdd = try BddSolver.init(gpa, ir, .{ .seed = 0xB0D });
    defer bdd.deinit(gpa);

    var rejection = try RejectionSampler.init(gpa, ir, .{ .seed = 0xB0D, .max_attempts = 200_000 });
    defer rejection.deinit(gpa);

    const out = try gpa.alloc(Value, n);
    defer gpa.free(out);

    if (bdd.isUnsat()) {
        // The rejection sampler cannot prove this, but it must not find one.
        try testing.expect(!rejection.next(out));
        return;
    }

    const count = @exp2(bdd.log2Count());
    try testing.expect(count >= 1);
    // Enough draws that missing any one solution is vanishingly unlikely.
    const draws: usize = @intFromFloat(30.0 * count + 200.0);

    var from_bdd = Solutions.init(gpa);
    defer from_bdd.deinit();
    for (0..draws) |_| {
        try testing.expect(bdd.next(out));
        try from_bdd.put(assignmentKey(out, n), {});
    }

    var from_rejection = Solutions.init(gpa);
    defer from_rejection.deinit();
    for (0..draws) |_| {
        try testing.expect(rejection.next(out));
        try from_rejection.put(assignmentKey(out, n), {});
    }

    // The BDD's count is exact, so it is the arbiter of how many there are.
    try testing.expectEqual(@as(usize, @intFromFloat(@round(count))), from_bdd.count());
    try testing.expectEqual(from_bdd.count(), from_rejection.count());
    var it = from_rejection.keyIterator();
    while (it.next()) |key| try testing.expect(from_bdd.contains(key.*));
}

/// `constraint { op(x, y) == k }` over two `width`-bit variables.
fn binaryEqIr(gpa: Allocator, ir: *Ir, tag: Ir.Node.Tag, k: u64, width: u16) !void {
    const ty = Type.bit(width);
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = ty, .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = ty, .kind = .rand });
    const op = try ir.binary(gpa, tag, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    try constrain(gpa, ir, try ir.binary(gpa, .eq, op, try ir.constInt(gpa, k, ty)));
}

test "differential: arithmetic operators agree with the rejection sampler" {
    const gpa = testing.allocator;
    const tags = [_]Ir.Node.Tag{
        .add,  .sub, .mul,  .udiv, .umod, .sdiv, .smod,
        .band, .bor, .bxor, .sll,  .srl,  .sra,
    };
    // 0 exercises the division-by-zero path; 15 exercises the all-ones edge.
    for (tags) |tag| {
        for ([_]u64{ 0, 1, 7, 15 }) |k| {
            var ir: Ir = .{};
            defer ir.deinit(gpa);
            try binaryEqIr(gpa, &ir, tag, k, 4);
            expectSameSolutions(gpa, &ir, 2) catch |err| {
                std.debug.print("mismatch on {t} == {d}\n", .{ tag, k });
                return err;
            };
        }
    }
}

test "differential: comparisons agree with the rejection sampler" {
    const gpa = testing.allocator;
    const tags = [_]Ir.Node.Tag{
        .eq, .ne, .slt, .ult, .sle, .ule, .sgt, .ugt, .sge, .uge,
    };
    for (tags) |tag| {
        var ir: Ir = .{};
        defer ir.deinit(gpa);

        const ty = Type.bit(4);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = ty, .kind = .rand });
        const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = ty, .kind = .rand });
        try constrain(gpa, &ir, try ir.binary(gpa, tag, try ir.varRef(gpa, x), try ir.varRef(gpa, y)));

        expectSameSolutions(gpa, &ir, 2) catch |err| {
            std.debug.print("mismatch on comparison {t}\n", .{tag});
            return err;
        };
    }
}

test "differential: casts, unary and logical operators agree" {
    const gpa = testing.allocator;

    // sext to a wider width, then compare — catches sign-extension mistakes.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
        const wide = try ir.sext(gpa, try ir.varRef(gpa, x), 8);
        try constrain(gpa, &ir, try ir.binary(gpa, .ugt, wide, try ir.constInt(gpa, 0xf0, Type.bit(8))));
        try expectSameSolutions(gpa, &ir, 1);
    }

    // trunc back down, and zext of a negated value.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(6), .kind = .rand });
        const low = try ir.trunc(gpa, try ir.varRef(gpa, x), 3);
        const negated = try ir.unary(gpa, .neg, low);
        try constrain(gpa, &ir, try ir.binary(gpa, .eq, negated, try ir.constInt(gpa, 5, Type.bit(3))));
        try expectSameSolutions(gpa, &ir, 1);
    }

    // Bitwise complement and the logical connectives over truthiness.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
        const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
        const nx = try ir.unary(gpa, .bnot, try ir.varRef(gpa, x));
        const lhs = try ir.binary(gpa, .ult, nx, try ir.varRef(gpa, y));
        const rhs = try ir.binary(gpa, .ne, try ir.varRef(gpa, y), try ir.constInt(gpa, 0, Type.bit(4)));
        try constrain(gpa, &ir, try ir.binary(gpa, .implies, lhs, rhs));
        try constrain(gpa, &ir, try ir.unary(gpa, .lnot, try ir.binary(
            gpa,
            .iff,
            try ir.varRef(gpa, x),
            try ir.varRef(gpa, y),
        )));
        try expectSameSolutions(gpa, &ir, 2);
    }

    // Multi-member `in`, mixing bare values with ranges.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(5), .kind = .rand });
        const ty = Type.bit(5);
        try constrain(gpa, &ir, try ir.in(gpa, try ir.varRef(gpa, x), &.{
            try ir.constInt(gpa, 1, ty),
            try ir.range(gpa, try ir.constInt(gpa, 10, ty), try ir.constInt(gpa, 14, ty)),
            try ir.constInt(gpa, 31, ty),
        }));
        try expectSameSolutions(gpa, &ir, 1);
    }
}

// -- Structural constraints the rejection sampler cannot evaluate -------------

test "if_else and unique, which the rejection sampler panics on" {
    const gpa = testing.allocator;

    // if (x < 4) y == 1; else y == 2;
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const ty = Type.bit(4);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = ty, .kind = .rand });
        const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = ty, .kind = .rand });

        const cond = try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.constInt(gpa, 4, ty));
        const then_stmt = try ir.binary(gpa, .eq, try ir.varRef(gpa, y), try ir.constInt(gpa, 1, ty));
        const else_stmt = try ir.binary(gpa, .eq, try ir.varRef(gpa, y), try ir.constInt(gpa, 2, ty));
        const start = try ir.addExtra(gpa, &.{ @intFromEnum(then_stmt), @intFromEnum(else_stmt) });
        try constrain(gpa, &ir, try ir.addNode(gpa, .{
            .tag = .if_else,
            .data = .{ .lhs = @intFromEnum(cond), .rhs = @intFromEnum(start) },
        }));

        var s = try BddSolver.init(gpa, &ir, .{ .seed = 21 });
        defer s.deinit(gpa);
        // y is pinned either way, so there are exactly 16 solutions: one per x.
        try testing.expectApproxEqAbs(@log2(16.0), s.log2Count(), 1e-9);

        var out: [2]Value = undefined;
        for (0..300) |_| {
            try testing.expect(s.next(&out));
            try testing.expectEqual(@as(Value, if (out[0] < 4) 1 else 2), out[1]);
        }
    }

    // unique { x, y, z } over 2-bit variables.
    {
        var ir: Ir = .{};
        defer ir.deinit(gpa);
        const ty = Type.bit(2);
        const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = ty, .kind = .rand });
        const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = ty, .kind = .rand });
        const z = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = ty, .kind = .rand });
        const refs = [_]u32{
            @intFromEnum(try ir.varRef(gpa, x)),
            @intFromEnum(try ir.varRef(gpa, y)),
            @intFromEnum(try ir.varRef(gpa, z)),
        };
        const start = try ir.addExtra(gpa, &.{ 3, refs[0], refs[1], refs[2] });
        try constrain(gpa, &ir, try ir.addNode(gpa, .{
            .tag = .unique,
            .data = .{ .lhs = @intFromEnum(start) },
        }));

        var s = try BddSolver.init(gpa, &ir, .{ .seed = 22 });
        defer s.deinit(gpa);
        // Ordered triples of distinct values from 4: 4 * 3 * 2 = 24.
        try testing.expectApproxEqAbs(@log2(24.0), s.log2Count(), 1e-9);

        var out: [3]Value = undefined;
        for (0..300) |_| {
            try testing.expect(s.next(&out));
            try testing.expect(out[0] != out[1] and out[1] != out[2] and out[0] != out[2]);
        }
    }
}

test {
    testing.refAllDecls(@This());
}

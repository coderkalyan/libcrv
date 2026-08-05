//! An SMT-backed almost-uniform sampler, built on Bitwuzla.
//!
//! This engine exists for the constraint sets the cheap ones cannot handle:
//! sparse solution spaces where rejection sampling never lands a hit, and wide
//! datapath arithmetic where a decision diagram blows up. It is meant to be
//! dispatched to for those, not to be the first thing tried.
//!
//! ## The shape of the thing
//!
//! A solver call per `randomize()` would be both far too slow for a simulation
//! loop and badly distributed — solvers answer with whatever corner of the
//! space their heuristics reach first, and re-seeding does not fix that. So
//! Bitwuzla runs as a *compiler*, not an interpreter: all the expensive work
//! happens once in `init`, and `next` draws from what it produced.
//!
//! `init` runs four phases:
//!
//!   1. **Encode** the IR into Bitwuzla terms (`smt/Encoder.zig`).
//!   2. **Shrink** the problem to an independent support — the bits that
//!      actually have to be reasoned about (`smt/Support.zig`, on top of
//!      `Analysis`).
//!   3. **Count** the solutions with ApproxMC (`smt/Hashing.zig`). Its
//!      zero-hash step is exhaustive enumeration, so a small solution set comes
//!      back already listed and exactly counted — the "just enumerate it" case
//!      needs no separate code path.
//!   4. **Aim** the hash width so that one cell holds a workable batch.
//!
//! `next` is then either a table index (small sets) or a draw from a batch,
//! refilled by enumerating one random cell. Free bits — the ones `Analysis`
//! proved no constraint observes — are redrawn from the RNG on *every* sample
//! rather than read back from a model, which is both faster and the only
//! correct thing to do: a model would repeat whatever Bitwuzla happened to pick
//! for bits nothing pins.
//!
//! ## What it promises
//!
//! `guarantee()` reports which of three regimes produced the samples, and it is
//! fixed once `init` returns. `.exact` is uniform, full stop. `.almost_uniform`
//! means every individual solution is drawn with probability within a `1 + eps`
//! factor of `1/|S|`, with confidence `1 - delta` — a per-solution bound, not a
//! claim about aggregate distance. `.biased` means the guarantees were
//! abandoned to make progress, and it is off by default: an instance that
//! defeats the pipeline fails `init` rather than quietly returning a stream
//! nobody asked for.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");
const Solver = @import("Solver.zig");
const Analysis = @import("Analysis.zig");
const Encoder = @import("smt/Encoder.zig");
const Support = @import("smt/Support.zig");
const Hashing = @import("smt/Hashing.zig");
const bw = @import("bitwuzla");

const Value = Solver.Value;
const Limb = std.math.big.Limb;
const limb_bits = @bitSizeOf(Limb);

const SmtSampler = @This();

pub const Error = error{
    /// Built without `-Dbitwuzla`, so there is no backend to run on.
    BackendUnavailable,
    /// The IR uses a node this engine does not model. `dist`, `solve_before`,
    /// and `foreach` each change the distribution a sampler ought to produce,
    /// so they are rejected rather than silently ignored.
    Unsupported,
    /// The constraints have no solution. Unlike a rejection sampler's failure,
    /// this is a proof.
    Unsat,
    /// The pipeline could not resolve the instance within its budget, and
    /// `allow_biased_fallback` was not set.
    BudgetExceeded,
} || Allocator.Error;

/// What the sample stream is actually distributed like. Decided by `init` and
/// constant thereafter, so callers read it once.
pub const Guarantee = union(enum) {
    /// Exactly uniform: the solution set was enumerated in full.
    exact,
    /// Each solution's probability lies within a `1 + epsilon` factor of
    /// `1/|S|`, with confidence `1 - delta`.
    almost_uniform: struct { epsilon: f64, delta: f64 },
    /// No distributional guarantee at all.
    biased,
};

pub const Options = struct {
    seed: u64 = 0,

    /// Multiplicative tolerance on each solution's probability.
    epsilon: f64 = 0.8,
    /// Failure probability for the count. Reaching it needs
    /// `ceil(17 log2(3/delta))` rounds — 84 at the default — so `max_rounds`
    /// usually binds first, and the confidence actually delivered is reported
    /// by `guarantee()` rather than assumed.
    delta: f64 = 0.1,
    /// Ceiling on counting rounds.
    max_rounds: u32 = 17,

    /// Samples handed out per enumerated cell. Larger is faster; the samples
    /// within one batch come from a single cell without replacement, so they
    /// are negatively correlated with each other even though each remains
    /// individually almost-uniform. Set to 1 for fully independent draws.
    samples_per_cell: u32 = 8,

    /// Per-bit inclusion probability in each parity constraint. One half gives
    /// the 2-universal family the guarantees are stated for; lower values make
    /// the constraints shorter and cheaper at the cost of weakening them.
    xor_density: f64 = 0.5,

    /// Enumerate outright and sample by index when the solution set is at most
    /// this large. Costs nothing to attempt — the counting phase's first step
    /// is exactly this enumeration — and buys exact uniformity when it lands.
    exact_limit: u32 = 4096,

    /// Ceiling on Padoa definability queries during support minimization.
    support_queries: u32 = 10_000,
    /// Retries allowed when a cell comes back empty or overfull.
    cell_retries: u32 = 16,

    /// Which SAT engine Bitwuzla bit-blasts onto. CryptoMiniSat is the default
    /// because it recovers XOR structure from CNF and can run Gauss-Jordan over
    /// it alongside CDCL, which is what makes parity constraints affordable;
    /// libbitwuzla must be built with CryptoMiniSat support to use it.
    sat_solver: bw.SatSolver = .cms,

    /// Values pinning `.state` variables, laid out like a solution buffer.
    /// When null they are left free, matching what `RejectionSampler` does by
    /// drawing them along with everything else.
    state: ?[]const Value = null,

    /// Permit falling back to an unguaranteed stream instead of failing. Off by
    /// default: silently degrading from "uniform" to "whatever the solver
    /// produced" is the failure mode this engine exists to avoid.
    allow_biased_fallback: bool = false,
};

ir: *const Ir,
prng: std.Random.DefaultPrng,
options: Options,
guarantee_: Guarantee,

value_limbs: usize,
/// Limbs per whole solution (`value_limbs * ir.vars.len`).
stride: usize,
/// Bits no constraint observes, in solution layout: set bits are redrawn on
/// every sample.
free_mask: []Limb,

analysis: Analysis,
support: Support,
tm: bw.TermManager,
bw_options: bw.Options,
session: bw.Session,
encoder: Encoder,
xor_terms: []bw.Term,
bv1: bw.Sort,

/// Solutions available to hand out, `stride` limbs each.
batch: []Limb,
batch_len: usize,
batch_pos: usize,
/// How many solutions `batch` can hold.
batch_capacity: usize,

/// Parity constraints per cell, once the count has aimed it.
hash_width: u16,
/// Set when `batch` holds the entire solution set and never needs refilling.
enumerated: bool,

pub fn init(gpa: Allocator, ir: *const Ir, options: Options) Error!SmtSampler {
    if (!bw.available) return error.BackendUnavailable;

    const value_limbs = Solver.valueLimbs(ir);
    const stride = value_limbs * ir.vars.len;

    var analysis = try Analysis.run(gpa, ir, .{});
    errdefer analysis.deinit(gpa);

    // Exactly `stride`, not one more: `next` iterates it alongside the output
    // and a solution slot, and the three lengths have to agree.
    const free_mask = try gpa.alloc(Limb, stride);
    errdefer gpa.free(free_mask);
    fillFreeMask(ir, &analysis, free_mask, value_limbs);

    const tm = bw.TermManager.init();
    errdefer tm.deinit();

    const bw_options = bw.Options.init();
    errdefer bw_options.deinit();
    bw_options.setProduceModels(true);
    bw_options.setSeed(@truncate(options.seed));
    bw_options.setSatSolver(options.sat_solver);

    const encoder_options: Encoder.Options = .{ .state = options.state };

    // Support minimization builds its own session over two copies of the
    // formula; both live on `tm` and are released with it.
    var support = try Support.compute(gpa, ir, &analysis, tm, encoder_options, .{
        .max_queries = options.support_queries,
    });
    errdefer support.deinit(gpa);

    var encoder = try Encoder.encode(gpa, ir, tm, encoder_options);
    errdefer encoder.deinit(gpa);

    const session = bw.Session.init(tm, bw_options);
    errdefer session.deinit();
    encoder.assertAll(session);

    const xor_terms = try gpa.alloc(bw.Term, @max(support.bits.len, 1));
    errdefer gpa.free(xor_terms);

    const pivot = Hashing.pivotFor(options.epsilon);
    const capacity = @max(@as(usize, pivot) + 1, @as(usize, options.exact_limit));
    const batch = try gpa.alloc(Limb, capacity * @max(stride, 1));
    errdefer gpa.free(batch);

    var self: SmtSampler = .{
        .ir = ir,
        .prng = .init(options.seed),
        .options = options,
        .guarantee_ = .biased,
        .value_limbs = value_limbs,
        .stride = stride,
        .free_mask = free_mask,
        .analysis = analysis,
        .support = support,
        .tm = tm,
        .bw_options = bw_options,
        .session = session,
        .encoder = encoder,
        .xor_terms = xor_terms,
        .bv1 = tm.bvSort(1),
        .batch = batch,
        .batch_len = 0,
        .batch_pos = 0,
        .batch_capacity = capacity,
        .hash_width = 0,
        .enumerated = false,
    };

    try self.measure(gpa);
    return self;
}

/// The hashing context, rebuilt per use.
///
/// Deliberately *not* a stored field: it holds a pointer into this struct, and
/// `init` returns by value, so a cached one would point at the dead local the
/// moment it was handed back.
fn hashCtx(self: *SmtSampler) Hashing.Ctx {
    return .{
        .ir = self.ir,
        .enc = &self.encoder,
        .session = self.session,
        .tm = self.tm,
        .support = self.support.bits,
        .value_limbs = self.value_limbs,
        .stride = self.stride,
        .terms = self.xor_terms,
        .bv1 = self.bv1,
    };
}

/// Phases 3 and 4: count, then decide how the stream will be produced.
fn measure(self: *SmtSampler, gpa: Allocator) Error!void {
    const pivot = Hashing.pivotFor(self.options.epsilon);
    const rounds = @min(self.options.max_rounds, Hashing.roundsFor(self.options.delta));

    const estimates = try gpa.alloc(Hashing.Count, @max(rounds, 1));
    defer gpa.free(estimates);

    const ctx = self.hashCtx();
    const result = Hashing.count(&ctx, self.prng.random(), .{
        .pivot = pivot,
        .exact_limit = @intCast(@min(self.options.exact_limit, self.batch_capacity)),
        .density = self.options.xor_density,
    }, self.batch, estimates);

    switch (result) {
        .unsat => return error.Unsat,

        // Already enumerated in full by the zero-hash step and sitting in
        // `batch`: nothing more to do, and the answer is exact.
        .exact => |n| {
            self.batch_len = n;
            self.enumerated = true;
            self.guarantee_ = .exact;
        },

        .approx => |c| {
            self.hash_width = Hashing.aimFor(c, @max(pivot / 2, 1));
            self.guarantee_ = .{ .almost_uniform = .{
                .epsilon = self.options.epsilon,
                // Report the confidence the rounds actually bought, not the
                // one that was asked for.
                .delta = @max(self.options.delta, Hashing.deltaFor(rounds)),
            } };
        },

        .indeterminate => {
            if (!self.options.allow_biased_fallback) return error.BudgetExceeded;
            self.hash_width = 0;
            self.guarantee_ = .biased;
        },
    }
}

/// Mark the bits of each randomized variable that no constraint observes.
fn fillFreeMask(ir: *const Ir, analysis: *const Analysis, mask: []Limb, value_limbs: usize) void {
    @memset(mask, 0);
    const kinds = ir.vars.items(.kind);
    const types = ir.vars.items(.ty);
    for (0..ir.vars.len) |vi| {
        // A state variable is an input, not something to redraw.
        if (kinds[vi] == .state) continue;

        const v: Ir.Variable.Index = @enumFromInt(@as(u32, @intCast(vi)));
        const width: u16 = if (types[vi].width == 0) Ir.default_width else types[vi].width;
        const region = mask[vi * value_limbs ..][0..value_limbs];
        for (0..width) |b| {
            const bit: u16 = @intCast(b);
            if (!analysis.isFree(v, bit)) continue;
            region[@as(usize, bit) / limb_bits] |= @as(Limb, 1) << @intCast(bit % limb_bits);
        }
    }
}

pub fn deinit(self: *SmtSampler, gpa: Allocator) void {
    gpa.free(self.batch);
    gpa.free(self.xor_terms);
    self.encoder.deinit(gpa);
    self.session.deinit();
    self.bw_options.deinit();
    // Releases every sort and term the encodings and hashes created.
    self.tm.deinit();
    self.support.deinit(gpa);
    self.analysis.deinit(gpa);
    gpa.free(self.free_mask);
    self.* = undefined;
}

pub fn guarantee(self: *const SmtSampler) Guarantee {
    return self.guarantee_;
}

pub fn solver(self: *SmtSampler) Solver {
    return .{ .ptr = self, .vtable = &vtable };
}

const vtable: Solver.VTable = .{ .next = nextErased };

fn nextErased(ptr: *anyopaque, out: []Value) bool {
    const self: *SmtSampler = @ptrCast(@alignCast(ptr));
    return self.next(out);
}

/// Draw one satisfying assignment into `out`, which must hold at least
/// `Solver.valueLimbs(ir) * ir.vars.len` limbs.
///
/// False means no draw could be produced — every cell retry came back empty or
/// overfull, or the solver gave up. It never means "unsatisfiable": that was
/// settled in `init`.
pub fn next(self: *SmtSampler, out: []Value) bool {
    std.debug.assert(out.len >= self.stride);
    const rand = self.prng.random();

    const slot = if (self.enumerated) blk: {
        // The whole solution set is in hand, so an index picked uniformly is
        // exactly a uniform sample. Reading it in order would instead produce
        // a fixed cycle.
        if (self.batch_len == 0) return false;
        const i = rand.uintLessThan(usize, self.batch_len);
        break :blk self.batch[i * self.stride ..][0..self.stride];
    } else blk: {
        if (self.batch_pos >= self.batch_len and !self.refill()) return false;
        defer self.batch_pos += 1;
        break :blk self.batch[self.batch_pos * self.stride ..][0..self.stride];
    };

    // A model pins only the bits something observes. Everything else is drawn
    // fresh here, which is what keeps the free dimensions uniform rather than
    // frozen at whatever the solver last chose.
    for (out[0..self.stride], slot, self.free_mask) |*dst, fixed, free| {
        dst.* = (fixed & ~free) | (rand.int(Limb) & free);
    }
    return true;
}

/// Draw a fresh cell and stock the batch from it.
fn refill(self: *SmtSampler) bool {
    const pivot = Hashing.pivotFor(self.options.epsilon);
    const limit = @min(@as(usize, pivot) + 1, self.batch_capacity);
    const rand = self.prng.random();
    const ctx = self.hashCtx();

    var tries: u32 = 0;
    while (tries <= self.options.cell_retries) : (tries += 1) {
        const cell = Hashing.boundedSat(
            &ctx,
            rand,
            self.hash_width,
            self.options.xor_density,
            limit,
            self.batch,
        );
        switch (cell.status) {
            // Overfull: the hash was too coarse, so split further.
            .truncated => {
                self.hash_width +|= 1;
                continue;
            },
            .unknown => return false,
            .exhausted => {},
        }
        if (cell.found == 0) {
            // Empty: the hash was too fine. Back off, but never past zero.
            if (self.hash_width == 0) return false;
            self.hash_width -= 1;
            continue;
        }

        // Shuffle so the prefix handed out is a uniform subset of the cell
        // rather than whatever order the solver enumerated in.
        self.shuffleBatch(rand, cell.found);
        self.batch_len = @min(cell.found, @max(self.options.samples_per_cell, 1));
        self.batch_pos = 0;
        return true;
    }
    return false;
}

fn shuffleBatch(self: *SmtSampler, rand: std.Random, len: usize) void {
    if (self.stride == 0) return;
    var i = len;
    while (i > 1) {
        i -= 1;
        const j = rand.uintLessThan(usize, i + 1);
        if (j == i) continue;
        const a = self.batch[i * self.stride ..][0..self.stride];
        const b = self.batch[j * self.stride ..][0..self.stride];
        for (a, b) |*x, *y| std.mem.swap(Limb, x, y);
    }
}

// -- Tests -------------------------------------------------------------------

test "reports the backend as missing when it is not linked" {
    if (bw.available) return error.SkipZigTest;

    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Ir.Type.bit(8), .kind = .rand });
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{
        try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.constInt(gpa, 10, Ir.Type.bit(8))),
    });

    try std.testing.expectError(error.BackendUnavailable, SmtSampler.init(gpa, &ir, .{}));
}

test {
    std.testing.refAllDecls(@This());
}

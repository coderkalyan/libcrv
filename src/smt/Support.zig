//! Independent support: a set of *bits* such that any two solutions agreeing on
//! all of them agree on every bit a constraint can observe.
//!
//! This is the load-bearing phase of the hashing pipeline, not an optimization.
//! Each random parity constraint added later covers the support, and an XOR's
//! cost to the SAT solver grows with its length, so the difference between
//! hashing over 400 bits and over 14 is the difference between a component that
//! samples and one that does not.
//!
//! Two stages, cheap first:
//!
//!   1. **Structural**, straight from `Analysis`: drop bits no constraint
//!      observes, and drop every bit of a variable pinned by a top-level
//!      equality. Costs one pass over the IR and no solver calls at all.
//!
//!   2. **Semantic**, by Padoa's method: bit `b` is definable from the rest iff
//!      two copies of the constraint set that agree everywhere in the support
//!      except `b`, and disagree at `b`, are unsatisfiable. Each such question
//!      is a single incremental query against a formula built once, so the
//!      whole refinement is one solver instance and N assumption checks.
//!
//! Both stages only ever *shrink* the support, and both are conservative in the
//! same direction: an inconclusive result keeps the bit. A support that is
//! larger than necessary costs time; one that is too small would silently break
//! uniformity, so every uncertain case resolves toward the former.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("../Ir.zig");
const Analysis = @import("../Analysis.zig");
const Encoder = @import("Encoder.zig");
const bw = @import("bitwuzla");

const Support = @This();

/// One bit of one variable.
pub const Bit = struct {
    v: Ir.Variable.Index,
    bit: u16,
};

/// The support, in variable-then-bit order.
bits: []Bit,
/// How many bits the structural stage alone left, before refinement.
structural_len: usize,
/// False when refinement stopped early on its query budget. The support is
/// still sound, just larger than it might have been.
complete: bool,

pub const Options = struct {
    /// Run Padoa refinement. Disabling it leaves the structural support, which
    /// is sound but usually much larger.
    refine: bool = true,
    /// Ceiling on definability queries. Refinement is O(candidates) queries,
    /// each a full SAT call, so a large component can spend real time here.
    max_queries: u32 = 10_000,
};

pub const Error = Encoder.Error;

pub fn deinit(s: *Support, gpa: Allocator) void {
    gpa.free(s.bits);
    s.* = undefined;
}

/// Compute the support. `tm` is used only if refinement runs; the encodings it
/// builds are owned by `tm` and released with it.
pub fn compute(
    gpa: Allocator,
    ir: *const Ir,
    analysis: *const Analysis,
    tm: bw.TermManager,
    encoder_options: Encoder.Options,
    options: Options,
) Error!Support {
    var bits: std.ArrayListUnmanaged(Bit) = .empty;
    errdefer bits.deinit(gpa);

    // Stage 1. A pinned variable is a function of the others, and an
    // unobserved bit is a function of nothing at all; neither needs hashing.
    const kinds = ir.vars.items(.kind);
    for (0..ir.vars.len) |vi| {
        const v: Ir.Variable.Index = @enumFromInt(@as(u32, @intCast(vi)));
        if (kinds[vi] == .state) continue;
        if (analysis.defined[vi]) continue;
        const width = varWidth(ir, v);
        for (0..width) |b| {
            const bit: u16 = @intCast(b);
            if (analysis.isFree(v, bit)) continue;
            try bits.append(gpa, .{ .v = v, .bit = bit });
        }
    }

    var s: Support = .{
        .bits = try bits.toOwnedSlice(gpa),
        .structural_len = 0,
        .complete = true,
    };
    s.structural_len = s.bits.len;
    errdefer gpa.free(s.bits);

    if (options.refine and bw.available and s.bits.len > 1) {
        try s.padoa(gpa, ir, tm, encoder_options, options);
    }
    return s;
}

/// Stage 2: drop every bit the others already determine.
///
/// Builds two independent encodings over the same term manager — `mk_const`
/// hands back a fresh constant each call, so encoding twice yields two disjoint
/// variable sets — asserts both, and gates each "these two agree here" and
/// "these two differ here" claim behind its own boolean. Every definability
/// question is then one `check_sat_assuming` over a formula that never has to
/// be rebuilt.
fn padoa(
    s: *Support,
    gpa: Allocator,
    ir: *const Ir,
    tm: bw.TermManager,
    encoder_options: Encoder.Options,
    options: Options,
) Error!void {
    const n = s.bits.len;

    var lhs = try Encoder.encode(gpa, ir, tm, encoder_options);
    defer lhs.deinit(gpa);
    var rhs = try Encoder.encode(gpa, ir, tm, encoder_options);
    defer rhs.deinit(gpa);

    const bool_sort = tm.boolSort();

    const agree = try gpa.alloc(bw.Term, n);
    defer gpa.free(agree);
    const differ = try gpa.alloc(bw.Term, n);
    defer gpa.free(differ);
    const assumptions = try gpa.alloc(bw.Term, n);
    defer gpa.free(assumptions);
    const keep = try gpa.alloc(bool, n);
    defer gpa.free(keep);
    @memset(keep, true);

    const options_handle = bw.Options.init();
    defer options_handle.deinit();
    const session = bw.Session.init(tm, options_handle);
    defer session.deinit();

    lhs.assertAll(session);
    rhs.assertAll(session);

    for (s.bits, 0..) |b, i| {
        const a = tm.extract(lhs.varTerm(b.v), b.bit, b.bit);
        const c = tm.extract(rhs.varTerm(b.v), b.bit, b.bit);
        agree[i] = tm.constant(bool_sort);
        differ[i] = tm.constant(bool_sort);
        session.assert(tm.term2(.implies, agree[i], tm.term2(.equal, a, c)));
        session.assert(tm.term2(.implies, differ[i], tm.term2(.distinct, a, c)));
    }

    var queries: u32 = 0;
    for (0..n) |i| {
        if (queries >= options.max_queries) {
            s.complete = false;
            break;
        }
        queries += 1;

        // "Pin every other surviving bit, and force this one to differ."
        var len: usize = 0;
        for (0..n) |j| {
            if (j != i and keep[j]) {
                assumptions[len] = agree[j];
                len += 1;
            }
        }
        assumptions[len] = differ[i];
        len += 1;

        // Unsatisfiable means no two solutions can disagree here while
        // agreeing everywhere else: the bit is determined, so drop it. `sat`
        // and `unknown` both keep it, which is the safe direction.
        if (session.checkSatAssuming(assumptions[0..len]) == .unsat) keep[i] = false;
    }

    var out: usize = 0;
    for (s.bits, keep) |b, k| {
        if (!k) continue;
        s.bits[out] = b;
        out += 1;
    }

    // Re-home into an exactly sized allocation rather than shrinking the slice
    // in place, so the eventual `free` sees the length it was allocated with.
    const kept = try gpa.alloc(Bit, out);
    @memcpy(kept, s.bits[0..out]);
    gpa.free(s.bits);
    s.bits = kept;
}

fn varWidth(ir: *const Ir, v: Ir.Variable.Index) u16 {
    const ty = ir.vars.items(.ty)[@intFromEnum(v)];
    return if (ty.width == 0) Ir.default_width else ty.width;
}

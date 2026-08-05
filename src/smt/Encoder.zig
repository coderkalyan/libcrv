//! Translate the IR into Bitwuzla terms.
//!
//! The IR is already a bit-vector language with explicit `zext`/`sext`/`trunc`
//! and no implicit widening, so most tags map straight onto a `BITWUZLA_KIND_*`
//! with nothing to infer. The value of this module is entirely in the handful
//! of places where the obvious mapping would be **wrong** — where SMT-LIB and
//! `RejectionSampler` disagree about what an operator means. Getting those
//! right is what lets the two engines be compared against each other at all:
//!
//!   * **Division by zero.** SMT-LIB defines `bvudiv x 0` as all-ones and
//!     `bvurem x 0` as `x`; the interpreter yields 0. Every division is
//!     therefore wrapped in an explicit zero guard rather than emitted bare.
//!
//!   * **Shift amounts.** SMT-LIB requires both shift operands to share a
//!     sort; the IR does not. Truncating the amount to the value's width would
//!     turn "shift by 256" into "shift by 0", so both operands are instead
//!     widened to the larger width, shifted there, and the result narrowed.
//!
//!   * **Booleans.** The IR calls them "width 1", but Bitwuzla keeps `Bool` and
//!     `BV(1)` as separate sorts, and a `var_ref` to a 1-bit variable is a
//!     genuine bit-vector. So the encoding sort is chosen by *tag*, not width,
//!     and coercions are inserted only where an operand's sort does not match
//!     what its parent needs.
//!
//!   * **Mixed operand widths.** The interpreter reads operands at their own
//!     widths and masks the result, and for signed comparisons reads *both*
//!     operands at the left operand's width. Each case is reproduced here
//!     rather than assumed away.
//!
//! Nodes are encoded in one forward pass, which is valid for the same reason
//! evaluation is: the flattened tree is already in dependency order.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("../Ir.zig");
const Solver = @import("../Solver.zig");
const bw = @import("bitwuzla");

const Encoder = @This();

pub const Error = error{
    /// A node tag this encoder does not model (see the module docs).
    Unsupported,
} || Allocator.Error;

ir: *const Ir,
tm: bw.TermManager,
/// One term per node, indexed by `Ir.Node.Index`.
terms: []bw.Term,
/// Whether each node's term carries Bitwuzla's `Bool` sort rather than a
/// bit-vector sort.
is_bool: []bool,
/// Resolved width per node.
widths: []u16,
/// One term per variable: a free constant, or a literal when the caller pinned
/// it. Indexed by `Ir.Variable.Index`.
vars: []bw.Term,
/// Resolved width per variable.
var_widths: []u16,

/// Scratch for building base-2 literal strings, sized to the widest term.
lit_buf: []u8,

bv1: bw.Sort,
bv1_zero: bw.Term,
bv1_one: bw.Term,

pub const Options = struct {
    /// Values to pin `.state` variables to, laid out exactly like a solution
    /// buffer (`Solver.valueLimbs(ir)` limbs per variable, indexed by
    /// `Ir.Variable.Index`; entries for non-state variables are ignored).
    ///
    /// When null, state variables are left free for the solver to choose,
    /// which is what `RejectionSampler` effectively does today by drawing them
    /// along with everything else.
    state: ?[]const Solver.Value = null,
};

pub fn deinit(e: *Encoder, gpa: Allocator) void {
    gpa.free(e.terms);
    gpa.free(e.is_bool);
    gpa.free(e.widths);
    gpa.free(e.vars);
    gpa.free(e.var_widths);
    gpa.free(e.lit_buf);
    e.* = undefined;
}

/// The free constant standing for variable `v`.
pub fn varTerm(e: *const Encoder, v: Ir.Variable.Index) bw.Term {
    return e.vars[@intFromEnum(v)];
}

pub fn varWidth(e: *const Encoder, v: Ir.Variable.Index) u16 {
    return e.var_widths[@intFromEnum(v)];
}

/// Assert every constraint statement into `session`.
pub fn assertAll(e: *const Encoder, session: bw.Session) void {
    const extra = e.ir.extra.items;
    for (e.ir.constraints.items(.body)) |body| {
        for (extra[@intFromEnum(body.start)..][0..body.len]) |stmt| {
            session.assert(e.asBool(stmt));
        }
    }
}

pub fn encode(
    gpa: Allocator,
    ir: *const Ir,
    tm: bw.TermManager,
    options: Options,
) Error!Encoder {
    const n = ir.nodes.len;

    const widths = try gpa.alloc(u16, n);
    errdefer gpa.free(widths);
    var max_width: u16 = 1;
    for (widths, 0..) |*w, i| {
        w.* = ir.typeOf(@enumFromInt(@as(u32, @intCast(i)))).width;
        max_width = @max(max_width, w.*);
    }

    const var_widths = try gpa.alloc(u16, ir.vars.len);
    errdefer gpa.free(var_widths);
    for (ir.vars.items(.ty), var_widths) |ty, *w| {
        w.* = if (ty.width == 0) Ir.default_width else ty.width;
        max_width = @max(max_width, w.*);
    }

    const terms = try gpa.alloc(bw.Term, n);
    errdefer gpa.free(terms);
    const is_bool = try gpa.alloc(bool, n);
    errdefer gpa.free(is_bool);
    @memset(is_bool, false);
    const vars = try gpa.alloc(bw.Term, ir.vars.len);
    errdefer gpa.free(vars);
    const lit_buf = try gpa.alloc(u8, @as(usize, max_width) + 1);
    errdefer gpa.free(lit_buf);

    const bv1 = tm.bvSort(1);
    var e: Encoder = .{
        .ir = ir,
        .tm = tm,
        .terms = terms,
        .is_bool = is_bool,
        .widths = widths,
        .vars = vars,
        .var_widths = var_widths,
        .lit_buf = lit_buf,
        .bv1 = bv1,
        .bv1_zero = tm.bvValueU64(bv1, 0),
        .bv1_one = tm.bvValueU64(bv1, 1),
    };

    e.encodeVars(options);
    try e.encodeNodes();
    return e;
}

fn encodeVars(e: *Encoder, options: Options) void {
    const kinds = e.ir.vars.items(.kind);
    const limbs = Solver.valueLimbs(e.ir);
    for (e.vars, 0..) |*t, v| {
        const width = e.var_widths[v];
        const sort = e.tm.bvSort(width);
        if (kinds[v] == .state) {
            if (options.state) |state| {
                t.* = e.limbsTerm(sort, state[v * limbs ..][0..limbs], width);
                continue;
            }
        }
        t.* = e.tm.constant(sort);
    }
}

fn encodeNodes(e: *Encoder) Error!void {
    const tags = e.ir.nodes.items(.tag);
    const datas = e.ir.nodes.items(.data);
    for (tags, datas, 0..) |tag, d, i| {
        try e.encodeNode(@intCast(i), tag, d);
    }
}

fn encodeNode(e: *Encoder, i: u32, tag: Ir.Node.Tag, d: Ir.Node.Data) Error!void {
    const w = e.widths[i];
    switch (tag) {
        .int_literal => e.set(i, e.literalTerm(i), false),
        .bool_literal => e.set(i, if (d.lhs != 0) e.tm.mkTrue() else e.tm.mkFalse(), true),
        .var_ref => e.set(i, e.vars[d.lhs], false),

        .neg => e.set(i, e.tm.term1(.bv_neg, e.asBv(d.lhs, w)), false),
        .bnot => e.set(i, e.tm.term1(.bv_not, e.asBv(d.lhs, w)), false),
        .lnot => e.set(i, e.tm.term1(.not, e.asBool(d.lhs)), true),

        .add => e.binaryBv(i, .bv_add, d, w),
        .sub => e.binaryBv(i, .bv_sub, d, w),
        .mul => e.binaryBv(i, .bv_mul, d, w),
        .band => e.binaryBv(i, .bv_and, d, w),
        .bor => e.binaryBv(i, .bv_or, d, w),
        .bxor => e.binaryBv(i, .bv_xor, d, w),

        // Guarded: the IR's divisions yield 0 on a zero divisor, SMT-LIB's
        // do not.
        .sdiv => e.divide(i, .bv_sdiv, d, w),
        .udiv => e.divide(i, .bv_udiv, d, w),
        .srem => e.divide(i, .bv_srem, d, w),
        .urem => e.divide(i, .bv_urem, d, w),

        .sll => e.shift(i, .bv_shl, d, w),
        .srl => e.shift(i, .bv_shr, d, w),
        .sra => e.shift(i, .bv_ashr, d, w),

        .eq => e.set(i, e.tm.term2(.equal, e.wideL(d), e.wideR(d)), true),
        .ne => e.set(i, e.tm.term2(.distinct, e.wideL(d), e.wideR(d)), true),

        // Unsigned comparisons see each operand at its own width, so widening
        // both preserves the interpreter's raw comparison exactly.
        .ult => e.set(i, e.tm.term2(.bv_ult, e.wideL(d), e.wideR(d)), true),
        .ule => e.set(i, e.tm.term2(.bv_ule, e.wideL(d), e.wideR(d)), true),
        .ugt => e.set(i, e.tm.term2(.bv_ugt, e.wideL(d), e.wideR(d)), true),
        .uge => e.set(i, e.tm.term2(.bv_uge, e.wideL(d), e.wideR(d)), true),

        // Signed comparisons read *both* operands at the left operand's width,
        // matching `asI64(v, typeOf(lhs).width)` in the interpreter.
        .slt => e.signedCompare(i, .bv_slt, d),
        .sle => e.signedCompare(i, .bv_sle, d),
        .sgt => e.signedCompare(i, .bv_sgt, d),
        .sge => e.signedCompare(i, .bv_sge, d),

        .land => e.set(i, e.tm.term2(.and_, e.asBool(d.lhs), e.asBool(d.rhs)), true),
        .lor => e.set(i, e.tm.term2(.or_, e.asBool(d.lhs), e.asBool(d.rhs)), true),
        .implies => e.set(i, e.tm.term2(.implies, e.asBool(d.lhs), e.asBool(d.rhs)), true),
        .iff => e.set(i, e.tm.term2(.iff, e.asBool(d.lhs), e.asBool(d.rhs)), true),

        .zext => e.set(i, e.resize(e.asBv(d.lhs, e.widths[d.lhs]), e.widths[d.lhs], w, .zero), false),
        .sext => e.set(i, e.resize(e.asBv(d.lhs, e.widths[d.lhs]), e.widths[d.lhs], w, .sign), false),
        .trunc => e.set(i, e.resize(e.asBv(d.lhs, e.widths[d.lhs]), e.widths[d.lhs], w, .zero), false),

        // Consumed by its `in` parent, never a term of its own.
        .range => e.set(i, e.terms[d.lhs], false),
        .in => try e.membership(i, d),
        .if_else => e.conditional(i, d),
        .unique => try e.distinct(i, d),

        // Deliberately rejected rather than quietly dropped: each of these
        // changes the *distribution* an engine should produce, so ignoring one
        // would silently hand back samples the caller did not ask for.
        .dist, .dist_weight_eq, .dist_weight_div, .solve_before, .foreach => return error.Unsupported,
    }
}

fn set(e: *Encoder, i: u32, t: bw.Term, boolean: bool) void {
    e.terms[i] = t;
    e.is_bool[i] = boolean;
}

// -- Sort coercion -----------------------------------------------------------

/// Node `i` as a `Bool`. A bit-vector is true when non-zero, matching
/// `RejectionSampler.truthy`.
fn asBool(e: *const Encoder, i: u32) bw.Term {
    if (e.is_bool[i]) return e.terms[i];
    const zero = e.zeroOf(e.widths[i]);
    return e.tm.term2(.distinct, e.terms[i], zero);
}

/// Node `i` as a bit-vector of `want` bits.
fn asBv(e: *const Encoder, i: u32, want: u16) bw.Term {
    const t = if (e.is_bool[i]) e.tm.term3(.ite, e.terms[i], e.bv1_one, e.bv1_zero) else e.terms[i];
    const have: u16 = if (e.is_bool[i]) 1 else e.widths[i];
    return e.resize(t, have, want, .zero);
}

const Fill = enum { zero, sign };

/// Widen or narrow a bit-vector term to `want` bits.
fn resize(e: *const Encoder, t: bw.Term, have: u16, want: u16, fill: Fill) bw.Term {
    if (want == have) return t;
    if (want > have) return switch (fill) {
        .zero => e.tm.zeroExtend(t, want - have),
        .sign => e.tm.signExtend(t, want - have),
    };
    return e.tm.extract(t, want - 1, 0);
}

fn zeroOf(e: *const Encoder, width: u16) bw.Term {
    return e.tm.bvValueU64(e.tm.bvSort(width), 0);
}

// -- Operator helpers --------------------------------------------------------

/// A binary bit-vector operator evaluated at the result width, with the right
/// operand brought to that width first (the interpreter masks it there).
fn binaryBv(e: *Encoder, i: u32, kind: bw.Kind, d: Ir.Node.Data, w: u16) void {
    e.set(i, e.tm.term2(kind, e.asBv(d.lhs, w), e.asBv(d.rhs, w)), false);
}

/// `ITE(divisor == 0, 0, op(a, b))` — the IR yields 0 on a zero divisor, which
/// is not what any of SMT-LIB's four division operators do.
fn divide(e: *Encoder, i: u32, kind: bw.Kind, d: Ir.Node.Data, w: u16) void {
    const a = e.asBv(d.lhs, w);
    const b = e.asBv(d.rhs, w);
    const zero = e.zeroOf(w);
    const is_zero = e.tm.term2(.equal, b, zero);
    e.set(i, e.tm.term3(.ite, is_zero, zero, e.tm.term2(kind, a, b)), false);
}

/// Shifts, done at `max(value width, amount width)` and narrowed back.
///
/// Both SMT-LIB operands must share a sort, but the IR's do not have to. The
/// interpreter clamps the amount to the value's width, so an amount that
/// overshoots shifts everything out; narrowing the amount instead would wrap it
/// back into range and produce a completely different result.
fn shift(e: *Encoder, i: u32, kind: bw.Kind, d: Ir.Node.Data, w: u16) void {
    const value_width = e.opWidth(d.lhs);
    const amount_width = e.opWidth(d.rhs);
    const wide = @max(w, @max(value_width, amount_width));

    const value = e.asBv(d.lhs, value_width);
    const a = e.resize(value, value_width, wide, if (kind == .bv_ashr) .sign else .zero);
    const b = e.asBv(d.rhs, wide);
    e.set(i, e.resize(e.tm.term2(kind, a, b), wide, w, .zero), false);
}

/// Operands of a width-agnostic comparison, both widened to the larger width.
fn wideL(e: *const Encoder, d: Ir.Node.Data) bw.Term {
    return e.asBv(d.lhs, @max(e.opWidth(d.lhs), e.opWidth(d.rhs)));
}

fn wideR(e: *const Encoder, d: Ir.Node.Data) bw.Term {
    return e.asBv(d.rhs, @max(e.opWidth(d.lhs), e.opWidth(d.rhs)));
}

fn opWidth(e: *const Encoder, i: u32) u16 {
    return if (e.is_bool[i]) 1 else e.widths[i];
}

/// A signed comparison, with both operands taken at the left operand's width.
fn signedCompare(e: *Encoder, i: u32, kind: bw.Kind, d: Ir.Node.Data) void {
    const w = e.opWidth(d.lhs);
    e.set(i, e.tm.term2(kind, e.asBv(d.lhs, w), e.asBv(d.rhs, w)), true);
}

/// `value inside { ... }` — a disjunction of equalities and unsigned range
/// tests. The interpreter compares raw values, so members are widened rather
/// than narrowed, and ranges are unsigned.
fn membership(e: *Encoder, i: u32, d: Ir.Node.Data) Error!void {
    const extra = e.ir.extra.items;
    const members = extra[d.rhs + 1 ..][0..extra[d.rhs]];
    if (members.len == 0) {
        e.set(i, e.tm.mkFalse(), true);
        return;
    }

    const tags = e.ir.nodes.items(.tag);
    const datas = e.ir.nodes.items(.data);

    var acc: ?bw.Term = null;
    for (members) |m| {
        const test_term = if (tags[m] == .range) blk: {
            const r = datas[m];
            const w = @max(e.opWidth(d.lhs), @max(e.opWidth(r.lhs), e.opWidth(r.rhs)));
            const v = e.asBv(d.lhs, w);
            break :blk e.tm.term2(
                .and_,
                e.tm.term2(.bv_ule, e.asBv(r.lhs, w), v),
                e.tm.term2(.bv_ule, v, e.asBv(r.rhs, w)),
            );
        } else blk: {
            const w = @max(e.opWidth(d.lhs), e.opWidth(m));
            break :blk e.tm.term2(.equal, e.asBv(d.lhs, w), e.asBv(m, w));
        };
        acc = if (acc) |prev| e.tm.term2(.or_, prev, test_term) else test_term;
    }
    e.set(i, acc.?, true);
}

/// `if (c) t else f` as a constraint: both arms are statements, so this is
/// `(c -> t) and (!c -> f)`, with the second conjunct dropped when there is no
/// `else`.
fn conditional(e: *Encoder, i: u32, d: Ir.Node.Data) void {
    const extra = e.ir.extra.items;
    const cond = e.asBool(d.lhs);
    const then_term = e.tm.term2(.implies, cond, e.asBool(extra[d.rhs]));

    const else_node: Ir.Node.Index = @enumFromInt(extra[d.rhs + 1]);
    if (else_node == .null) {
        e.set(i, then_term, true);
        return;
    }
    const else_term = e.tm.term2(
        .implies,
        e.tm.term1(.not, cond),
        e.asBool(extra[d.rhs + 1]),
    );
    e.set(i, e.tm.term2(.and_, then_term, else_term), true);
}

/// `unique { ... }` — pairwise distinctness, compared unsigned at the widest
/// member's width.
fn distinct(e: *Encoder, i: u32, d: Ir.Node.Data) Error!void {
    const extra = e.ir.extra.items;
    const members = extra[d.lhs + 1 ..][0..extra[d.lhs]];
    if (members.len < 2) {
        e.set(i, e.tm.mkTrue(), true);
        return;
    }

    var w: u16 = 1;
    for (members) |m| w = @max(w, e.opWidth(m));

    // A stack buffer would have to be sized for the worst case; the member
    // count is unbounded, so borrow the literal scratch only if it fits.
    var storage: [16]bw.Term = undefined;
    if (members.len <= storage.len) {
        for (members, 0..) |m, k| storage[k] = e.asBv(m, w);
        e.set(i, e.tm.termN(.distinct, storage[0..members.len]), true);
        return;
    }

    // Fall back to an explicit conjunction of pairwise disequalities, which
    // needs no allocation at the cost of a larger term.
    var acc: ?bw.Term = null;
    for (members, 0..) |a, ai| {
        for (members[ai + 1 ..]) |b| {
            const ne = e.tm.term2(.distinct, e.asBv(a, w), e.asBv(b, w));
            acc = if (acc) |prev| e.tm.term2(.and_, prev, ne) else ne;
        }
    }
    e.set(i, acc.?, true);
}

// -- Literals ----------------------------------------------------------------

/// An `int_literal` node as a bit-vector value.
fn literalTerm(e: *Encoder, i: u32) bw.Term {
    const d = e.ir.nodes.items(.data)[i];
    const width = e.widths[i];
    const extra = e.ir.extra.items;
    const words = extra[d.lhs + 1 ..][0..extra[d.lhs]];

    // Most-significant character first, exactly `width` of them.
    var bit: u16 = width;
    var pos: usize = 0;
    while (bit > 0) : (pos += 1) {
        bit -= 1;
        const word = @as(usize, bit) / 32;
        const on = word < words.len and (words[word] >> @intCast(bit % 32)) & 1 != 0;
        e.lit_buf[pos] = if (on) '1' else '0';
    }
    e.lit_buf[width] = 0;
    return e.tm.bvValueBin(e.tm.bvSort(width), e.lit_buf[0..width :0]);
}

/// A pinned variable's value, given as little-endian limbs.
fn limbsTerm(e: *Encoder, sort: bw.Sort, value: []const Solver.Value, width: u16) bw.Term {
    const limb_bits = @bitSizeOf(Solver.Value);
    var bit: u16 = width;
    var pos: usize = 0;
    while (bit > 0) : (pos += 1) {
        bit -= 1;
        const limb = @as(usize, bit) / limb_bits;
        const on = limb < value.len and (value[limb] >> @intCast(bit % limb_bits)) & 1 != 0;
        e.lit_buf[pos] = if (on) '1' else '0';
    }
    e.lit_buf[width] = 0;
    return e.tm.bvValueBin(sort, e.lit_buf[0..width :0]);
}

// -- Tests -------------------------------------------------------------------
//
// The encoder is the foundation every later phase stands on: support
// minimization, counting, and sampling all assume the term it built means what
// the IR node meant. So it is checked against the reference evaluator
// *exhaustively* rather than by spot-checking interesting inputs -- for a
// handful of narrow variables the whole assignment space is a few thousand
// points, and enumerating it catches the disagreements nobody thought to look
// for. Every case below is one where the naive mapping is wrong.

const RejectionSampler = @import("../RejectionSampler.zig");
const Type = Ir.Type;

/// Check that Bitwuzla and the interpreter agree on *every* assignment.
///
/// Each variable is pinned to a candidate value and the result compared with
/// the evaluator's verdict on the same assignment, so a disagreement is
/// reported as the exact point where the two engines diverge.
fn expectAgreement(gpa: Allocator, ir: *const Ir) !void {
    if (!bw.available) return error.SkipZigTest;

    var sampler = try RejectionSampler.init(gpa, ir, .{});
    defer sampler.deinit(gpa);

    const tm = bw.TermManager.init();
    defer tm.deinit();
    const opts = bw.Options.init();
    defer opts.deinit();

    var enc = try Encoder.encode(gpa, ir, tm, .{});
    defer enc.deinit(gpa);

    const session = bw.Session.init(tm, opts);
    defer session.deinit();
    enc.assertAll(session);

    const limbs = Solver.valueLimbs(ir);
    const nvars = ir.vars.len;
    const values = try gpa.alloc(Solver.Value, @max(limbs * nvars, 1));
    defer gpa.free(values);

    // Odometer over the variables' full ranges. Widths are kept small enough
    // by the callers that the whole product is a few thousand points.
    var total: u64 = 1;
    for (0..nvars) |v| {
        const w = enc.var_widths[v];
        std.debug.assert(w <= 8);
        total *= @as(u64, 1) << @intCast(w);
    }

    var point: u64 = 0;
    while (point < total) : (point += 1) {
        @memset(values, 0);
        var rest = point;
        for (0..nvars) |v| {
            const span = @as(u64, 1) << @intCast(enc.var_widths[v]);
            values[v * limbs] = rest % span;
            rest /= span;
        }

        const expected = sampler.check(values);

        session.push(1);
        for (0..nvars) |v| {
            const sort = tm.bvSort(enc.var_widths[v]);
            session.assert(tm.term2(
                .equal,
                enc.varTerm(@enumFromInt(@as(u32, @intCast(v)))),
                tm.bvValueU64(sort, values[v * limbs]),
            ));
        }
        const actual = session.checkSat() == .sat;
        session.pop(1);

        if (expected != actual) {
            std.debug.print(
                "encoder disagrees with the interpreter at point {d}: " ++
                    "interpreter={}, bitwuzla={}\n",
                .{ point, expected, actual },
            );
            return error.TestExpectedEqual;
        }
    }
}

fn constraintOne(gpa: Allocator, ir: *Ir, node: Ir.Node.Index) !void {
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{node});
}

test "division by zero agrees with the interpreter" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // SMT-LIB says `bvudiv x 0` is all ones and `bvurem x 0` is x; the IR says
    // both are 0. Encoding either bare would disagree on every y == 0.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
    const q = try ir.binary(gpa, .udiv, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    const r = try ir.binary(gpa, .urem, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .eq, q, r));

    try expectAgreement(gpa, &ir);
}

test "signed division and remainder agree with the interpreter" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // Covers the divide-by-zero guard and the minInt / -1 corner, where the
    // interpreter's wrapping negate has to line up with `bvsdiv`.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
    const q = try ir.binary(gpa, .sdiv, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    const r = try ir.binary(gpa, .srem, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    try constraintOne(gpa, &ir, try ir.binary(gpa, .ne, q, r));

    try expectAgreement(gpa, &ir);
}

test "shifts agree when the amount can overshoot the width" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // A 6-bit amount against a 4-bit value: amounts above 3 shift everything
    // out. Narrowing the amount to 4 bits instead would wrap 16 back to 0 and
    // leave the value untouched.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const s = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(6), .kind = .rand });
    const left = try ir.binary(gpa, .sll, try ir.varRef(gpa, x), try ir.varRef(gpa, s));
    const right = try ir.binary(gpa, .srl, try ir.varRef(gpa, x), try ir.varRef(gpa, s));
    const arith = try ir.binary(gpa, .sra, try ir.varRef(gpa, x), try ir.varRef(gpa, s));
    const both = try ir.binary(gpa, .eq, left, right);
    try constraintOne(gpa, &ir, try ir.binary(gpa, .lor, both, try ir.binary(gpa, .eq, arith, left)));

    try expectAgreement(gpa, &ir);
}

test "signed and unsigned comparisons agree with the interpreter" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(5), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(5), .kind = .rand });
    const signed = try ir.binary(gpa, .slt, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    const unsigned = try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    // True exactly where the two orderings disagree, i.e. across the sign bit.
    try constraintOne(gpa, &ir, try ir.binary(gpa, .ne, signed, unsigned));

    try expectAgreement(gpa, &ir);
}

test "casts and truncation agree with the interpreter" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const wide = try ir.sext(gpa, try ir.varRef(gpa, x), 8);
    const narrowed = try ir.trunc(gpa, wide, 4);
    const zeroed = try ir.zext(gpa, try ir.varRef(gpa, x), 8);
    // sext then trunc round-trips; sext and zext differ exactly on negatives.
    const round_trip = try ir.binary(gpa, .eq, narrowed, try ir.varRef(gpa, x));
    const differs = try ir.binary(gpa, .ne, wide, zeroed);
    try constraintOne(gpa, &ir, try ir.binary(gpa, .land, round_trip, differs));

    try expectAgreement(gpa, &ir);
}

test "set membership agrees with the interpreter" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // `in` compares unsigned, so [10:3] is empty rather than reversed.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(5), .kind = .rand });
    const membership = try ir.in(gpa, try ir.varRef(gpa, x), &.{
        try ir.range(gpa, try ir.constInt(gpa, 3, Type.bit(5)), try ir.constInt(gpa, 9, Type.bit(5))),
        try ir.constInt(gpa, 20, Type.bit(5)),
        try ir.range(gpa, try ir.constInt(gpa, 10, Type.bit(5)), try ir.constInt(gpa, 3, Type.bit(5))),
    });
    try constraintOne(gpa, &ir, membership);

    try expectAgreement(gpa, &ir);
}

test "boolean connectives over bit-vector operands agree" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // Mixes Bool-sorted results with a bit-vector read for truth, which is
    // where the Bool/BV(1) coercions get exercised in both directions.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
    const cmp = try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    const masked = try ir.binary(gpa, .band, try ir.varRef(gpa, x), try ir.constInt(gpa, 1, Type.bit(4)));
    const implication = try ir.binary(gpa, .implies, cmp, masked);
    try constraintOne(gpa, &ir, try ir.binary(gpa, .lor, implication, try ir.unary(gpa, .lnot, cmp)));

    try expectAgreement(gpa, &ir);
}

test "conditional constraints agree with the interpreter" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
    const cond = try ir.binary(gpa, .ugt, try ir.varRef(gpa, x), try ir.constInt(gpa, 7, Type.bit(4)));
    const then_stmt = try ir.binary(gpa, .eq, try ir.varRef(gpa, y), try ir.constInt(gpa, 0, Type.bit(4)));
    const else_stmt = try ir.binary(gpa, .ugt, try ir.varRef(gpa, y), try ir.constInt(gpa, 0, Type.bit(4)));
    const payload = try ir.addExtra(gpa, &.{ @intFromEnum(then_stmt), @intFromEnum(else_stmt) });
    const node = try ir.addNode(gpa, .{
        .tag = .if_else,
        .data = .{ .lhs = @intFromEnum(cond), .rhs = @intFromEnum(payload) },
    });
    try constraintOne(gpa, &ir, node);

    // The interpreter does not model `if_else`, so only the encoder side is
    // exercised here: the constraint must be satisfiable and must force y to
    // zero exactly when x exceeds 7.
    if (!bw.available) return error.SkipZigTest;

    const tm = bw.TermManager.init();
    defer tm.deinit();
    const opts = bw.Options.init();
    defer opts.deinit();
    var enc = try Encoder.encode(gpa, &ir, tm, .{});
    defer enc.deinit(gpa);
    const session = bw.Session.init(tm, opts);
    defer session.deinit();
    enc.assertAll(session);

    const sort = tm.bvSort(4);
    for (0..16) |xv| {
        for (0..16) |yv| {
            session.push(1);
            session.assert(tm.term2(.equal, enc.varTerm(@enumFromInt(0)), tm.bvValueU64(sort, xv)));
            session.assert(tm.term2(.equal, enc.varTerm(@enumFromInt(1)), tm.bvValueU64(sort, yv)));
            const sat = session.checkSat() == .sat;
            session.pop(1);

            const want = if (xv > 7) yv == 0 else yv > 0;
            try std.testing.expectEqual(want, sat);
        }
    }
}

test "unsupported nodes are rejected rather than ignored" {
    const gpa = std.testing.allocator;
    if (!bw.available) return error.SkipZigTest;

    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const payload = try ir.addExtra(gpa, &.{ 1, 0, 1, 1 });
    const node = try ir.addNode(gpa, .{
        .tag = .solve_before,
        .data = .{ .lhs = @intFromEnum(payload) },
    });
    _ = x;
    try constraintOne(gpa, &ir, node);

    const tm = bw.TermManager.init();
    defer tm.deinit();
    try std.testing.expectError(error.Unsupported, Encoder.encode(gpa, &ir, tm, .{}));
}

test {
    std.testing.refAllDecls(@This());
}

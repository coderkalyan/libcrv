//! Bit-blasting — compiling IR expressions into BDDs.
//!
//! Every IR node of width `w` becomes a `[w]Ref`: a little-endian vector of
//! Boolean functions, least-significant bit first. A boolean is just a
//! width-1 vector, so there is no second expression kind to carry around.
//!
//! The sweep is *linear*. `Ir` nodes only ever refer to lower indices, so the
//! array is already in topological order and one forward pass in index order
//! suffices — the same structure `RejectionSampler.evaluate` relies on. No
//! worklist, no recursion, and structural sharing in the IR is inherited for
//! free: a subexpression referenced twice is compiled once.
//!
//! The operands of a value-producing operator share one width: `typeOf` gives
//! such a node a single type, and widths change only through the explicit cast
//! nodes. Mismatched operands are malformed IR, so they trip an assert rather
//! than being quietly reinterpreted — which also means operand vectors can be
//! read in place instead of copied through a width-fitting temporary. The one
//! exception is a shift amount, which is a count rather than a value of the
//! operand's type and may be any width.
//!
//! **What is expensive.** Bit-blasting is where a BDD engine gets to be fast or
//! gets to explode, and the difference is not subtle:
//!
//!   * Casts and constant shifts are pure *wiring* — they create no nodes.
//!   * Bitwise ops, comparisons, ranges, and ripple-carry addition are linear
//!     under the interleaved order (see `order.zig`).
//!   * Multiplication and division by a power-of-two constant are wiring too,
//!     which matters more than it sounds: alignment constraints like
//!     `(addr & 63) == 0` are everywhere in real stimulus.
//!   * Multiplication and division of one *variable* by another are
//!     provably exponential — Bryant showed in 1991 that `x * y == z` has no
//!     good variable order, for any order. That is a theorem, not a heuristic
//!     failure, so those are refused above `max_mul_width` rather than
//!     optimistically attempted.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("../Ir.zig");
const Manager = @import("Manager.zig");
const Order = @import("order.zig");

const Ref = Manager.Ref;

pub const Error = error{
    /// A variable-by-variable multiply or divide wider than `max_mul_width`.
    OperandTooWide,
    /// A node this engine does not model (`dist`, `solve_before`, `foreach`).
    UnsupportedNode,
} || Manager.Error;

pub const Options = struct {
    /// Widest variable-by-variable multiply/divide to attempt.
    max_mul_width: u16 = 12,
};

/// Compile `stmts` into a single BDD: the conjunction of every statement.
///
/// `slot_of_var` maps an `Ir.Variable.Index` to its slot in `ord`, or `no_slot`
/// for a variable outside this component (which cannot occur in a well-formed
/// component and is asserted against).
pub fn blast(
    gpa: Allocator,
    m: *Manager,
    ir: *const Ir,
    types: []const Ir.Type,
    ord: *const Order,
    slot_of_var: []const u32,
    stmts: []const Ir.Node.Index,
    options: Options,
) Error!Ref {
    var b: Blaster = .{
        .gpa = gpa,
        .m = m,
        .ir = ir,
        .types = types,
        .ord = ord,
        .slot_of_var = slot_of_var,
        .options = options,
        .scratch = .init(gpa),
    };
    defer b.deinit();
    return b.run(stmts);
}

/// Sentinel in `slot_of_var` for a variable this component does not own.
pub const no_slot: u32 = std.math.maxInt(u32);

const Blaster = struct {
    gpa: Allocator,
    m: *Manager,
    ir: *const Ir,
    types: []const Ir.Type,
    ord: *const Order,
    slot_of_var: []const u32,
    options: Options,

    /// One `[]Ref` per live node, carved out of `vec_pool` by `vec_at`.
    vec_pool: []Ref = &.{},
    vec_at: []u32 = &.{},
    live: []bool = &.{},

    /// Per-node temporaries. Reset after each node, so peak use is one node's
    /// worth rather than the whole sweep's.
    scratch: std.heap.ArenaAllocator,

    fn deinit(b: *Blaster) void {
        b.gpa.free(b.vec_pool);
        b.gpa.free(b.vec_at);
        b.gpa.free(b.live);
        b.scratch.deinit();
    }

    fn run(b: *Blaster, stmts: []const Ir.Node.Index) Error!Ref {
        const n = b.ir.nodes.len;

        // Which nodes this component actually needs. Children have lower
        // indices, so one reverse sweep propagates liveness.
        b.live = try b.gpa.alloc(bool, n);
        @memset(b.live, false);
        for (stmts) |s| b.live[@intFromEnum(s)] = true;
        var i = n;
        while (i > 0) {
            i -= 1;
            if (b.live[i]) b.ir.forEachChild(@enumFromInt(@as(u32, @intCast(i))), b.live, markLive);
        }

        // Carve one bit-vector per live node out of a single allocation.
        b.vec_at = try b.gpa.alloc(u32, n);
        var total: u64 = 0;
        for (0..n) |k| {
            b.vec_at[k] = @intCast(total);
            if (b.live[k]) total += b.types[k].width;
        }
        b.vec_pool = try b.gpa.alloc(Ref, @intCast(total));

        for (0..n) |k| {
            if (!b.live[k]) continue;
            try b.node(@intCast(k));
            _ = b.scratch.reset(.retain_capacity);
        }

        var acc: Ref = .one;
        for (stmts) |s| acc = try b.m.conj(acc, try b.truthy(@intFromEnum(s)));
        return acc;
    }

    fn markLive(live: []bool, child: Ir.Child) void {
        switch (child) {
            .node => |index| live[@intFromEnum(index)] = true,
            .variable => {},
        }
    }

    /// Node `i`'s bit-vector.
    fn vec(b: *const Blaster, i: u32) []Ref {
        return b.vec_pool[b.vec_at[i]..][0..b.width(i)];
    }

    fn width(b: *const Blaster, i: u32) u16 {
        return b.types[i].width;
    }

    fn tmp(b: *Blaster, w: usize) Allocator.Error![]Ref {
        return b.scratch.allocator().alloc(Ref, w);
    }

    // -- Per-node dispatch ---------------------------------------------------

    fn node(b: *Blaster, i: u32) Error!void {
        const tag = b.ir.nodes.items(.tag)[i];
        const d = b.ir.nodes.items(.data)[i];
        const w = b.width(i);
        const out = b.vec(i);

        switch (tag) {
            .int_literal => b.literal(out, d),
            .bool_literal => out[0] = if (d.lhs != 0) .one else .zero,
            .var_ref => try b.varRef(out, d.lhs),

            .zext => fitU(out, b.vec(d.lhs)),
            .sext => fitS(out, b.vec(d.lhs)),
            .trunc => fitU(out, b.vec(d.lhs)),

            .bnot => for (out, b.vec(d.lhs)) |*o, x| {
                o.* = x.not();
            },
            .neg => {
                const zeros = try b.constVec(w, 0);
                _ = try b.addInto(out, zeros, try b.notVec(b.vec(d.lhs)), .one);
            },
            .add => {
                b.assertOperands(d);
                _ = try b.addInto(out, b.vec(d.lhs), b.vec(d.rhs), .zero);
            },
            .sub => {
                b.assertOperands(d);
                _ = try b.addInto(out, b.vec(d.lhs), try b.notVec(b.vec(d.rhs)), .one);
            },

            .band, .bor, .bxor => {
                b.assertOperands(d);
                for (out, b.vec(d.lhs), b.vec(d.rhs)) |*o, x, y| o.* = switch (tag) {
                    .band => try b.m.conj(x, y),
                    .bor => try b.m.disj(x, y),
                    else => try b.m.xor(x, y),
                };
            },

            .mul => try b.mul(out, d, w),
            .udiv => try b.divide(out, d, w, .unsigned, .quotient),
            .umod => try b.divide(out, d, w, .unsigned, .remainder),
            .sdiv => try b.divide(out, d, w, .signed, .quotient),
            .smod => try b.divide(out, d, w, .signed, .remainder),

            .sll, .srl, .sra => try b.shift(out, tag, d, w),

            .eq, .ne, .slt, .ult, .sle, .ule, .sgt, .ugt, .sge, .uge => out[0] = try b.compare(tag, d),
            .lnot => out[0] = (try b.truthy(d.lhs)).not(),
            .land => out[0] = try b.m.conj(try b.truthy(d.lhs), try b.truthy(d.rhs)),
            .lor => out[0] = try b.m.disj(try b.truthy(d.lhs), try b.truthy(d.rhs)),
            .implies => out[0] = try b.m.implies(try b.truthy(d.lhs), try b.truthy(d.rhs)),
            .iff => out[0] = try b.m.xnor(try b.truthy(d.lhs), try b.truthy(d.rhs)),

            // A `range` is only meaningful as an `in` member; standing alone it
            // is false, matching the sampler's zero value.
            .range => @memset(out, .zero),
            .in => out[0] = try b.membership(d),
            .if_else => out[0] = try b.ifElse(d),
            .unique => out[0] = try b.unique(d),

            .dist, .dist_weight_eq, .dist_weight_div, .solve_before, .foreach => return error.UnsupportedNode,
        }
    }

    // -- Leaves --------------------------------------------------------------

    fn literal(b: *Blaster, out: []Ref, d: Ir.Node.Data) void {
        const e = b.ir.extra.items;
        const words = e[d.lhs + 1 ..][0..e[d.lhs]];
        for (out, 0..) |*o, bit| {
            const word = bit / 32;
            const set = word < words.len and (words[word] >> @intCast(bit % 32)) & 1 == 1;
            o.* = if (set) .one else .zero;
        }
    }

    fn varRef(b: *Blaster, out: []Ref, v: u32) Error!void {
        const slot = b.slot_of_var[v];
        std.debug.assert(slot != no_slot);
        for (out, 0..) |*o, bit| o.* = try b.m.variable(b.ord.level(slot, @intCast(bit)));
    }

    // -- Width fitting -------------------------------------------------------

    /// A value-producing operator gives its operands one width; anything else
    /// is malformed IR. Compiled out in release builds, and what licenses the
    /// callers to read operand vectors in place.
    fn assertOperands(b: *const Blaster, d: Ir.Node.Data) void {
        std.debug.assert(b.width(d.lhs) == b.width(d.rhs));
    }

    const Mode = enum { unsigned, signed };

    fn constVec(b: *Blaster, w: u16, value: u64) Allocator.Error![]Ref {
        const out = try b.tmp(w);
        for (out, 0..) |*o, bit| o.* = if (bit < 64 and (value >> @intCast(bit)) & 1 == 1) .one else .zero;
        return out;
    }

    fn notVec(b: *Blaster, a: []const Ref) Allocator.Error![]Ref {
        const out = try b.tmp(a.len);
        for (out, a) |*o, x| o.* = x.not();
        return out;
    }

    // -- Arithmetic ----------------------------------------------------------

    /// Ripple-carry `out = a + c + carry_in`, returning the carry out. Linear
    /// in the interleaved order — one bit of carry is all the diagram must
    /// remember between levels.
    fn addInto(b: *Blaster, out: []Ref, a: []const Ref, c: []const Ref, carry_in: Ref) Error!Ref {
        var carry = carry_in;
        for (out, a, c) |*o, x, y| {
            const axb = try b.m.xor(x, y);
            o.* = try b.m.xor(axb, carry);
            // Majority: if a and b differ the carry passes through, else it is a.
            carry = try b.m.ite(axb, carry, x);
        }
        return carry;
    }

    fn mul(b: *Blaster, out: []Ref, d: Ir.Node.Data, w: u16) Error!void {
        b.assertOperands(d);
        const a = b.vec(d.lhs);
        const c = b.vec(d.rhs);

        // `x * 2^k` is a shift, and a constant operand needs no AND gate.
        if (constantShift(c)) |k| return b.shiftConst(out, a, k, .zero);
        if (constantShift(a)) |k| return b.shiftConst(out, c, k, .zero);
        if (!isConstant(a) and !isConstant(c) and w > b.options.max_mul_width) {
            return error.OperandTooWide;
        }

        @memset(out, .zero);
        const acc = try b.tmp(w);
        @memcpy(acc, out);
        for (0..w) |k| {
            if (c[k] == .zero) continue;
            // Partial product: a shifted left by k, gated on bit k of c.
            const partial = try b.tmp(w);
            for (partial, 0..) |*p, bit| {
                if (bit < k) {
                    p.* = .zero;
                } else {
                    p.* = if (c[k] == .one) a[bit - k] else try b.m.conj(a[bit - k], c[k]);
                }
            }
            const sum = try b.tmp(w);
            _ = try b.addInto(sum, acc, partial, .zero);
            @memcpy(acc, sum);
        }
        @memcpy(out, acc);
    }

    const DivPart = enum { quotient, remainder };

    fn divide(b: *Blaster, out: []Ref, d: Ir.Node.Data, w: u16, mode: Mode, part: DivPart) Error!void {
        b.assertOperands(d);
        const a_raw = b.vec(d.lhs);
        const c_raw = b.vec(d.rhs);

        // Unsigned division by a power of two is wiring: the quotient is a
        // shift and the remainder is a mask. This is the alignment-constraint
        // fast path, and it is worth a lot.
        if (mode == .unsigned) {
            if (constantShift(c_raw)) |k| {
                switch (part) {
                    .quotient => {
                        for (out, 0..) |*o, bit| o.* = if (bit + k < w) a_raw[bit + k] else .zero;
                    },
                    .remainder => {
                        for (out, 0..) |*o, bit| o.* = if (bit < k) a_raw[bit] else .zero;
                    },
                }
                return;
            }
        }

        if (!isConstant(a_raw) and !isConstant(c_raw) and w > b.options.max_mul_width) {
            return error.OperandTooWide;
        }

        // Signed division is truncating, so operate on magnitudes and re-apply
        // the signs afterwards.
        const a_neg = if (mode == .signed) a_raw[w - 1] else Ref.zero;
        const c_neg = if (mode == .signed) c_raw[w - 1] else Ref.zero;
        const a = if (mode == .signed) try b.condNegate(a_raw, a_neg) else a_raw;
        const c = if (mode == .signed) try b.condNegate(c_raw, c_neg) else c_raw;

        // Restoring division, most-significant bit first. `rem` needs one extra
        // bit so that `rem << 1 | a_i` cannot overflow before the compare.
        const ext = w + 1;
        const rem = try b.tmp(ext);
        @memset(rem, .zero);
        const quo = try b.tmp(w);
        @memset(quo, .zero);

        const c_ext = try b.tmp(ext);
        fitU(c_ext, c);

        var bit: usize = w;
        while (bit > 0) {
            bit -= 1;
            // rem = (rem << 1) | a[bit]
            const shifted = try b.tmp(ext);
            shifted[0] = a[bit];
            for (1..ext) |k| shifted[k] = rem[k - 1];

            // ge = shifted >= c_ext, via the borrow out of shifted - c_ext.
            const diff = try b.tmp(ext);
            const borrow = try b.addInto(diff, shifted, try b.notVec(c_ext), .one);
            quo[bit] = borrow; // carry out of a subtract means "no borrow"

            for (rem, shifted, diff) |*r, s, f| r.* = try b.m.ite(borrow, f, s);
        }

        const magnitude = try b.tmp(w);
        switch (part) {
            .quotient => @memcpy(magnitude, quo),
            .remainder => fitU(magnitude, rem),
        }

        const signed_result = if (mode == .signed) blk: {
            // Truncating division: the quotient's sign is the operands' signs
            // XORed, the remainder takes the dividend's sign.
            const want_neg = if (part == .quotient) try b.m.xor(a_neg, c_neg) else a_neg;
            break :blk try b.condNegate(magnitude, want_neg);
        } else magnitude;

        // Division by zero yields zero, matching `RejectionSampler`.
        var nonzero: Ref = .zero;
        for (c_raw) |x| nonzero = try b.m.disj(nonzero, x);
        for (out, signed_result) |*o, x| o.* = try b.m.conj(nonzero, x);
    }

    /// `cond ? -a : a`, two's complement.
    fn condNegate(b: *Blaster, a: []const Ref, cond: Ref) Error![]Ref {
        if (cond == .zero) {
            const copy = try b.tmp(a.len);
            @memcpy(copy, a);
            return copy;
        }
        // -a == ~a + 1, so conditionally complement and add the condition in.
        const flipped = try b.tmp(a.len);
        for (flipped, a) |*f, x| f.* = try b.m.xor(x, cond);
        const out = try b.tmp(a.len);
        const zeros = try b.tmp(a.len);
        @memset(zeros, .zero);
        _ = try b.addInto(out, flipped, zeros, cond);
        return out;
    }

    // -- Shifts --------------------------------------------------------------

    fn shiftConst(b: *Blaster, out: []Ref, a: []const Ref, k: usize, fill: Ref) void {
        _ = b;
        for (out, 0..) |*o, bit| o.* = if (bit >= k) a[bit - k] else fill;
    }

    fn shift(b: *Blaster, out: []Ref, tag: Ir.Node.Tag, d: Ir.Node.Data, w: u16) Error!void {
        // The shifted value carries the result width by construction; the
        // amount is a count, not a value of that type, so it may be any width.
        std.debug.assert(b.width(d.lhs) == w);
        const a = b.vec(d.lhs);
        const amount = b.vec(d.rhs);
        // For `sra` the sign comes from the result width, matching the sampler.
        const fill: Ref = if (tag == .sra) a[w - 1] else .zero;

        var cur = try b.tmp(w);
        @memcpy(cur, a);

        // Barrel shifter: one mux stage per bit of the shift amount that can
        // matter. Stages beyond `w` would shift everything out, and are folded
        // into `too_far` below.
        const stages = std.math.log2_int_ceil(u32, @max(2, w));
        for (0..@min(stages, amount.len)) |k| {
            if (amount[k] == .zero) continue;
            const step = @as(usize, 1) << @intCast(k);
            const next = try b.tmp(w);
            for (next, 0..) |*o, bit| {
                const from: Ref = switch (tag) {
                    .sll => if (bit >= step) cur[bit - step] else fill,
                    else => if (bit + step < w) cur[bit + step] else fill,
                };
                o.* = try b.m.ite(amount[k], from, cur[bit]);
            }
            cur = next;
        }

        // A shift of `w` or more saturates: zero, or all-sign for `sra`. The
        // comparison is done at a width that can actually represent `w`, so a
        // narrow shift operand does not wrap the bound around to zero.
        const need = std.math.log2_int_ceil(u32, @as(u32, w) + 1);
        const cmp_width: u16 = @intCast(@max(amount.len, need));
        const amt = try b.tmp(cmp_width);
        fitU(amt, amount);
        const bound = try b.constVec(cmp_width, w);
        const too_far = try b.ugeChain(amt, bound);

        for (out, cur) |*o, x| o.* = try b.m.ite(too_far, fill, x);
    }

    // -- Comparisons ---------------------------------------------------------

    fn compare(b: *Blaster, tag: Ir.Node.Tag, d: Ir.Node.Data) Error!Ref {
        b.assertOperands(d);
        const a = b.vec(d.lhs);
        const c = b.vec(d.rhs);

        return switch (tag) {
            .eq => try b.eqChain(a, c),
            .ne => (try b.eqChain(a, c)).not(),
            .ult => try b.ultChain(a, c),
            .ule => (try b.ultChain(c, a)).not(),
            .ugt => try b.ultChain(c, a),
            .uge => (try b.ultChain(a, c)).not(),
            // Signed compare is unsigned compare with the sign bits flipped,
            // which maps the two's-complement order onto the unsigned one.
            .slt => try b.ultChain(try b.flipSign(a), try b.flipSign(c)),
            .sle => (try b.ultChain(try b.flipSign(c), try b.flipSign(a))).not(),
            .sgt => try b.ultChain(try b.flipSign(c), try b.flipSign(a)),
            .sge => (try b.ultChain(try b.flipSign(a), try b.flipSign(c))).not(),
            else => unreachable,
        };
    }

    fn flipSign(b: *Blaster, a: []const Ref) Allocator.Error![]Ref {
        const out = try b.tmp(a.len);
        @memcpy(out, a);
        out[a.len - 1] = out[a.len - 1].not();
        return out;
    }

    fn eqChain(b: *Blaster, a: []const Ref, c: []const Ref) Error!Ref {
        var acc: Ref = .one;
        for (a, c) |x, y| acc = try b.m.conj(acc, try b.m.xnor(x, y));
        return acc;
    }

    /// Unsigned `a < c`, accumulated from the least significant bit up: a
    /// higher bit decides outright, and only a tie defers to what is below.
    fn ultChain(b: *Blaster, a: []const Ref, c: []const Ref) Error!Ref {
        var lt: Ref = .zero;
        for (a, c) |x, y| {
            const decide = try b.m.conj(x.not(), y);
            lt = try b.m.ite(try b.m.xnor(x, y), lt, decide);
        }
        return lt;
    }

    fn ugeChain(b: *Blaster, a: []const Ref, c: []const Ref) Error!Ref {
        return (try b.ultChain(a, c)).not();
    }

    // -- Sets, structure -----------------------------------------------------

    /// True when node `i` is non-zero — the sampler's notion of truthiness,
    /// which for a wider-than-one value is an OR reduction.
    fn truthy(b: *Blaster, i: u32) Error!Ref {
        const v = b.vec(i);
        if (v.len == 1) return v[0];
        var acc: Ref = .zero;
        for (v) |x| acc = try b.m.disj(acc, x);
        return acc;
    }

    fn membership(b: *Blaster, d: Ir.Node.Data) Error!Ref {
        const e = b.ir.extra.items;
        const members = e[d.rhs + 1 ..][0..e[d.rhs]];
        const w = b.width(d.lhs);
        const value = b.vec(d.lhs);

        var acc: Ref = .zero;
        for (members) |member| {
            const hit = if (b.ir.nodes.items(.tag)[member] == .range) blk: {
                const rd = b.ir.nodes.items(.data)[member];
                // A range's bounds are compared against the value, so all three
                // must agree in width.
                std.debug.assert(b.width(rd.lhs) == w and b.width(rd.rhs) == w);
                break :blk try b.m.conj(
                    try b.ugeChain(value, b.vec(rd.lhs)),
                    try b.ugeChain(b.vec(rd.rhs), value),
                );
            } else blk: {
                std.debug.assert(b.width(member) == w);
                break :blk try b.eqChain(value, b.vec(member));
            };
            acc = try b.m.disj(acc, hit);
        }
        return acc;
    }

    fn ifElse(b: *Blaster, d: Ir.Node.Data) Error!Ref {
        const e = b.ir.extra.items;
        const cond = try b.truthy(d.lhs);
        const then_node: Ir.Node.Index = @enumFromInt(e[d.rhs]);
        const else_node: Ir.Node.Index = @enumFromInt(e[d.rhs + 1]);

        var acc = try b.m.implies(cond, try b.truthy(@intFromEnum(then_node)));
        if (else_node != .null) {
            acc = try b.m.conj(acc, try b.m.implies(cond.not(), try b.truthy(@intFromEnum(else_node))));
        }
        return acc;
    }

    fn unique(b: *Blaster, d: Ir.Node.Data) Error!Ref {
        const e = b.ir.extra.items;
        const items = e[d.lhs + 1 ..][0..e[d.lhs]];

        var acc: Ref = .one;
        for (items, 0..) |x, k| {
            for (items[k + 1 ..]) |y| {
                // Values compared for distinctness must be the same width.
                std.debug.assert(b.width(x) == b.width(y));
                acc = try b.m.conj(acc, (try b.eqChain(b.vec(x), b.vec(y))).not());
            }
        }
        return acc;
    }
};

// -- Vector helpers ----------------------------------------------------------

/// Copy `src` into `out`, zero-extending or truncating to `out`'s width.
fn fitU(out: []Ref, src: []const Ref) void {
    for (out, 0..) |*o, bit| o.* = if (bit < src.len) src[bit] else .zero;
}

/// Copy `src` into `out`, sign-extending or truncating.
fn fitS(out: []Ref, src: []const Ref) void {
    const sign = if (src.len == 0) Ref.zero else src[src.len - 1];
    for (out, 0..) |*o, bit| o.* = if (bit < src.len) src[bit] else sign;
}

/// Every bit is a constant.
fn isConstant(v: []const Ref) bool {
    for (v) |x| if (!x.isTerminal()) return false;
    return true;
}

/// If `v` is the constant `2^k`, that `k`. Used to turn multiply and divide by
/// a power of two into pure wiring.
fn constantShift(v: []const Ref) ?usize {
    var found: ?usize = null;
    for (v, 0..) |x, bit| {
        if (!x.isTerminal()) return null;
        if (x == .one) {
            if (found != null) return null;
            found = bit;
        }
    }
    return found;
}

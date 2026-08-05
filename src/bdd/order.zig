//! Variable ordering — mapping a component's variable *bits* onto BDD levels.
//!
//! This is the single biggest determinant of whether a BDD engine works. The
//! same Boolean function can need a linear number of nodes under one order and
//! an exponential number under another, so everything else in the engine is
//! downstream of the choice made here.
//!
//! Unlike partitioning (which any engine wants, and which therefore lives in
//! `Partition`), this is BDD-specific: it only means anything to a
//! representation with a fixed total order over variables.
//!
//! The order is **significance-major, variable-minor, most-significant first**.
//! For variables of widths `w₀..wₖ` with `W = max wᵢ`:
//!
//!     level 0..     bit W-1 of every variable wide enough to have one
//!     then          bit W-2 of every such variable
//!     ...
//!     then          bit 0 of every variable
//!
//! Three properties, each load-bearing:
//!
//!   * **Most-significant first** makes comparisons and `x inside {[lo:hi]}`
//!     linear chains — the decision is usually made in the top few bits and the
//!     rest of the diagram collapses. Least-significant-first turns the same
//!     constraints quadratic or worse, and range membership is *the* dominant
//!     pattern in constrained random verification.
//!
//!   * **Interleaving** makes relational and additive constraints linear.
//!     `x + y == z` with all of `x`'s bits above all of `y`'s is exponential:
//!     the diagram must remember every bit of `x` before it sees any of `y`.
//!     Interleaved, it need only carry one bit of state — the carry. This rule
//!     is not an optimization; without it the engine does not work.
//!
//!   * **Aligning by significance rather than by bit index** makes mixed widths
//!     behave. Pairing bit 7 of an 8-bit variable with bit 7 of a 32-bit one
//!     would misalign every carry and comparison between them; pairing equal
//!     *significance* keeps arithmetic between different widths linear.
//!
//! One exception rides above all of that: a variable used *only* as a shift
//! amount is placed before every value bit. An amount is a selector, not a
//! datum — deciding it first collapses each branch to a fixed wiring of the
//! operand, whereas leaving it below forces the diagram to remember the entire
//! operand before it learns how far to shift. Significance alignment would
//! otherwise bury it: an amount is `ceil(log2(w))` bits against a `w`-bit
//! operand, so its bits are all low-significance and land at the very bottom.
//! The rule is deliberately narrow — a variable used anywhere else keeps its
//! normal place, so it cannot pull a variable out of an interleaving that some
//! other constraint depends on.
//!
//! The order is static — chosen once, before anything is built. Dynamic
//! reordering (sifting) is the standard escape hatch when a static heuristic
//! misses, and the interface here does not preclude adding it later.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("../Ir.zig");

const Order = @This();

/// Which variable bit a level decides.
pub const Bit = struct {
    /// Position within the component's variable list, not an `Ir.Variable.Index`.
    slot: u32,
    /// Bit position within that variable, 0 = least significant.
    bit: u16,
};

/// Total levels — the sum of the component's variable widths.
levels: u32,
/// Level -> the variable bit it decides. Length `levels`.
bits: []Bit,
/// Start of each slot's run in `level_of`. Length `slots + 1`.
slot_start: []u32,
/// `(slot, bit)` -> level, flattened by `slot_start`. Length `levels`.
level_of: []u32,

pub const Error = error{
    /// The component needs more levels than `max_levels` allows. Also what
    /// keeps `Manager.ite`'s recursion depth bounded.
    TooManyBits,
} || Allocator.Error;

/// Classify every variable in `ir` as "referenced only as a shift amount".
///
/// One reverse sweep, the same shape as the reachability marking elsewhere:
/// each node hands its context down to its children, except that a shift routes
/// its right operand into the amount context no matter what context it is in
/// itself. A variable then qualifies when every live reference to it arrived
/// through that route.
///
/// Computed once per IR and shared across components, since it depends only on
/// how a variable is used, not on which component it lands in. Caller owns the
/// returned slice, indexed by `Ir.Variable.Index`.
pub fn shiftAmountOnly(gpa: Allocator, ir: *const Ir) Allocator.Error![]bool {
    const n = ir.nodes.len;

    const as_amount = try gpa.alloc(bool, n);
    defer gpa.free(as_amount);
    const as_value = try gpa.alloc(bool, n);
    defer gpa.free(as_value);
    @memset(as_amount, false);
    @memset(as_value, false);

    for (ir.constraints.items(.body)) |body| {
        for (ir.extra.items[@intFromEnum(body.start)..][0..body.len]) |raw| as_value[raw] = true;
    }

    const tags = ir.nodes.items(.tag);
    const datas = ir.nodes.items(.data);

    var i = n;
    while (i > 0) {
        i -= 1;
        if (!as_amount[i] and !as_value[i]) continue;
        var ctx: Propagate = .{
            .as_amount = as_amount,
            .as_value = as_value,
            .from_amount = as_amount[i],
            .from_value = as_value[i],
        };
        switch (tags[i]) {
            .sll, .srl, .sra => {
                const d = datas[i];
                ctx.visit(.{ .node = @enumFromInt(d.lhs) });
                as_amount[d.rhs] = true;
            },
            else => ir.forEachChild(@enumFromInt(@as(u32, @intCast(i))), &ctx, Propagate.visit),
        }
    }

    const only = try gpa.alloc(bool, ir.vars.len);
    errdefer gpa.free(only);
    @memset(only, false);

    const used_as_value = try gpa.alloc(bool, ir.vars.len);
    defer gpa.free(used_as_value);
    @memset(used_as_value, false);

    for (tags, datas, 0..) |tag, d, k| {
        if (tag != .var_ref) continue;
        if (as_amount[k]) only[d.lhs] = true;
        if (as_value[k]) used_as_value[d.lhs] = true;
    }
    for (only, used_as_value) |*flag, mixed| flag.* = flag.* and !mixed;
    return only;
}

/// Hands one node's context down to its operand nodes.
const Propagate = struct {
    as_amount: []bool,
    as_value: []bool,
    from_amount: bool,
    from_value: bool,

    fn visit(p: *Propagate, child: Ir.Child) void {
        switch (child) {
            .node => |index| {
                const j = @intFromEnum(index);
                p.as_amount[j] = p.as_amount[j] or p.from_amount;
                p.as_value[j] = p.as_value[j] or p.from_value;
            },
            .variable => {},
        }
    }
};

/// Assign levels to the bits of `vars`, in the order they are given.
///
/// `amount_only` comes from `shiftAmountOnly` and is indexed by
/// `Ir.Variable.Index`; an empty slice means no variable qualifies.
pub fn init(
    gpa: Allocator,
    ir: *const Ir,
    vars: []const Ir.Variable.Index,
    amount_only: []const bool,
    max_levels: u32,
) Error!Order {
    var total: u64 = 0;
    for (vars) |v| total += widthOf(ir, v);
    if (total > max_levels) return error.TooManyBits;

    var self: Order = .{
        .levels = @intCast(total),
        .bits = &.{},
        .slot_start = &.{},
        .level_of = &.{},
    };
    errdefer self.deinit(gpa);

    self.bits = try gpa.alloc(Bit, self.levels);
    self.level_of = try gpa.alloc(u32, self.levels);
    self.slot_start = try gpa.alloc(u32, vars.len + 1);

    self.slot_start[0] = 0;
    for (vars, 0..) |v, slot| self.slot_start[slot + 1] = self.slot_start[slot] + widthOf(ir, v);

    // Shift-amount-only variables form a leading group; everything else
    // follows. Within each group the significance interleaving is identical, so
    // a constraint set with no shifts gets exactly the order it would have had.
    var next: u32 = 0;
    for ([_]bool{ true, false }) |amounts_pass| {
        var widest: u16 = 0;
        for (vars) |v| {
            if (isAmountOnly(amount_only, v) != amounts_pass) continue;
            widest = @max(widest, widthOf(ir, v));
        }

        var significance: u16 = widest;
        while (significance > 0) {
            significance -= 1;
            for (vars, 0..) |v, slot| {
                if (isAmountOnly(amount_only, v) != amounts_pass) continue;
                // A bit's significance *is* its position: bit `k` carries
                // weight 2^k whatever the variable's width. Aligning on the
                // position (and not on distance from the top) is what keeps
                // arithmetic between differently sized variables linear.
                if (widthOf(ir, v) <= significance) continue;
                self.bits[next] = .{ .slot = @intCast(slot), .bit = significance };
                self.level_of[self.slot_start[slot] + significance] = next;
                next += 1;
            }
        }
    }
    std.debug.assert(next == self.levels);

    return self;
}

pub fn deinit(self: *Order, gpa: Allocator) void {
    gpa.free(self.bits);
    gpa.free(self.slot_start);
    gpa.free(self.level_of);
    self.* = undefined;
}

/// The level deciding bit `bit` of the variable in `slot`.
pub fn level(self: *const Order, slot: u32, bit: u16) u32 {
    return self.level_of[self.slot_start[slot] + bit];
}

fn isAmountOnly(amount_only: []const bool, v: Ir.Variable.Index) bool {
    const i = @intFromEnum(v);
    return i < amount_only.len and amount_only[i];
}

/// A variable's width, resolving the "unspecified" encoding.
pub fn widthOf(ir: *const Ir, v: Ir.Variable.Index) u16 {
    const ty = ir.vars.items(.ty)[@intFromEnum(v)];
    return if (ty.width == 0) Ir.default_width else ty.width;
}

// -- Tests -------------------------------------------------------------------

const Type = Ir.Type;

test "equal widths interleave, most significant first" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(3), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(3), .kind = .rand });

    var ord = try Order.init(gpa, &ir, &.{ x, y }, &.{}, 64);
    defer ord.deinit(gpa);

    try std.testing.expectEqual(@as(u32, 6), ord.levels);

    // x2 y2 x1 y1 x0 y0 — alternating slots, descending bit position.
    const expected = [_]Order.Bit{
        .{ .slot = 0, .bit = 2 }, .{ .slot = 1, .bit = 2 },
        .{ .slot = 0, .bit = 1 }, .{ .slot = 1, .bit = 1 },
        .{ .slot = 0, .bit = 0 }, .{ .slot = 1, .bit = 0 },
    };
    try std.testing.expectEqualSlices(Order.Bit, &expected, ord.bits);

    // The reverse map agrees with the forward one.
    for (ord.bits, 0..) |b, lvl| try std.testing.expectEqual(@as(u32, @intCast(lvl)), ord.level(b.slot, b.bit));
}

test "mixed widths align by significance, not by bit index" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const narrow = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(2), .kind = .rand });
    const wide = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });

    var ord = try Order.init(gpa, &ir, &.{ narrow, wide }, &.{}, 64);
    defer ord.deinit(gpa);

    try std.testing.expectEqual(@as(u32, 6), ord.levels);

    // The wide variable's top two bits have no counterpart, so they lead alone;
    // from significance 1 down the two variables interleave.
    const expected = [_]Order.Bit{
        .{ .slot = 1, .bit = 3 },
        .{ .slot = 1, .bit = 2 },
        .{ .slot = 0, .bit = 1 },
        .{ .slot = 1, .bit = 1 },
        .{ .slot = 0, .bit = 0 },
        .{ .slot = 1, .bit = 0 },
    };
    try std.testing.expectEqualSlices(Order.Bit, &expected, ord.bits);

    // Equal significance shares adjacent levels, which is what keeps carries
    // and comparisons between differently sized variables linear.
    try std.testing.expectEqual(ord.level(0, 1) + 1, ord.level(1, 1));
}

test "a variable used only as a shift amount leads the order" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // x << s == 4, with s sized as the IR requires of an amount.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const s = try ir.addVariable(gpa, .{
        .id = @enumFromInt(1),
        .ty = Type.bit(Ir.shiftAmountWidth(4)),
        .kind = .rand,
    });
    const shifted = try ir.binary(gpa, .sll, try ir.varRef(gpa, x), try ir.varRef(gpa, s));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{
        try ir.binary(gpa, .eq, shifted, try ir.constInt(gpa, 4, Type.bit(4))),
    });

    const amount_only = try Order.shiftAmountOnly(gpa, &ir);
    defer gpa.free(amount_only);
    try std.testing.expect(!amount_only[0]); // x is a value
    try std.testing.expect(amount_only[1]); // s is only ever an amount

    var ord = try Order.init(gpa, &ir, &.{ x, s }, amount_only, 64);
    defer ord.deinit(gpa);

    // Both of s's bits come first; x's four follow, still most significant
    // first. Without this the amount would sit at the bottom, since its bits
    // are all low-significance.
    const expected = [_]Order.Bit{
        .{ .slot = 1, .bit = 1 },
        .{ .slot = 1, .bit = 0 },
        .{ .slot = 0, .bit = 3 },
        .{ .slot = 0, .bit = 2 },
        .{ .slot = 0, .bit = 1 },
        .{ .slot = 0, .bit = 0 },
    };
    try std.testing.expectEqualSlices(Order.Bit, &expected, ord.bits);
}

test "a variable used as a value as well keeps its normal place" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // s is a shift amount in one statement and a plain value in another, so
    // hoisting it could break an interleaving the second one depends on.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const s = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(2), .kind = .rand });
    const shifted = try ir.binary(gpa, .sll, try ir.varRef(gpa, x), try ir.varRef(gpa, s));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{
        try ir.binary(gpa, .eq, shifted, try ir.constInt(gpa, 4, Type.bit(4))),
        try ir.binary(gpa, .ugt, try ir.varRef(gpa, s), try ir.constInt(gpa, 0, Type.bit(2))),
    });

    const amount_only = try Order.shiftAmountOnly(gpa, &ir);
    defer gpa.free(amount_only);
    try std.testing.expect(!amount_only[1]);

    // So the order is the plain significance interleaving.
    var ord = try Order.init(gpa, &ir, &.{ x, s }, amount_only, 64);
    defer ord.deinit(gpa);
    try std.testing.expectEqual(Order.Bit{ .slot = 0, .bit = 3 }, ord.bits[0]);
}

test "an oversized component is rejected up front" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const v = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(64), .kind = .rand });
    try std.testing.expectError(error.TooManyBits, Order.init(gpa, &ir, &.{v}, &.{}, 32));
}

test "a variable-free component has no levels" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    var ord = try Order.init(gpa, &ir, &.{}, &.{}, 64);
    defer ord.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 0), ord.levels);
}

test {
    std.testing.refAllDecls(@This());
}

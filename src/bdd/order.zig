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

/// Assign levels to the bits of `vars`, in the order they are given.
pub fn init(
    gpa: Allocator,
    ir: *const Ir,
    vars: []const Ir.Variable.Index,
    max_levels: u32,
) Error!Order {
    var total: u64 = 0;
    var widest: u16 = 0;
    for (vars) |v| {
        const w = widthOf(ir, v);
        total += w;
        widest = @max(widest, w);
    }
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

    // Walk significance from most to least; at each step every variable wide
    // enough to reach that significance contributes its bit, in slot order.
    var next: u32 = 0;
    var significance: u16 = widest;
    while (significance > 0) {
        significance -= 1;
        for (vars, 0..) |v, slot| {
            // A bit's significance *is* its position: bit `k` carries weight
            // 2^k whatever the variable's width. Aligning on the position (and
            // not on distance from the top) is what keeps arithmetic between
            // differently sized variables linear.
            if (widthOf(ir, v) <= significance) continue;
            self.bits[next] = .{ .slot = @intCast(slot), .bit = significance };
            self.level_of[self.slot_start[slot] + significance] = next;
            next += 1;
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

    var ord = try Order.init(gpa, &ir, &.{ x, y }, 64);
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

    var ord = try Order.init(gpa, &ir, &.{ narrow, wide }, 64);
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

test "an oversized component is rejected up front" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const v = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(64), .kind = .rand });
    try std.testing.expectError(error.TooManyBits, Order.init(gpa, &ir, &.{v}, 32));
}

test "a variable-free component has no levels" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    var ord = try Order.init(gpa, &ir, &.{}, 64);
    defer ord.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 0), ord.levels);
}

test {
    std.testing.refAllDecls(@This());
}

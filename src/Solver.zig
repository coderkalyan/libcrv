//! Standardized constraint-solver interface.
//!
//! A `Solver` is a type-erased handle — a pointer plus a vtable, in the style of
//! `std.mem.Allocator` — over a concrete solver that has been bound to an `Ir`.
//! Each call to `next` draws one satisfying assignment of the IR's variables.
//! Callers can therefore swap engines (rejection sampling today; exact/SMT/BDD
//! backends later) behind one API without changing their code.
//!
//! Values are little-endian limb (`u64`) vectors. Each variable occupies
//! `valueLimbs(ir)` consecutive limbs in the output buffer — 1 in the common
//! case where every variable fits in 64 bits, so `out[i]` is just variable
//! `i`'s value. The buffer passed to `next` must therefore be at least
//! `valueLimbs(ir) * ir.vars.len` long; on a `true` result it holds the
//! assignment, variable `i` at `out[i * valueLimbs .. ][0..valueLimbs]`.

const std = @import("std");
const Ir = @import("Ir.zig");

const Solver = @This();

comptime {
    // A solution buffer is a vector of `Value`s, while `valueLimbs` counts
    // big-int limbs; the two descriptions only coincide where a limb is 64 bits
    // wide. Every 64-bit target qualifies. Stating it here turns a 32-bit build
    // into a build error rather than a silent disagreement about the layout of
    // every value the solver hands back.
    if (@bitSizeOf(std.math.big.Limb) != 64) @compileError("libcrv requires a 64-bit big-int limb");
}

/// One limb of a variable's value (a 64-bit little-endian word).
pub const Value = u64;

/// Number of limbs each variable value occupies in a solution buffer:
/// `ceil(maxVarWidth / 64)`, at least 1. Solver-agnostic — it depends only on
/// the IR's variable widths — so callers size `out` as `valueLimbs(ir) *
/// ir.vars.len`.
pub fn valueLimbs(ir: *const Ir) usize {
    var max_width: u16 = 1;
    for (ir.vars.items(.ty)) |ty| {
        const w: u16 = if (ty.width == 0) 64 else ty.width;
        max_width = @max(max_width, w);
    }
    return std.math.big.int.calcTwosCompLimbCount(max_width);
}

/// Pointer to the concrete solver instance.
ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    /// Draw one satisfying assignment into `out` and return `true`, or return
    /// `false` if the solver gave up. For an incomplete engine such as
    /// rejection sampling, `false` means "no assignment found within budget",
    /// which does not prove the constraints are unsatisfiable.
    next: *const fn (ptr: *anyopaque, out: []Value) bool,
};

/// Draw one satisfying assignment. See `VTable.next`.
pub fn next(self: Solver, out: []Value) bool {
    return self.vtable.next(self.ptr, out);
}

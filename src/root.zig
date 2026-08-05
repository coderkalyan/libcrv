//! libcrv — a constrained random verification library.
//!
//! Public API root. Downstream code pulls this in with `@import("crv")`.

const std = @import("std");

/// The intermediate representation: a flattened-tree IR that constraint queries
/// are built into, and that the solver consumes. See `Ir.zig` for the layout.
pub const Ir = @import("Ir.zig");

/// Standardized, swappable solver interface (type-erased pointer + vtable).
pub const Solver = @import("Solver.zig");

/// Solver-independent static analyses over the IR: which bits of each variable
/// a constraint can actually observe, and which variables are pinned by the
/// others. Any engine can use these to shrink what it has to reason about.
pub const Analysis = @import("Analysis.zig");

/// A rejection-sampling solver: draw random values, evaluate the IR, accept the
/// first assignment that satisfies every constraint.
pub const RejectionSampler = @import("RejectionSampler.zig");

test {
    std.testing.refAllDecls(@This());
}

//! libcrv — a constrained random verification library.
//!
//! Public API root. Downstream code pulls this in with `@import("crv")`.

const std = @import("std");

/// The intermediate representation: a flattened-tree IR that constraint queries
/// are built into, and that the solver consumes. See `Ir.zig` for the layout.
pub const Ir = @import("Ir.zig");

/// Splits a constraint set into independent sub-problems. Solver-agnostic:
/// any engine benefits from solving and sampling components separately.
pub const Partition = @import("Partition.zig");

/// Standardized, swappable solver interface (type-erased pointer + vtable).
pub const Solver = @import("Solver.zig");

/// A rejection-sampling solver: draw random values, evaluate the IR, accept the
/// first assignment that satisfies every constraint.
pub const RejectionSampler = @import("RejectionSampler.zig");

/// A BDD solver: compile the constraints into a decision diagram once, then
/// draw exactly uniform assignments from it in one root-to-leaf walk each.
pub const BddSolver = @import("BddSolver.zig");

test {
    std.testing.refAllDecls(@This());
}

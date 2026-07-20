//! libcrv — a constrained random verification library.
//!
//! Public API root. Downstream code pulls this in with `@import("crv")`.

const std = @import("std");

/// The intermediate representation: a flattened-tree IR that constraint queries
/// are built into, and that the solver consumes. See `Ir.zig` for the layout.
pub const Ir = @import("Ir.zig");

test {
    std.testing.refAllDecls(@This());
}

//! Constraint partitioning — splitting a constraint set into independent
//! sub-problems.
//!
//! Two constraint statements interact only if their expression cones share a
//! variable. Statements that share nothing can be solved, and sampled,
//! completely independently: the joint distribution over all variables is the
//! product of the per-component distributions, so drawing each component on its
//! own is *exact*, not an approximation.
//!
//! That matters more than it might sound. Real constraint sets are archipelagos
//! — a packet class has an address block, a length block, and a QoS block that
//! barely touch — and any engine whose cost is superlinear in problem size (a
//! BDD's is, in the worst case exponential) pays that cost per component rather
//! than once over the whole set. It also localizes failure: one intractable
//! island does not poison the tractable ones.
//!
//! None of that is specific to one engine, which is why this sits beside `Ir`
//! rather than inside a solver — a future SAT, interval, or hybrid engine wants
//! the same decomposition.
//!
//! The analysis is two linear sweeps over the flattened IR, no worklist:
//!
//!   1. **Reachability** — mark the nodes reachable from some constraint body.
//!      Because a node only ever refers to lower indices, one *reverse* sweep
//!      suffices. Dead IR (built but never constrained) is skipped, so it
//!      cannot fuse two components that are genuinely independent.
//!
//!   2. **Union-find** — one *forward* sweep carrying, per node, a single
//!      representative variable of its cone. Each node unions its children's
//!      representatives, so by the time the sweep reaches a statement, every
//!      variable in that statement's cone is in one set.
//!
//! Grouping is per *statement*, not per constraint block: two statements inside
//! one `constraint { ... }` are separate requirements, so keeping them apart
//! yields finer components.
//!
//! A statement whose cone names no variable (`1 == 0`) is still live — one of
//! them can render the whole set unsatisfiable — so those are collected into a
//! single variable-free component rather than dropped.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");

const Partition = @This();

/// The independent sub-problems, in no particular order. At most one component
/// has an empty `vars`: the one holding the constant-valued statements.
components: []Component,
/// Backing storage for every component's `vars`.
var_pool: []Ir.Variable.Index,
/// Backing storage for every component's `stmts`.
stmt_pool: []Ir.Node.Index,
/// Variables named by no live constraint. Nothing restricts them, so a solver
/// may draw them independently and need not model them at all.
free_vars: []Ir.Variable.Index,

pub const Component = struct {
    /// Variables this component owns, ascending by index.
    vars: []const Ir.Variable.Index,
    /// Constraint statements over exactly those variables.
    stmts: []const Ir.Node.Index,
};

/// Sentinel for "this node's cone names no variable".
const none = std.math.maxInt(u32);

/// Partition `ir`'s constraint statements. The result owns its storage and must
/// be `deinit`ed.
pub fn init(gpa: Allocator, ir: *const Ir) Allocator.Error!Partition {
    const node_count = ir.nodes.len;
    const var_count = ir.vars.len;

    var self: Partition = .{
        .components = &.{},
        .var_pool = &.{},
        .stmt_pool = &.{},
        .free_vars = &.{},
    };
    errdefer self.deinit(gpa);

    // Every constraint statement, flattened in declaration order.
    var stmts: std.ArrayListUnmanaged(Ir.Node.Index) = .empty;
    defer stmts.deinit(gpa);
    for (ir.constraints.items(.body)) |body| {
        const raw = ir.extra.items[@intFromEnum(body.start)..][0..body.len];
        try stmts.appendSlice(gpa, @ptrCast(raw));
    }

    // -- Sweep 1: reachability from the constraint bodies ---------------------

    const live = try gpa.alloc(bool, node_count);
    defer gpa.free(live);
    @memset(live, false);
    for (stmts.items) |s| live[@intFromEnum(s)] = true;

    // Children always have a lower index than their parent, so marking in
    // reverse index order propagates liveness in one pass.
    var n = node_count;
    while (n > 0) {
        n -= 1;
        if (live[n]) ir.forEachChild(@enumFromInt(@as(u32, @intCast(n))), live, markLive);
    }

    // -- Sweep 2: union-find over the live cones ------------------------------

    const uf = try gpa.alloc(u32, var_count);
    defer gpa.free(uf);
    for (uf, 0..) |*slot, v| slot.* = @intCast(v);

    const rep = try gpa.alloc(u32, node_count);
    defer gpa.free(rep);
    @memset(rep, none);

    var folder: Fold = .{ .uf = uf, .rep = rep, .acc = none };
    for (0..node_count) |i| {
        if (!live[i]) continue;
        folder.acc = none;
        ir.forEachChild(@enumFromInt(@as(u32, @intCast(i))), &folder, Fold.visit);
        rep[i] = folder.acc;
    }

    // -- Assign every statement and variable to a component slot --------------

    // Union-find root (or `none`, for the constant statements) -> component.
    var slot_of_root: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer slot_of_root.deinit(gpa);

    const stmt_slot = try gpa.alloc(u32, stmts.items.len);
    defer gpa.free(stmt_slot);

    var n_components: u32 = 0;
    for (stmts.items, stmt_slot) |s, *slot| {
        const r = rep[@intFromEnum(s)];
        const root = if (r == none) none else find(uf, r);
        const gop = try slot_of_root.getOrPut(gpa, root);
        if (!gop.found_existing) {
            gop.value_ptr.* = n_components;
            n_components += 1;
        }
        slot.* = gop.value_ptr.*;
    }

    // A variable belongs to a component only if some statement landed on its
    // root; everything else is free.
    const var_slot = try gpa.alloc(u32, var_count);
    defer gpa.free(var_slot);

    var free_count: u32 = 0;
    for (var_slot, 0..) |*slot, v| {
        slot.* = slot_of_root.get(find(uf, @intCast(v))) orelse none;
        if (slot.* == none) free_count += 1;
    }

    // -- Carve the pools into per-component windows ---------------------------

    self.components = try gpa.alloc(Component, n_components);
    self.stmt_pool = try gpa.alloc(Ir.Node.Index, stmts.items.len);
    self.var_pool = try gpa.alloc(Ir.Variable.Index, var_count - free_count);
    self.free_vars = try gpa.alloc(Ir.Variable.Index, free_count);

    const stmt_at = try gpa.alloc(u32, n_components + 1);
    defer gpa.free(stmt_at);
    const var_at = try gpa.alloc(u32, n_components + 1);
    defer gpa.free(var_at);
    @memset(stmt_at, 0);
    @memset(var_at, 0);

    for (stmt_slot) |slot| stmt_at[slot + 1] += 1;
    for (var_slot) |slot| if (slot != none) {
        var_at[slot + 1] += 1;
    };
    for (1..n_components + 1) |k| {
        stmt_at[k] += stmt_at[k - 1];
        var_at[k] += var_at[k - 1];
    }

    for (self.components, 0..) |*c, k| c.* = .{
        .stmts = self.stmt_pool[stmt_at[k]..stmt_at[k + 1]],
        .vars = self.var_pool[var_at[k]..var_at[k + 1]],
    };

    // `stmt_at[k]` / `var_at[k]` double as fill cursors: each starts at its
    // window's base and is bumped as entries land.
    for (stmts.items, stmt_slot) |s, slot| {
        self.stmt_pool[stmt_at[slot]] = s;
        stmt_at[slot] += 1;
    }
    var free_at: u32 = 0;
    for (var_slot, 0..) |slot, v| {
        const index: Ir.Variable.Index = @enumFromInt(@as(u32, @intCast(v)));
        if (slot == none) {
            self.free_vars[free_at] = index;
            free_at += 1;
        } else {
            self.var_pool[var_at[slot]] = index;
            var_at[slot] += 1;
        }
    }

    return self;
}

pub fn deinit(self: *Partition, gpa: Allocator) void {
    gpa.free(self.components);
    gpa.free(self.var_pool);
    gpa.free(self.stmt_pool);
    gpa.free(self.free_vars);
    self.* = undefined;
}

fn markLive(live: []bool, child: Ir.Child) void {
    switch (child) {
        .node => |node| live[@intFromEnum(node)] = true,
        .variable => {},
    }
}

/// Folds one node's children into a single union-find set, carrying out a
/// representative variable for the node's whole cone. `rep` holds the already
/// computed representatives of lower-indexed nodes.
const Fold = struct {
    uf: []u32,
    rep: []const u32,
    acc: u32,

    fn visit(self: *Fold, child: Ir.Child) void {
        const v = switch (child) {
            .variable => |index| @intFromEnum(index),
            .node => |node| self.rep[@intFromEnum(node)],
        };
        if (v == none) return;
        if (self.acc == none) self.acc = v else unite(self.uf, self.acc, v);
    }
};

fn find(uf: []u32, x: u32) u32 {
    var root = x;
    while (uf[root] != root) root = uf[root];
    // Path compression keeps repeated lookups near-constant.
    var cur = x;
    while (uf[cur] != root) {
        const next = uf[cur];
        uf[cur] = root;
        cur = next;
    }
    return root;
}

fn unite(uf: []u32, a: u32, b: u32) void {
    const ra = find(uf, a);
    const rb = find(uf, b);
    if (ra != rb) uf[@max(ra, rb)] = @min(ra, rb);
}

// -- Tests -------------------------------------------------------------------

const Type = Ir.Type;

test "independent constraints split into separate components" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // x and y are constrained separately; z is never mentioned.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
    _ = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(4), .kind = .rand });

    const cx = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 3, Type.bit(4)));
    const cy = try ir.binary(gpa, .eq, try ir.varRef(gpa, y), try ir.constInt(gpa, 5, Type.bit(4)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ cx, cy });

    var p = try Partition.init(gpa, &ir);
    defer p.deinit(gpa);

    // Two statements in one block, but no shared variable: two components.
    try std.testing.expectEqual(@as(usize, 2), p.components.len);
    for (p.components) |c| {
        try std.testing.expectEqual(@as(usize, 1), c.vars.len);
        try std.testing.expectEqual(@as(usize, 1), c.stmts.len);
    }
    // z is unconstrained.
    try std.testing.expectEqual(@as(usize, 1), p.free_vars.len);
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(p.free_vars[0]));
}

test "a shared variable fuses statements into one component" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // x < y and y < z chain all three together.
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });
    const z = try ir.addVariable(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(4), .kind = .rand });

    const a = try ir.binary(gpa, .ult, try ir.varRef(gpa, x), try ir.varRef(gpa, y));
    const b = try ir.binary(gpa, .ult, try ir.varRef(gpa, y), try ir.varRef(gpa, z));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{a});
    _ = try ir.addConstraint(gpa, @enumFromInt(1), .{}, &.{b});

    var p = try Partition.init(gpa, &ir);
    defer p.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), p.components.len);
    try std.testing.expectEqual(@as(usize, 3), p.components[0].vars.len);
    try std.testing.expectEqual(@as(usize, 2), p.components[0].stmts.len);
    try std.testing.expectEqual(@as(usize, 0), p.free_vars.len);
}

test "dead IR does not fuse independent components" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const y = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(4), .kind = .rand });

    // An expression naming both, built but never constrained.
    _ = try ir.binary(gpa, .add, try ir.varRef(gpa, x), try ir.varRef(gpa, y));

    const cx = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 1, Type.bit(4)));
    const cy = try ir.binary(gpa, .eq, try ir.varRef(gpa, y), try ir.constInt(gpa, 2, Type.bit(4)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ cx, cy });

    var p = try Partition.init(gpa, &ir);
    defer p.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), p.components.len);
}

test "variable-free statements form their own component" {
    const gpa = std.testing.allocator;
    var ir: Ir = .{};
    defer ir.deinit(gpa);

    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(4), .kind = .rand });
    const contradiction = try ir.binary(
        gpa,
        .eq,
        try ir.constInt(gpa, 1, Type.bit(4)),
        try ir.constInt(gpa, 0, Type.bit(4)),
    );
    const cx = try ir.binary(gpa, .eq, try ir.varRef(gpa, x), try ir.constInt(gpa, 3, Type.bit(4)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{ contradiction, cx });

    var p = try Partition.init(gpa, &ir);
    defer p.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), p.components.len);
    var saw_constant = false;
    for (p.components) |c| {
        if (c.vars.len == 0) {
            saw_constant = true;
            try std.testing.expectEqual(@as(usize, 1), c.stmts.len);
        }
    }
    try std.testing.expect(saw_constant);
}

test {
    std.testing.refAllDecls(@This());
}

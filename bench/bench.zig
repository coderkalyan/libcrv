//! Wall-clock benchmarks for `crv.SmtSampler`.
//!
//!     zig build bench -Dbitwuzla -Dbitwuzla-include=... -Dbitwuzla-lib=...
//!
//! Reports the two numbers that matter separately, because they have wildly
//! different magnitudes and only one of them is paid per `randomize()`:
//! **compile** (encode, minimize the support, count — once) and **per sample**
//! (a batch pop, or a cell enumeration amortized over the batch).
//!
//! `bench/distribution.py` covers the other half of the question — whether the
//! samples are actually uniform — by porting this engine's control flow and
//! running it against brute-force ground truth. That one needs no solver.

const std = @import("std");
const crv = @import("crv");

const Case = struct {
    name: []const u8,
    note: []const u8,
    build: *const fn (std.mem.Allocator, *crv.Ir) anyerror!void,
};

const Type = crv.Ir.Type;

/// A 64-bit variable whose low byte alone is constrained. The support should
/// collapse to 8 bits and leave 56 free, which is the case the demanded-bit
/// analysis exists for.
fn freeBits(gpa: std.mem.Allocator, ir: *crv.Ir) !void {
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(64), .kind = .rand });
    const low = try ir.binary(gpa, .band, try ir.varRef(gpa, x), try ir.constInt(gpa, 0xff, Type.bit(64)));
    const member = try ir.in(gpa, low, &.{
        try ir.range(gpa, try ir.constInt(gpa, 10, Type.bit(64)), try ir.constInt(gpa, 20, Type.bit(64))),
    });
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{member});
}

/// Dense but irregular: about a seventh of the space, far past `exact_limit`,
/// so this exercises the counting and cell-sampling path end to end.
fn modular(gpa: std.mem.Allocator, ir: *crv.Ir) !void {
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(20), .kind = .rand });
    const rem = try ir.binary(gpa, .urem, try ir.varRef(gpa, x), try ir.constInt(gpa, 7, Type.bit(20)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{
        try ir.binary(gpa, .eq, rem, try ir.constInt(gpa, 3, Type.bit(20))),
    });
}

/// A multiplier feeding a modulo: roughly one solution in 500, scattered with
/// no structure. Rejection sampling needs ~500 draws per hit here.
fn quadratic(gpa: std.mem.Allocator, ir: *crv.Ir) !void {
    const x = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(20), .kind = .rand });
    const wide = try ir.zext(gpa, try ir.varRef(gpa, x), 40);
    const sq = try ir.binary(gpa, .mul, wide, wide);
    const rem = try ir.binary(gpa, .urem, sq, try ir.constInt(gpa, 1021, Type.bit(40)));
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{
        try ir.binary(gpa, .eq, rem, try ir.constInt(gpa, 835, Type.bit(40))),
    });
}

/// Two coupled 16-bit operands through a multiplier — the shape a decision
/// diagram blows up on, and the reason this engine exists.
fn multiplier(gpa: std.mem.Allocator, ir: *crv.Ir) !void {
    const a = try ir.addVariable(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(16), .kind = .rand });
    const b = try ir.addVariable(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(16), .kind = .rand });
    const wa = try ir.zext(gpa, try ir.varRef(gpa, a), 32);
    const wb = try ir.zext(gpa, try ir.varRef(gpa, b), 32);
    _ = try ir.addConstraint(gpa, @enumFromInt(0), .{}, &.{
        try ir.binary(gpa, .eq, try ir.binary(gpa, .mul, wa, wb), try ir.constInt(gpa, 1440, Type.bit(32))),
    });
}

const cases = [_]Case{
    .{ .name = "free bits", .note = "64-bit x, only the low byte constrained", .build = freeBits },
    .{ .name = "modular", .note = "20-bit x, x % 7 == 3 (|S| ~ 150k)", .build = modular },
    .{ .name = "quadratic", .note = "20-bit x, x*x % 1021 == 835 (1 in ~500)", .build = quadratic },
    .{ .name = "multiplier", .note = "16-bit a*b == 1440, coupled operands", .build = multiplier },
};

const samples = 10_000;

pub fn main() !void {
    var gpa_state: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    std.debug.print("SmtSampler benchmarks ({d} samples per case)\n\n", .{samples});

    for (cases) |case| {
        var ir: crv.Ir = .{};
        defer ir.deinit(gpa);
        try case.build(gpa, &ir);

        std.debug.print("{s}\n  {s}\n", .{ case.name, case.note });

        var timer = try std.time.Timer.start();
        var sampler = crv.SmtSampler.init(gpa, &ir, .{ .seed = 1 }) catch |err| {
            std.debug.print("  init failed: {s}\n\n", .{@errorName(err)});
            continue;
        };
        defer sampler.deinit(gpa);
        const compile_ns = timer.read();

        const limbs = crv.Solver.valueLimbs(&ir);
        const out = try gpa.alloc(crv.Solver.Value, limbs * ir.vars.len);
        defer gpa.free(out);

        const solver = sampler.solver();
        timer.reset();
        var drawn: usize = 0;
        for (0..samples) |_| {
            if (!solver.next(out)) break;
            drawn += 1;
        }
        const sample_ns = timer.read();

        std.debug.print("  guarantee   {s}\n", .{@tagName(sampler.guarantee())});
        std.debug.print("  compile     {d:.1} ms\n", .{@as(f64, @floatFromInt(compile_ns)) / 1e6});
        if (drawn == 0) {
            std.debug.print("  sampling    produced nothing\n\n", .{});
            continue;
        }
        std.debug.print("  per sample  {d:.0} ns  ({d} drawn)\n", .{
            @as(f64, @floatFromInt(sample_ns)) / @as(f64, @floatFromInt(drawn)),
            drawn,
        });
        std.debug.print("  throughput  {d:.0} samples/s\n\n", .{
            @as(f64, @floatFromInt(drawn)) * 1e9 / @as(f64, @floatFromInt(sample_ns)),
        });
    }
}

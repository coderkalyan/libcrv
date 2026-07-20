const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The library's root module. Downstream projects add this repository as a
    // dependency and import it with `@import("crv")`.
    const mod = b.addModule("crv", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static library artifact. Handy for C consumers and for verifying that
    // the module compiles and links on its own.
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "crv",
        .root_module = mod,
    });
    b.installArtifact(lib);

    // `zig build test` — run the unit tests declared in the library sources.
    const lib_tests = b.addTest(.{ .root_module = mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
}

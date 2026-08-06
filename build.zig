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

    // The C API is rooted in its own file rather than in `src/root.zig`, so a
    // Zig program that imports the module above emits none of the `crv_*`
    // symbols and cannot collide with a libcrv it also links.
    const c_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    // The header, translated to Zig, so `c_api.zig` can assert that `crv_op`
    // and friends carry the IR's own enum values. This goes through a build
    // step rather than `@cImport` because only a step makes `crv.h` a tracked
    // input: an in-source `@cImport` is cached against the Zig files alone, so
    // editing the header would leave the check reporting a stale pass.
    const header = b.addTranslateC(.{
        .root_source_file = b.path("include/crv.h"),
        .target = target,
        .optimize = optimize,
    });
    c_mod.addImport("crv.h", header.createModule());

    const static = b.addLibrary(.{
        .linkage = .static,
        .name = "crv",
        .root_module = c_mod,
    });
    static.installHeader(b.path("include/crv.h"), "crv.h");
    b.installArtifact(static);

    const shared = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "crv",
        .root_module = c_mod,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });
    b.installArtifact(shared);

    // `zig build test` — the Zig unit tests, the C API's own tests, and a smoke
    // test that compiles the real header as C and links the real library.
    const lib_tests = b.addTest(.{ .root_module = mod });
    const c_api_tests = b.addTest(.{ .root_module = c_mod });

    const smoke_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    smoke_mod.addCSourceFile(.{
        .file = b.path("test/smoke.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror" },
    });
    smoke_mod.addIncludePath(b.path("include"));
    smoke_mod.linkLibrary(static);
    const smoke = b.addExecutable(.{ .name = "crv-smoke", .root_module = smoke_mod });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(lib_tests).step);
    test_step.dependOn(&b.addRunArtifact(c_api_tests).step);
    test_step.dependOn(&b.addRunArtifact(smoke).step);
}

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The SMT backend is opt-in, and off by default: libcrv builds and tests
    // with no external dependency at all. Enabling it links a system-installed
    // libbitwuzla, which itself pulls in GMP, MPFR, and a SAT solver.
    //
    //     zig build -Dbitwuzla \
    //         -Dbitwuzla-include=/usr/local/include \
    //         -Dbitwuzla-lib=/usr/local/lib
    //
    // `SmtSampler` defaults to CryptoMiniSat as the SAT engine, so libbitwuzla
    // should be built with CryptoMiniSat support; see `SmtSampler.Options`.
    const bitwuzla = b.option(
        bool,
        "bitwuzla",
        "Link the Bitwuzla SMT backend, enabling crv.SmtSampler (default: false)",
    ) orelse false;
    const bitwuzla_include = b.option(
        []const u8,
        "bitwuzla-include",
        "Directory containing bitwuzla/c/bitwuzla.h",
    );
    const bitwuzla_lib = b.option(
        []const u8,
        "bitwuzla-lib",
        "Directory containing libbitwuzla",
    );

    // Two files with the same public API; the rest of the library is written
    // against that API and so compiles identically either way.
    const backend = b.createModule(.{
        .root_source_file = b.path(if (bitwuzla)
            "src/bitwuzla/enabled.zig"
        else
            "src/bitwuzla/disabled.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (bitwuzla) {
        if (bitwuzla_include) |dir| backend.addIncludePath(.{ .cwd_relative = dir });
        if (bitwuzla_lib) |dir| backend.addLibraryPath(.{ .cwd_relative = dir });
        backend.link_libc = true;
        backend.link_libcpp = true;
        backend.linkSystemLibrary("bitwuzla", .{});
    }

    // The library's root module. Downstream projects add this repository as a
    // dependency and import it with `@import("crv")`.
    const mod = b.addModule("crv", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "bitwuzla", .module = backend }},
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

    // `zig build bench` — wall-clock numbers for the solver engines. Needs the
    // backend; without it every case reports `BackendUnavailable`. The
    // distribution half of the picture is `bench/distribution.py`, which needs
    // no solver at all.
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "crv", .module = mod }},
    });
    const bench_exe = b.addExecutable(.{ .name = "bench", .root_module = bench_mod });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.step.dependOn(b.getInstallStep());

    const bench_step = b.step("bench", "Run solver benchmarks");
    bench_step.dependOn(&run_bench.step);
}

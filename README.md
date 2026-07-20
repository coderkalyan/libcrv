# libcrv

A Zig library. This is an initial scaffold — replace the placeholder API in
`src/root.zig` with the real thing.

## Requirements

- [Zig](https://ziglang.org/) `0.16.0` or newer.

## Building and testing

```sh
zig build          # build the static library into zig-out/
zig build test     # run the unit tests
```

## Using it as a dependency

Add libcrv to your project's `build.zig.zon`:

```sh
zig fetch --save git+https://github.com/coderkalyan/libcrv
```

Then wire the module into your `build.zig`:

```zig
const crv = b.dependency("libcrv", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("crv", crv.module("crv"));
```

And import it in your code:

```zig
const std = @import("std");
const crv = @import("crv");

pub fn build(gpa: std.mem.Allocator) !crv.Ir {
    var ir: crv.Ir = .{};

    // Ids are yours to assign — e.g. indices into your symbol table.
    const x_id: crv.Ir.Variable.Id = @enumFromInt(1);
    const c_id: crv.Ir.Constraint.Id = @enumFromInt(2);

    // rand bit [3:0] x;  constraint c { x inside {[0:15]}; }
    const x = try ir.addVariable(gpa, .{ .id = x_id, .ty = crv.Ir.Type.bit(4), .kind = .rand });
    const membership = try ir.in(gpa, try ir.varRef(gpa, x), &.{
        try ir.range(gpa, try ir.constInt(gpa, 0, .bit(4)), try ir.constInt(gpa, 15, .bit(4))),
    });
    _ = try ir.addConstraint(gpa, c_id, .{}, &.{membership});

    return ir; // hash with ir.hash(); cache with ir.serialize(gpa)
}
```

## The IR

`crv.Ir` is a *flattened tree*: expression and constraint nodes live in one
contiguous array and refer to each other by index rather than by pointer. Those
indices are distinct `enum(u32)` types (`Node.Index`, `Variable.Index`, ...) so
the compiler catches an index used against the wrong table. Side tables
(`vars`, `constraints`, `extra`) hold the rest. Variables and constraints carry
an opaque per-instance `u32` id (`Variable.Id`, `Constraint.Id`) — the IR stores
no text; resolve an id against whatever symbol table your builder keeps on the
side.

Because it is all plain POD arrays:

- `ir.hash()` is a **Blake3** content digest over the arrays (256-bit;
  collisions are astronomically unlikely) — structurally identical constraint
  sets hash equal, so it doubles as a collision-resistant cache key.
- `ir.serialize(gpa)` / `Ir.deserialize(gpa, bytes)` blit the arrays to and
  from a byte buffer behind a `magic` + 32-bit `format_version` header, with a
  trailing Blake3 checksum that is verified *before* any length is trusted — so
  a corrupt or tampered cache file fails cleanly rather than driving an
  allocation off a garbage count.

The node tags cover the scalar constraint subset (bit-vector randomization,
arithmetic/relational/logical ops, `in`, `zext`/`sext`/`trunc` sizing casts,
`dist`, `if`/`else`, `unique`, `solve...before`). A value's type is **just a bit
width** — signedness lives on the operators (`slt`/`ult`, `sdiv`/`udiv`,
`sra`/`srl`, `sext`/`zext`), so any op that depends on signedness comes in a
signed and an unsigned form. Widths change only through the cast nodes. See the
module doc comment in [`src/Ir.zig`](src/Ir.zig) for the full node encoding and
the roadmap (arrays/`foreach`, wide literals).

## Layout

```
build.zig        # build graph: module, static library, test step
build.zig.zon    # package manifest (name, version, dependencies)
src/root.zig     # library root — the public API
src/Ir.zig       # the flattened-tree IR
```

## License

MIT — see [LICENSE](LICENSE).

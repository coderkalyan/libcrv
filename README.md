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

## Solving

`crv.Solver` is a swappable solver interface — a type-erased pointer + vtable,
like `std.mem.Allocator` — so an engine can be chosen at runtime. Each `next`
draws one satisfying assignment into a caller-provided `[]Value` (a `u64` per
variable, indexed by `Variable.Index`).

`crv.RejectionSampler` is the first engine: it seeds an RNG, draws a random
value for every variable, and evaluates the whole IR over that draw, accepting
the first assignment that satisfies every constraint. Evaluation is a single
allocation-free linear sweep over the packed node arrays (the IR is already in
evaluation order), so re-evaluation is tight and cache-friendly.

Node values live in two parallel vectors, always allocated: a `u64` vector and
a `std.math.big.int` vector over a limb pool. Each node is stored in one or the
other by its width (`<= 64` bits → the `u64` vector, else the big.int vector),
so a mostly-narrow IR keeps its narrow nodes on the fast integer path even when
a few nodes exceed 64 bits. The big.int ops run over the preallocated pool, so
the hot loop never allocates.

Values are little-endian limb (`u64`) vectors: each variable occupies
`Solver.valueLimbs(ir)` limbs — 1 in the common ≤64-bit case, so `out[i]` is
just variable `i`'s value.

```zig
var sampler = try crv.RejectionSampler.init(gpa, &ir, .{ .seed = 0 });
defer sampler.deinit(gpa);

const limbs = crv.Solver.valueLimbs(&ir);      // 1 unless a variable exceeds 64 bits
const out = try gpa.alloc(crv.Solver.Value, limbs * ir.vars.len);
defer gpa.free(out);

if (sampler.next(out)) {
    // variable i's value is out[i * limbs ..][0..limbs]
}
```

Evaluation is **strict and exact**: `ir.typeOf(node)` resolves every node's
width, and each operation evaluates at that width. Widths change only through
explicit `zext`/`sext`/`trunc` cast nodes — there is no implicit widening, so an
operator's result is its operands' width, wrapped there. (To add two 4-bit
values without wrapping, `zext` them to a wider type first.) Signedness is chosen
per operation (`slt` vs `ult`, `sdiv` vs `udiv`, `sra` vs `srl`), not carried
by the type. `ir.typeOf` resolves a node's width recursively: leaves/casts/literals are explicitly typed,
everything else propagates its operand type or yields a 1-bit `bool`.

It is incomplete — a `false` result means "no assignment found within the
attempt budget", not "unsatisfiable". The interpreter covers the scalar
expression subset (arithmetic/bitwise/shift, comparisons, logical ops,
`in`/`range`) and tracks cumulative `attempts`/`hits` counters; evaluating an unsupported node
(`dist` or the structural constraints) panics for now.

## Layout

```
build.zig                # build graph: module, static library, test step
build.zig.zon            # package manifest (name, version, dependencies)
src/root.zig             # library root — the public API
src/Ir.zig               # the flattened-tree IR
src/Solver.zig           # the swappable solver interface
src/RejectionSampler.zig # the rejection-sampling engine
```

## License

MIT — see [LICENSE](LICENSE).

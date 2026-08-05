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
signed and an unsigned form. Widths change only through the cast nodes.
Constants come from `constInt` (a 64-bit value) or `constBig` (an
arbitrary-precision `std.math.big.int.Const`); both are stored inline at their
declared width, so literals wider than 64 bits are first-class. See the module
doc comment in [`src/Ir.zig`](src/Ir.zig) for the full node encoding and the
roadmap (arrays/`foreach`).

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

### The BDD engine

`crv.BddSolver` compiles the whole constraint set into a reduced ordered binary
decision diagram once, then draws from it. The cost model inverts: setup is
expensive, and every draw afterwards is a single root-to-leaf walk that
allocates nothing and **cannot fail**.

```zig
var bdd = try crv.BddSolver.init(gpa, &ir, .{ .seed = 0 });
defer bdd.deinit(gpa);

if (bdd.isUnsat()) return error.Contradiction; // decided, not "gave up"
std.debug.print("{d} legal stimuli\n", .{@exp2(bdd.log2Count())});

var out: [1]crv.Solver.Value = undefined;
_ = bdd.next(&out); // exactly uniform over the solution set
```

Three things follow from having the whole solution set in hand, none of which a
sampler can offer at any price:

- **Exactly uniform** draws, however sparse the solution space. A rejection
  sampler that accepts one draw in 2³² is not slow, it is broken.
- **Unsatisfiability is decided.** `next` returns `false` only when no
  assignment exists — a contradiction in your constraints is a diagnosis, not a
  timeout.
- **The solution count is exact** (`log2Count`), which is a coverage
  measurement rather than a solver statistic.

Measured on a 32-bit address constrained to a 4 KiB window *and* 64-byte
aligned — 64 solutions out of 2³², where `RejectionSampler` accepts about one
draw in 67 million and gives up:

| Constraint set | Build | Per draw |
| --- | --- | --- |
| Windowed + aligned 32-bit address | 0.35 ms | 150 ns |
| `x + y == z`, three 32-bit variables | 0.5 ms | 620 ns |
| `x % 64 == 0`, 32-bit (power of two → wiring) | 0.5 ms | 50 ns |
| `x % 100 == 7`, 32-bit (restoring division array) | 18 ms | 190 ns |
| 64 independent 32-bit range constraints | 3 ms | 10.3 µs |

Four decisions carry that performance:

1. **Variable ordering** (`src/bdd/order.zig`) is significance-major and
   most-significant-first, interleaving the bits of variables that interact.
   `x + y == z` is linear under this order and *exponential* if each variable's
   bits are kept contiguous. This is not a tuning knob; without it the engine
   does not work.
2. **Partitioning** (`src/Partition.zig`) splits the constraints into
   independent components and solves each separately. Sampling them
   independently is exact — independence is what a component boundary means —
   and it scopes the expensive step to the largest island rather than the whole
   set.
3. **Model counting in log space.** Variables may be 65535 bits wide, so counts
   reach 2^65535; counts are kept as `log2` and converted once, at build time,
   into integer branch thresholds, so the sampling loop touches no floating
   point.
4. **No garbage collector.** Each component builds into a bump-allocated
   manager and is then frozen out into a compact read-only graph, after which
   the manager is reset in constant time. Collecting would mark nearly every
   node and sweep almost nothing.

`init` fails, loudly, rather than degrading:

| Error | Cause |
| --- | --- |
| `NodeBudgetExceeded` | the diagram outgrew `node_budget` (default 2²¹ nodes) |
| `TooManyBits` | one component spans more than `max_levels` random bits |
| `OperandTooWide` | variable-by-variable multiply/divide above `max_mul_width` |
| `UnsupportedNode` | `dist`, `solve_before`, or `foreach` |

`OperandTooWide` is a consequence of a theorem, not a missing optimization:
Bryant proved in 1991 that `x * y == z` has BDD size ≥ 2^(n/8) under *every*
variable order. Narrow multiplies are genuinely cheap and are compiled; wide
ones are hopeless and say so. Choosing a different engine on failure is the
caller's decision — a hybrid that degrades gracefully belongs behind its own
`Solver` implementation, not hidden inside this one.

Scope matches the rejection sampler plus `if_else` and `unique`. As there, and
so the two engines agree, `soft` constraint flags are treated as hard and
`randc` variables as plain `rand`. The engines are cross-checked against each
other: a differential test builds the same IR for both and compares their
*solution sets* exactly, which is what pins down the operator edge cases
(division by zero, `sra`'s sign width, saturating shift amounts, truncating
signed division).

## Layout

```
build.zig                # build graph: module, static library, test step
build.zig.zon            # package manifest (name, version, dependencies)
src/root.zig             # library root — the public API
src/Ir.zig               # the flattened-tree IR
src/Partition.zig        # splits constraints into independent sub-problems
src/Solver.zig           # the swappable solver interface
src/RejectionSampler.zig # the rejection-sampling engine
src/BddSolver.zig        # the BDD engine
src/bdd/Manager.zig      #   ROBDD package (unique table, ITE, complement edges)
src/bdd/order.zig        #   variable ordering: IR variable bits -> BDD levels
src/bdd/blast.zig        #   bit-blaster: IR expressions -> BDDs
src/bdd/sample.zig       #   model counting and uniform sampling
```

`Partition` sits outside `src/bdd/` on purpose: decomposing a constraint set
into independent sub-problems is useful to any engine, not just this one.
Bit-level ordering is BDD-specific and stays with it.

## License

MIT — see [LICENSE](LICENSE).

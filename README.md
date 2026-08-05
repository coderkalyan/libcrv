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
`srem`/`urem`, `sra`/`srl`, `sext`/`zext`), so any op that depends on signedness
comes in a signed and an unsigned form. `srem` is truncated remainder — its sign
follows the dividend, as `@rem` and SMT-LIB's `bvsrem` do, not `bvsmod` — and
every division yields 0 on a zero divisor, which SMT-LIB does *not* agree with,
so a backend has to encode that guard explicitly. Widths change only through the
cast nodes.
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

## The SMT sampler

`crv.SmtSampler` is the engine for what rejection sampling cannot reach: sparse
solution spaces (`x == 42` on a 32-bit variable is one draw in four billion) and
wide datapath arithmetic. It is meant to be *dispatched to* for hard constraint
sets, not tried first.

Bitwuzla runs as a **compiler, not an interpreter**. A solver call per
`randomize()` would be orders of magnitude too slow for a simulation loop, and
badly distributed besides — solvers answer from whatever corner their heuristics
reach first, and re-seeding does not fix that. So `init` does the expensive work
once and `next` draws from the result:

1. **Encode** the IR into Bitwuzla terms.
2. **Shrink** to an independent support — the bits that actually have to be
   reasoned about — structurally from `Analysis`, then by Padoa definability
   queries.
3. **Count** with ApproxMC. Its zero-hash step is exhaustive enumeration, so a
   small solution set comes back already listed and exactly counted.
4. **Sample** by enumerating one random hash cell per batch.

Uniformity comes from the hash family's pairwise independence, not from the
solver's behaviour, which is why solver bias never enters the picture.
`guarantee()` reports what the stream is actually distributed like — `.exact`,
`.almost_uniform`, or `.biased` — and is fixed once `init` returns. The biased
fallback is off by default: an instance that defeats the pipeline fails `init`
rather than quietly returning samples nobody asked for.

```zig
var s = try crv.SmtSampler.init(gpa, &ir, .{ .seed = 0 }); // the expensive part
defer s.deinit(gpa);
switch (s.guarantee()) { .exact => {}, else => {} }
_ = s.solver().next(out);                                   // a draw
```

The backend is **opt-in**, so the default build has no external dependencies:

```sh
zig build -Dbitwuzla -Dbitwuzla-include=/usr/local/include -Dbitwuzla-lib=/usr/local/lib
```

Without it `SmtSampler.init` returns `error.BackendUnavailable` and everything
else works unchanged. `libbitwuzla` should be built with CryptoMiniSat, which
`Options.sat_solver` selects by default — it recovers XOR structure from CNF and
can run Gauss-Jordan over it, which is what makes parity constraints affordable.

`dist`, `solve_before`, and `foreach` are rejected with `error.Unsupported`
rather than ignored, since each changes the distribution a sampler should
produce.

### Benchmarks

`zig build bench` reports wall-clock — compile once, then per sample — and
needs the backend. `bench/distribution.py` answers the other half, whether the
samples are actually uniform, by porting the engine's control flow and running
it against brute-force ground truth; it needs no solver, so it runs anywhere.
`bench/sweep.py` measures what the counting rounds buy.

Measured by the simulation, per solution `p_i * |S|` against the `1 + eps` band
and a chi-squared over the whole solution set:

| instance | \|S\| | path | chi²/dof | within 1+eps | calls/sample |
|---|---|---|---|---|---|
| `popcount(x)==3`, 18-bit | 816 | exact | 0.967 | 100% | 0 |
| `popcount(x)==3`, 18-bit | 816 | hashed | 0.852 | 100% | 3.3 |
| `popcount`, 1 sample/cell | 816 | hashed | 0.909 | 100% | 26.9 |
| `x*x % 1021 == 835`, 20-bit | 2054 | hashed | 1.043 | 100% | 4.1 |
| `a*b == 1440`, 11-bit | 36 | exact | 1.134 | 100% | 0 |
| low byte in `[10:20]`, 20-bit | 45056 | exact | 1.000 | — | 0 |

Across 12 seeds the mean chi-squared z-score is −0.06 on the hashed path and
+0.24 on the free-bit path, so there is no detectable systematic bias.

The counting default is a deliberate trade. The `1 - delta` bound needs
`ceil(17 log2(3/delta))` rounds — 84 for `delta = 0.1` — and `max_rounds`
defaults to 17, which **proves nothing** and reports `delta = 1.0` to say so.
Measured over 25 seeds it is nonetheless accurate:

| rounds | proven delta | estimates in band | spread | compile calls |
|---|---|---|---|---|
| 3 | vacuous | 25/25 | 0.75–1.20× | 824 |
| 9 | vacuous | 25/25 | 0.90–1.12× | 2326 |
| 17 (default) | vacuous | 25/25 | 0.92–1.04× | 4329 |
| 84 | 0.098 | 25/25 | 0.98–1.02× | 21046 |

Set `max_rounds = Hashing.roundsFor(delta)` when the guarantee itself matters
rather than just the accuracy.

## Static analysis

`crv.Analysis` is solver-independent, and any engine can use it:

- **Demanded bits** — per variable, the mask of bits some constraint can
  observe. `x & 0xff == 0x42` on a 32-bit `x` leaves 24 bits free, so a decision
  procedure sees 8 bits instead of 32 and the rest are drawn straight from the
  RNG. Since the solution set factors as (demanded assignments) × (free bits),
  drawing them independently is exactly uniform — but only if they are redrawn
  per sample rather than read back from one solver model.
- **Defined variables** — those pinned by a top-level `v == expr`, accepted only
  once everything they depend on is, which admits chains and rejects cycles.

Both are sound under-approximations: never a free bit that is observable, never
a pinned variable that is not.

## Layout

```
build.zig                # build graph: module, static library, test step
build.zig.zon            # package manifest (name, version, dependencies)
src/root.zig             # library root — the public API
src/Ir.zig               # the flattened-tree IR
src/Solver.zig           # the swappable solver interface
src/Analysis.zig         # demanded-bit and definition analyses
src/RejectionSampler.zig # the rejection-sampling engine
src/SmtSampler.zig       # the SMT-backed almost-uniform sampler
src/smt/Encoder.zig      # Ir -> Bitwuzla terms
src/smt/Support.zig      # independent support minimization
src/smt/Hashing.zig      # XOR hashing, ApproxMC, cell enumeration
src/bitwuzla/            # backend binding: enabled.zig / disabled.zig
```

## License

MIT — see [LICENSE](LICENSE).

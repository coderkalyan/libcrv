# libcrv

A constrained random verification library: build a constraint set into a
compact IR, hash or cache it, and draw satisfying assignments from it. Usable
from Zig as a module, and from C through [`include/crv.h`](include/crv.h).

## Requirements

- [Zig](https://ziglang.org/) `0.16.0` or newer.

## Building and testing

```sh
zig build          # static + shared C library and the header into zig-out/
zig build test     # Zig unit tests, C API tests, and the C smoke test
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
  allocation off a garbage count. `ir.serializedSize()` gives the exact length
  without serializing, for callers that want to supply the buffer.

`ir.validate(gpa)` is the pass that makes it safe to walk an IR nobody vetted:
every operand index must exist and refer *backwards* (evaluation is one forward
sweep, so this is what makes the tree acyclic and in evaluation order), every
`extra` payload must lie inside the pool, and operand widths must agree wherever
the evaluator assumes a single width. `deserialize` runs it before returning, so
a blob that survives the checksum and the pass can be consumed without bounds
checks.

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
`in`/`range`) and tracks cumulative `attempts`/`hits` counters. `dist` and the
structural constraints are roadmap: `RejectionSampler.supports` says so, and
`init` rejects an IR that uses one with `error.UnsupportedNode` rather than
giving up partway through a draw.

## The C API

`zig build` installs `libcrv.a`, `libcrv.so` and `crv.h`. The header is
hand-written and declaration-only, so bindgen, cffi and DPI can all parse it.
Nodes, variables and constraints are `uint32_t` indices into the IR that
produced them.

```c
#include <crv.h>

crv_ir ir;
crv_ir_init(&ir);

crv_var x;
crv_node xr, lo, hi, rng, member;
crv_var_add(&ir, /*id=*/1, /*width=*/4, CRV_VAR_RAND, &x); // rand bit [3:0] x;
crv_node_var(&ir, x, &xr);
crv_node_const_u64(&ir, 3, 4, &lo);
crv_node_const_u64(&ir, 7, 4, &hi);
crv_node_range(&ir, lo, hi, &rng);                         // [3:7]
crv_node_in(&ir, xr, &rng, 1, &member);                    // x inside {[3:7]}
crv_constraint_add(&ir, /*id=*/2, 0, &member, 1, NULL);

crv_solver *s;
crv_rejection_sampler_new(&ir, NULL, &s);

uint64_t values[1];
if (crv_solver_next(s, values, 1) == CRV_OK) {
    printf("x = %llu\n", (unsigned long long)values[0]);
}

crv_solver_free(s);
crv_ir_deinit(&ir);
```

A `crv_ir` is a value, not a handle: it is the Zig `Ir` struct seen through a
fixed-size, suitably aligned byte buffer, so C code places it on the stack or
inside its own structures and the library never allocates it. (The library
refuses to compile if `Ir` ever outgrows that buffer, so the size is checked
rather than assumed.) `crv_solver` stays an opaque, library-allocated handle,
because its size depends on the engine behind it and a header constant sized to
the largest one would become an ABI liability the moment a heavier backend
lands.

Every call returns a `crv_status` and writes its result through a trailing
out-parameter (`NULL` to discard); negative statuses are hard errors, and
`CRV_EXHAUSTED` from `crv_solver_next` is the budget running out, not a proof
of unsatisfiability. Operators go through one `crv_node_binary(ir, CRV_OP_ADD,
a, b, &out)` entry point rather than one export per operator, and the `crv_op`
values are mapped to IR tags explicitly, so the internal tag ordering is free
to change without breaking a compiled consumer.

The C layer adds no logic and no state of its own — it is argument checking plus
a call into the Zig core — but it does check what Zig's type system would
otherwise catch: indices are bounds-checked, operand widths must agree
(`CRV_ERR_WIDTH_MISMATCH`; change width with `crv_node_cast`), deserialized
blobs are validated, and a solver refuses an IR containing a node it cannot
evaluate. What it deliberately does not do is defend a C caller from hazards a
Zig caller also has: a solver borrows its IR, and appending to that IR while the
solver lives is undefined for both. No buffer crosses the boundary for the
caller to free: where the library produces bytes, the caller supplies the
storage.

Solutions come back in the same layout Zig sees — little-endian 64-bit words,
`crv_value_words(&ir)` per variable — so there is no marshalling in either
direction; wide literals go in the same way through `crv_node_const_bits`.

## Layout

```
build.zig                # build graph: Zig module, C library, test step
build.zig.zon            # package manifest (name, version, dependencies)
include/crv.h            # the public C header
src/root.zig             # Zig module root — the public API
src/c_api.zig            # C ABI bindings; root of the C library artifact
src/Ir.zig               # the flattened-tree IR
src/Solver.zig           # the swappable solver interface
src/RejectionSampler.zig # the rejection-sampling engine
test/smoke.c             # C smoke test, compiled and run by `zig build test`
```

## License

MIT — see [LICENSE](LICENSE).

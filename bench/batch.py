"""Where do the per-sample solver calls go, and what does consuming more of
each cell cost in distribution quality?"""
from collections import Counter
from distribution import Sampler, Oracle, popcount_k, chi_square

S = popcount_k(18, 3)
N = 120_000
print(f"|S| = {len(S)}   pivot = 72   aim target = 36 solutions/cell\n")
print(f"{'per_cell':>9} {'calls/sample':>13} {'cell size':>10} {'chi2/dof':>9} "
      f"{'z':>7} {'distinct':>9} {'iid exp':>8}")

iid_expected = len(S) * (1 - (1 - 1 / len(S)) ** N)

for k in (1, 4, 8, 16, 36, 72):
    smp = Sampler(Oracle(S, 18), seed=7, exact_limit=64, samples_per_cell=k)
    cells = []
    orig = smp._refill

    def traced(_o=orig, _c=cells, _s=smp):
        before = _s.o.calls
        r = _o()
        if r:
            _c.append(_s.o.calls - before)
        return r

    smp._refill = traced

    counts = Counter()
    for _ in range(N):
        counts[smp.next()] += 1
    chi2, dof, z = chi_square(counts, S, N)
    # A refill costs (cell size + 1) calls: one per model, plus the final unsat.
    avg_cell = sum(cells) / len(cells) - 1 if cells else 0
    print(f"{k:>9} {smp.o.calls/N:>13.2f} {avg_cell:>10.1f} {chi2/dof:>9.4f} "
          f"{z:>+7.2f} {len(counts):>9} {iid_expected:>8.0f}")

print("\nDistinct-value count is the correlation probe: samples inside one")
print("batch are drawn without replacement, so they repeat less often than")
print("independent draws would. The gap widens as per_cell grows.")

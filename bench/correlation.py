"""Does batching actually correlate consecutive samples, and by how much?

Probe: distinct values in each consecutive window of 8. Independent draws from
816 solutions collide inside a window of 8 with probability ~C(8,2)/816 = 3.4%,
so they average slightly under 8 distinct. Samples taken from one cell without
replacement are always distinct, so they average exactly 8.
"""
from distribution import Sampler, Oracle, popcount_k

S = popcount_k(18, 3)
N, W = 200_000, 8
ideal = 8 - 28 / len(S)
print(f"|S| = {len(S)}, windows of {W}, {N} samples")
print(f"iid expectation ~ {ideal:.4f} distinct per window\n")
print(f"{'per_cell':>9} {'distinct/window':>16} {'excess vs iid':>14}")
for k in (1, 8, 36):
    smp = Sampler(Oracle(S, 18), seed=11, exact_limit=64, samples_per_cell=k)
    run = [smp.next() for _ in range(N)]
    tot = sum(len(set(run[i:i + W])) for i in range(0, N - W, W))
    avg = tot / (N // W - 1)
    print(f"{k:>9} {avg:>16.4f} {avg - ideal:>+14.4f}")

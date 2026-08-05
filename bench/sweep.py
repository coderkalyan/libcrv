"""Follow-ups: how many counting rounds are actually needed, and does the
free-bit factorization hold on its marginals."""

import math
from collections import Counter
from distribution import (Sampler, Oracle, popcount_k, delta_for, rounds_for,
                 chi_square)


def count_accuracy(support, nbits, rounds_list, seeds, epsilon=0.8):
    size = len(support)
    bound = 1.0 + epsilon
    print("=== count accuracy vs max_rounds ===")
    print(f"  |S| = {size}, {seeds} seeds each, eps band "
          f"[{1/bound:.3f}, {bound:.3f}]")
    print(f"  {'rounds':>6} {'delta(thy)':>11} {'in band':>10} {'failed':>7} "
          f"{'min':>7} {'max':>7} {'calls':>8}")
    for r in rounds_list:
        ratios, calls, failed = [], [], 0
        for sd in range(seeds):
            try:
                smp = Sampler(Oracle(support, nbits), seed=1000 + sd,
                              exact_limit=64, max_rounds=r)
            except ValueError:
                failed += 1          # Zig equivalent: error.BudgetExceeded
                continue
            ratios.append(smp.count_estimate / size)
            calls.append(smp.compile_calls)
        if not ratios:
            print(f"  {r:>6} {'vacuous':>11} {'-':>10} {failed:>7}")
            continue
        good = sum(1 for x in ratios if 1 / bound <= x <= bound)
        thy = delta_for(r)
        thy_s = f"{thy:.3f}" if thy <= 1 else "vacuous"
        print(f"  {r:>6} {thy_s:>11} {good:>4}/{len(ratios):<5} {failed:>7} "
              f"{min(ratios):>7.3f} {max(ratios):>7.3f} "
              f"{sum(calls)//len(calls):>8}")


def free_bit_marginals(n=400_000):
    """The factorization claim, tested on its two factors separately.

    A per-solution histogram over 45056 solutions needs millions of samples
    before the ratio band means anything (Poisson noise at ~7 samples/solution
    spans 0 to ~3 all on its own). The marginals need far fewer, and they are
    what the argument actually rests on: the support projection uniform over
    its 11 values, and the 12 free bits uniform over their 4096.
    """
    proj = [v for v in range(256) if 10 <= v <= 20]
    smp = Sampler(Oracle(proj, 8), free_bits=12, seed=6)

    low, high = Counter(), Counter()
    for _ in range(n):
        s = smp.next()
        low[s & 0xFF] += 1
        high[s >> 8] += 1

    c1, d1, z1 = chi_square(low, proj, n)
    c2, d2, z2 = chi_square(high, list(range(1 << 12)), n)
    print("\n=== free-bit factorization, tested on marginals ===")
    print(f"  {n} samples; support = low 8 bits (11 values), 12 bits free")
    print(f"  support projection : chi2/dof = {c1/d1:.4f}  z = {z1:+.2f}")
    print(f"  free bits          : chi2/dof = {c2/d2:.4f}  z = {z2:+.2f}")
    lo = min(low[v] * len(proj) / n for v in proj)
    hi = max(low[v] * len(proj) / n for v in proj)
    print(f"  projection p_i*|P| range = [{lo:.4f}, {hi:.4f}]")
    print(f"  solver calls: compile = {smp.compile_calls}, "
          f"per sample = {smp.o.calls/n:.3f}")


if __name__ == "__main__":
    s = popcount_k(18, 3)
    count_accuracy(s, 18, [1, 3, 9, 17, 41, 84], seeds=25)
    print(f"\n  (theory demands {rounds_for(0.1)} rounds for delta=0.1)")
    free_bit_marginals()

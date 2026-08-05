"""Faithful port of SmtSampler's control flow, run against brute-force truth.

The Zig engine cannot be built here (no Zig toolchain, no Bitwuzla), but the
part of it that decides *distribution* is pure algorithm: pivotFor, aimFor,
LogSATSearch, the cell retry loop, the shuffle-and-take batch, and the free-bit
redraw. Those are ported here line for line. The solver is replaced by a brute
-force oracle over an explicitly enumerated solution set, which returns exactly
what an ideal SAT solver would return -- so cell sizes, call counts, and the
resulting sample distribution are all the real ones.

What this validates: uniformity of the sampled stream, the epsilon bound, the
effect of samples_per_cell, the free-bit factorization, and how many solver
calls a compile and a sample cost.

What it cannot validate: wall-clock, which lives entirely in Bitwuzla/CMS.
"""

import math
import random
from collections import Counter

# ---------------------------------------------------------------- parameters
# Ported verbatim from src/smt/Hashing.zig.


def pivot_for(epsilon):
    a = 1.0 + epsilon / (1.0 + epsilon)
    b = 1.0 + 1.0 / epsilon
    return math.ceil(9.84 * a * b * b)


def rounds_for(delta):
    return math.ceil(17.0 * math.log2(3.0 / delta))


def delta_for(rounds):
    return 3.0 / (2.0 ** (rounds / 17.0))


def aim_for(mantissa, exp, target):
    """Hashing.aimFor: floor(log2 count) - floor(log2 target), clamped at 0."""
    if mantissa == 0:
        return 0
    total = exp + mantissa.bit_length() - 1
    want = max(target, 1).bit_length() - 1
    return max(0, total - want)


# ------------------------------------------------------------------- oracle


class Oracle:
    """Stands in for Bitwuzla: an explicit solution set over `nbits` support bits.

    `calls` counts check_sat invocations, matching what Hashing.enumerate does:
    one per model found, plus one final unsat, or exactly `limit` when the
    enumeration is truncated.
    """

    def __init__(self, projections, nbits):
        self.all = list(projections)
        self.nbits = nbits
        self.calls = 0

    def bounded_sat(self, rng, m, density, limit):
        """Hashing.boundedSat: push, add m parity constraints, enumerate, pop."""
        cell = self.all
        added = 0
        while added < m:
            mask = 0
            for b in range(self.nbits):
                if rng.random() < density:
                    mask |= 1 << b
            if mask == 0:
                continue  # empty selection constrains nothing; redraw
            parity = rng.getrandbits(1)
            cell = [s for s in cell if ((s & mask).bit_count() & 1) == parity]
            added += 1

        found = min(len(cell), limit)
        if found >= limit:
            self.calls += limit
            return cell[:limit], "truncated"
        self.calls += found + 1
        return cell[:found], "exhausted"


# ------------------------------------------------------------------ sampler


class Sampler:
    """Port of SmtSampler: measure() then next()."""

    def __init__(self, oracle, free_bits=0, seed=0, epsilon=0.8, delta=0.1,
                 max_rounds=17, samples_per_cell=None, exact_limit=4096,
                 xor_density=0.5, cell_retries=16):
        self.o = oracle
        self.free_bits = free_bits
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.samples_per_cell = (samples_per_cell if samples_per_cell is not None
                                 else max(pivot_for(epsilon) // 2, 1))
        self.xor_density = xor_density
        self.cell_retries = cell_retries
        self.pivot = pivot_for(epsilon)
        self.rounds = min(max_rounds, rounds_for(delta))
        self.exact_limit = exact_limit

        self.batch, self.pos = [], 0
        self.hash_width = 0
        self.enumerated = False
        self.guarantee = None
        self.compile_calls = 0
        self.count_estimate = None
        self._measure()
        self.compile_calls = self.o.calls
        self.o.calls = 0

    def _measure(self):
        base, status = self.o.bounded_sat(self.rng, 0, self.xor_density,
                                          self.exact_limit)
        if status == "exhausted":
            if not base:
                raise ValueError("unsat")
            self.batch = base
            self.enumerated = True
            self.guarantee = "exact"
            return

        estimates, previous = [], 1
        for _ in range(self.rounds):
            found = self._log_sat_search(previous)
            if found is None:
                continue
            previous = found[0]
            estimates.append(found)
        if not estimates:
            raise ValueError("indeterminate")

        estimates.sort(key=lambda e: e[1] * (2 ** e[0]))
        m, cell = estimates[len(estimates) // 2]
        self.count_estimate = cell * (2 ** m)
        self.hash_width = aim_for(cell, m, max(self.pivot // 2, 1))
        self.guarantee = "almost_uniform"

    def _log_sat_search(self, warm):
        limit = self.pivot + 1
        lo, hi = 0, self.o.nbits
        start = max(1, min(warm, max(self.o.nbits, 1)))
        _, status = self.o.bounded_sat(self.rng, start, self.xor_density, limit)
        if status == "truncated":
            lo = start + 1
        else:
            hi = start
        while lo < hi:
            mid = lo + (hi - lo) // 2
            _, status = self.o.bounded_sat(self.rng, mid, self.xor_density, limit)
            if status == "truncated":
                lo = mid + 1
            else:
                hi = mid
        if lo > self.o.nbits:
            return None
        cell, status = self.o.bounded_sat(self.rng, lo, self.xor_density, limit)
        if status != "exhausted":
            return None
        return (lo, len(cell))

    def _refill(self):
        limit = self.pivot + 1
        for _ in range(self.cell_retries + 1):
            cell, status = self.o.bounded_sat(self.rng, self.hash_width,
                                              self.xor_density, limit)
            if status == "truncated":
                self.hash_width += 1
                continue
            if not cell:
                if self.hash_width == 0:
                    return False
                self.hash_width -= 1
                continue
            self.rng.shuffle(cell)
            self.batch = cell[:max(self.samples_per_cell, 1)]
            self.pos = 0
            return True
        return False

    def next(self):
        if self.enumerated:
            base = self.batch[self.rng.randrange(len(self.batch))]
        else:
            if self.pos >= len(self.batch) and not self._refill():
                return None
            base = self.batch[self.pos]
            self.pos += 1
        # Free bits are redrawn every sample, never read back from a model.
        if self.free_bits:
            return (self.rng.getrandbits(self.free_bits) << self.o.nbits) | base
        return base


# ------------------------------------------------------------------ metrics


def chi_square(counts, support, n):
    expected = n / len(support)
    chi2 = sum((counts.get(s, 0) - expected) ** 2 / expected for s in support)
    dof = len(support) - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    return chi2, dof, z


def report(name, sampler, support, n_samples, note=""):
    counts = Counter()
    drawn = 0
    for _ in range(n_samples):
        s = sampler.next()
        if s is None:
            break
        counts[s] += 1
        drawn += 1

    size = len(support)
    chi2, dof, z = chi_square(counts, support, drawn)
    ratios = [counts.get(s, 0) * size / drawn for s in support]
    lo, hi = min(ratios), max(ratios)
    bound = 1.0 + sampler.epsilon
    inside = sum(1 for r in ratios if 1.0 / bound <= r <= bound)

    print(f"\n=== {name} ===")
    if note:
        print(f"  {note}")
    print(f"  |S| = {size}, samples = {drawn}, guarantee = {sampler.guarantee}")
    if sampler.count_estimate is not None:
        err = sampler.count_estimate / size
        print(f"  count estimate = {sampler.count_estimate} "
              f"({err:.3f}x true, eps bound {1/bound:.3f}-{bound:.3f}x)")
    print(f"  solver calls: compile = {sampler.compile_calls}, "
          f"per sample = {sampler.o.calls / max(drawn,1):.2f}")
    print(f"  chi2/dof = {chi2/dof:.4f}   z = {z:+.2f}")
    print(f"  p_i*|S| range = [{lo:.3f}, {hi:.3f}]   "
          f"within 1+eps: {inside}/{size} ({100*inside/size:.1f}%)")
    return z


# ---------------------------------------------------------------- instances


def popcount_k(nbits, k):
    return [x for x in range(1 << nbits) if x.bit_count() == k]


def quadratic_residue(nbits, mod, target):
    return [x for x in range(1 << nbits) if (x * x) % mod == target]


def modular(nbits, mod, target):
    return [x for x in range(1 << nbits) if x % mod == target]


def product_pairs(nbits, target):
    out = []
    for a in range(1, 1 << nbits):
        if target % a == 0:
            b = target // a
            if b < (1 << nbits):
                out.append(a | (b << nbits))
    return out


SAMPLES = 120_000


if __name__ == "__main__":
    print("SmtSampler distribution simulation")
    print(f"pivot(eps=0.8) = {pivot_for(0.8)}, "
          f"rounds(delta=0.1) = {rounds_for(0.1)}, "
          f"delta at 17 rounds = {delta_for(17):.3f}")

    # 1. Control: the exact path. Should be indistinguishable from uniform.
    s = popcount_k(18, 3)
    report("popcount(x)==3 over 18 bits -- EXACT path",
           Sampler(Oracle(s, 18), seed=1, exact_limit=4096), s, SAMPLES,
           "enumerated in full; sampling is an index draw")

    # 2. Same instance forced onto the hashing path, so the histogram is
    #    directly comparable to the control above.
    report("popcount(x)==3 over 18 bits -- HASHED path",
           Sampler(Oracle(s, 18), seed=2, exact_limit=64), s, SAMPLES,
           "exact_limit lowered to force hash-cell sampling")

    # 3. Independent draws: one sample per cell.
    report("popcount -- HASHED, samples_per_cell=1",
           Sampler(Oracle(s, 18), seed=3, exact_limit=64, samples_per_cell=1),
           s, SAMPLES, "fully independent draws, ~8x the solver calls")

    # 4. Sparse arithmetic: a quadratic residue, so solutions are scattered
    #    with no structure a sampler could accidentally exploit.
    target = pow(123, 2, 1021)
    q = quadratic_residue(20, 1021, target)
    report(f"x*x mod 1021 == {target} over 20 bits -- HASHED",
           Sampler(Oracle(q, 20), seed=4, exact_limit=64), q, SAMPLES,
           f"density {len(q)/(1<<20):.5f}; rejection needs ~{(1<<20)//len(q)} draws/hit")

    # 5. Coupled multiplier: solutions are wildly unevenly placed.
    p = product_pairs(11, 1440)
    report("a*b == 1440, 11-bit a and b -- EXACT",
           Sampler(Oracle(p, 22), seed=5), p, SAMPLES,
           "irregular solution placement across a 22-bit space")

    # 6. Free bits: only the low byte is constrained, 12 bits are unobserved.
    #    The support is 8 bits wide; the other 12 are redrawn per sample.
    proj = [v for v in range(256) if 10 <= v <= 20]
    full = [(h << 8) | v for v in proj for h in range(1 << 12)]
    report("low byte in [10,20], 20-bit x -- FREE BITS",
           Sampler(Oracle(proj, 8), free_bits=12, seed=6), full, 300_000,
           "support is 8 bits; 12 free bits drawn from the RNG each sample")

    # 7. A larger set on the hashing path, to see the count estimate and cost.
    m7 = modular(20, 7, 3)
    smp = Sampler(Oracle(m7, 20), seed=7)
    print(f"\n=== x mod 7 == 3 over 20 bits -- HASHED (cost only) ===")
    print(f"  |S| = {len(m7)}, guarantee = {smp.guarantee}")
    print(f"  count estimate = {smp.count_estimate} "
          f"({smp.count_estimate/len(m7):.3f}x true)")
    print(f"  solver calls: compile = {smp.compile_calls}")
    for _ in range(2000):
        smp.next()
    print(f"  solver calls per sample = {smp.o.calls/2000:.2f}")


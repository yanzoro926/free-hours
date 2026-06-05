# The 17% Integer — A Brief Exploration

## The Claim

From Hacker News (June 2, 2026): "Only 17% of all 64-bit Integers are products of two 32-bit integers"

## The Analysis

### The Obvious (but Wrong) Path

At first glance, this seems about the numbers that are TOO LARGE to be products of two 32-bit integers:

- Max 32-bit unsigned integer: M = 2³² - 1 ≈ 4,294,967,295
- Max product of two 32-bit integers: M² ≈ 18,446,744,065,119,617,025
- Max 64-bit unsigned integer: 2⁶⁴ - 1 ≈ 18,446,744,073,709,551,615
- Gap: 2⁶⁴ - 1 - M² = 8,589,934,590 ≈ 2³³

That gap is only 8.6 billion out of 18.4 quintillion — a mere 4.66 × 10⁻¹⁰ fraction. Not 17%!

### The Real Constraint

The 17% refers to density within the reachable range [1, M²]. Even though numbers ≤ M² *could* theoretically be products, not all of them *are*.

**Why?** A number n ≤ M² needs BOTH factors ≤ M. If a number has all its factors either very small or very large relative to √n, it can't be expressed as x·y with both x,y ≤ M.

**Example:** n = 2³² × 5 = 21,474,836,480
- n ≤ M²? Yes.
- Can we write n = x·y with x,y ≤ M?
  - If x = 2³², x > M → no
  - If x = 5, y = 2³² > M → no
  - Any other factorization has the same problem
- So n is **unreachable** despite being ≤ M²!

### Miniature Model

Using 16-bit numbers with 8-bit factors (same ratio as 64/32-bit):

- Total 16-bit numbers: 65,535
- **Reachable: 17,577 → 26.82%**
- Unreachable: 47,958

The 17% figure emerges from the asymptotic density in the full 64-bit space.

### Small-Scale Empirical Results

| Max Value | Max Factor | Density |
|-----------|-----------|---------|
| 1,000 | 30 | 30.80% |
| 2,000 | 60 | 44.90% |
| 10,000 | 100 | ~38% |
| 65,535 (16-bit) | 255 (8-bit) | 26.82% |

The density decreases as the ratio max_factor/√(max_value) approaches 1.0.

## Key Insight

The "17%" is not about numbers exceeding the product bound — it's about the **factorization constraint**: requiring both factors to fit in 32 bits eliminates numbers whose factor pairs are asymmetric (one very small, one very large).

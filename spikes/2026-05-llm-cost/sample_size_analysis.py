"""Sample-size calculations for confident decade-evolution detection.

For each of the four modes, given the methods-aware schema and the 26 fields
× 10 years cell structure, calculate what sample sizes are needed to detect
realistic effect sizes.

Effect sizes of interest:
  - 5pp annual change in primary/secondary mix (e.g. 50% → 55%): the
    AI-augmented productivity signal we're trying to measure.
  - 5pp change at low base rates (e.g. 10% → 15%): emergence of primary
    methods/software output in fields where it was previously rare.
  - 10pp change at low base rates (e.g. 5% → 15%): a strong shift to detect.
"""
import math


def n_for_two_proportion_test(p1: float, p2: float, alpha=0.05, power=0.8) -> int:
    """Approximate sample size per group for two-proportion z-test."""
    z_alpha = 1.96  # two-sided 0.05
    z_beta = 0.84   # power 0.8
    p_bar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p1 - p2) ** 2
    return math.ceil(numerator / denominator)


def main():
    scenarios = [
        ("Mid-range, 5pp shift (50% → 55%)", 0.50, 0.55),
        ("Mid-range, 3pp shift (50% → 53%)", 0.50, 0.53),
        ("Asymmetric, 5pp shift (20% → 25%)", 0.20, 0.25),
        ("Asymmetric, 5pp shift (10% → 15%)", 0.10, 0.15),
        ("Low-base, 5pp shift (5% → 10%)", 0.05, 0.10),
        ("Low-base, 10pp shift (5% → 15%)", 0.05, 0.15),
        ("Very-low-base, 3pp shift (2% → 5%)", 0.02, 0.05),
    ]

    print("Sample size per CELL (per group), 80% power, p<0.05 two-sided:")
    print(f"{'Scenario':<48} {'n/group':>10}")
    print("-" * 60)
    for label, p1, p2 in scenarios:
        n = n_for_two_proportion_test(p1, p2)
        print(f"{label:<48} {n:>10,}")

    print()
    print("Interpretation for 26 fields × 10 years = 260 cells:")
    print()
    for total_budget in (100_000, 200_000, 300_000, 500_000):
        per_cell = total_budget / 260
        print(f"  {total_budget:>7,} total → {per_cell:>5,.0f} per cell")
    print()
    print("If we want EVERY cell to support 5pp detection at 50/50 baseline:")
    print(f"  Per-cell requirement: ~{n_for_two_proportion_test(0.50, 0.55):,}")
    print(f"  Floor only:           260 × that = {260 * n_for_two_proportion_test(0.50, 0.55):,}")
    print()
    print("If we want EVERY cell to support 5pp detection at 10%/15% baseline")
    print("(detecting emergence of primary methods in low-output fields):")
    print(f"  Per-cell requirement: ~{n_for_two_proportion_test(0.10, 0.15):,}")
    print(f"  Floor only:           260 × that = {260 * n_for_two_proportion_test(0.10, 0.15):,}")
    print()
    print("If we want ONLY large-corpus fields to support 5pp detection")
    print("(accept lower power in Veterinary, Chemical Engineering, etc.):")
    print(f"  Floor 500/cell × 260 = {500 * 260:,}")
    print(f"  Plus proportional excess to 300k: budget supports")
    print(f"  ~5pp detection in major fields, ~10pp in smallest")


if __name__ == "__main__":
    main()

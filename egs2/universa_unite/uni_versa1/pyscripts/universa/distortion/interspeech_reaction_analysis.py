import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# ==========================
# Configuration
# ==========================

CSV_PATH = "master_reaction_analysis_all_metrics.csv"
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42
OUTPUT_DIR = "metric_analysis_outputs2"

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Load Data
# ==========================

df = pd.read_csv(CSV_PATH)
prefixes = sorted(df["prefix_seconds"].dropna().unique())

print("\n==============================")
print(" FULL MULTI-METRIC ANALYSIS ")
print("==============================\n")

delta_cols = [
    c for c in df.columns
    if c.startswith("delta_anchor_vs_clean_")
    and not c.endswith("_full")
]

metric_names = [c.replace("delta_anchor_vs_clean_", "") for c in delta_cols]

print(f"Detected {len(metric_names)} metrics:")
for m in metric_names:
    print("  -", m)

# ==========================
# Bootstrap Function
# ==========================

def bootstrap_ci(values, n_boot=1000):
    if len(values) == 0:
        return (np.nan, np.nan)

    boot_means = []
    n = len(values)

    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    return np.percentile(boot_means, [2.5, 97.5])

# ==========================
# Main Analysis
# ==========================

all_metric_results = []

for metric in metric_names:

    print(f"\n----- Metric: {metric} -----")

    anchor_col = f"delta_anchor_vs_clean_{metric}"
    arecho_col = f"delta_arecho_vs_clean_{metric}"

    for p in prefixes:

        subset = df[df["prefix_seconds"] == p]
        paired = subset[[anchor_col, arecho_col]].dropna()

        if len(paired) < 5:
            continue

        delta_anchor = paired[anchor_col].values
        delta_arecho = paired[arecho_col].values

        # ======================
        # Robustness (Magnitude)
        # ======================

        mad_anchor = np.mean(np.abs(delta_anchor))
        mad_arecho = np.mean(np.abs(delta_arecho))

        mean_anchor = np.mean(delta_anchor)
        mean_arecho = np.mean(delta_arecho)

        std_anchor = np.std(delta_anchor, ddof=1)
        std_arecho = np.std(delta_arecho, ddof=1)

        # Safe t-test
        if np.allclose(delta_anchor, delta_arecho):
            p_value = 1.0
        else:
            _, p_value = stats.ttest_rel(delta_anchor, delta_arecho)

        # Cohen's d
        diff = delta_anchor - delta_arecho
        std_diff = np.std(diff, ddof=1)
        cohen_d = 0.0 if std_diff == 0 else np.mean(diff) / std_diff

        ci_anchor = bootstrap_ci(np.abs(delta_anchor), BOOTSTRAP_SAMPLES)
        ci_arecho = bootstrap_ci(np.abs(delta_arecho), BOOTSTRAP_SAMPLES)

        # ======================
        # Directional Analysis
        # ======================

        anchor_over_rate = np.mean(delta_anchor > 0)
        anchor_under_rate = np.mean(delta_anchor < 0)

        arecho_over_rate = np.mean(delta_arecho > 0)
        arecho_under_rate = np.mean(delta_arecho < 0)

        anchor_bias = np.mean(delta_anchor)
        arecho_bias = np.mean(delta_arecho)
        # ======================
        # Metric direction
        # ======================

        LOWER_IS_BETTER = {
            "asr_match_error_rate": True,
        }

        def is_lower_better(metric):
            return LOWER_IS_BETTER.get(metric, False)

        lower_better = is_lower_better(metric)

        # Convert to quality-normalized bias
        if lower_better:
            anchor_quality_bias = -anchor_bias
            arecho_quality_bias = -arecho_bias
        else:
            anchor_quality_bias = anchor_bias
            arecho_quality_bias = arecho_bias

        anchor_optimistic_rate = np.mean(
            (-delta_anchor if lower_better else delta_anchor) > 0
        )

        arecho_optimistic_rate = np.mean(
            (-delta_arecho if lower_better else delta_arecho) > 0
        )

        anchor_bias_ci = bootstrap_ci(delta_anchor, BOOTSTRAP_SAMPLES)
        arecho_bias_ci = bootstrap_ci(delta_arecho, BOOTSTRAP_SAMPLES)

        skew_anchor = anchor_over_rate - anchor_under_rate
        skew_arecho = arecho_over_rate - arecho_under_rate

        print(
            f"Prefix {p}s | N={len(paired)} | "
            f"MAD A={mad_anchor:.3f}, R={mad_arecho:.3f} | "
            f"Bias A={anchor_bias:.3f}, R={arecho_bias:.3f} | "
            f"p={p_value:.4g}, d={cohen_d:.3f}"
        )

        all_metric_results.append({
            "metric": metric,
            "prefix": p,
            "N": len(paired),

            # Magnitude
            "MAD_anchor": mad_anchor,
            "MAD_arecho": mad_arecho,
            "MAD_anchor_CI_low": ci_anchor[0],
            "MAD_anchor_CI_high": ci_anchor[1],
            "MAD_arecho_CI_low": ci_arecho[0],
            "MAD_arecho_CI_high": ci_arecho[1],

            # Signed stats
            "mean_anchor": mean_anchor,
            "mean_arecho": mean_arecho,
            "std_anchor": std_anchor,
            "std_arecho": std_arecho,

            # Directional
            "anchor_over_rate": anchor_over_rate,
            "anchor_under_rate": anchor_under_rate,
            "arecho_over_rate": arecho_over_rate,
            "arecho_under_rate": arecho_under_rate,
            "anchor_bias": anchor_bias,
            "arecho_bias": arecho_bias,
            "anchor_bias_CI_low": anchor_bias_ci[0],
            "anchor_bias_CI_high": anchor_bias_ci[1],
            "arecho_bias_CI_low": arecho_bias_ci[0],
            "arecho_bias_CI_high": arecho_bias_ci[1],
            "skew_anchor": skew_anchor,
            "skew_arecho": skew_arecho,

            # Stats test
            "p_value": p_value,
            "cohen_d": cohen_d,
        })

multi_df = pd.DataFrame(all_metric_results)
multi_df.to_csv(os.path.join(OUTPUT_DIR, "multi_metric_summary.csv"), index=False)

print("\nSaved summary table.")

# ==========================
# Reaction Curves (Magnitude)
# ==========================

for metric in metric_names:

    sub = multi_df[multi_df["metric"] == metric]
    if len(sub) == 0:
        continue

    sub = sub.sort_values("prefix")
    x = sub["prefix"].values

    plt.figure(figsize=(6.2, 4.2))

    plt.plot(x, sub["MAD_anchor"], marker="o", linewidth=2, label="ANCHOR")
    plt.fill_between(x, sub["MAD_anchor_CI_low"], sub["MAD_anchor_CI_high"], alpha=0.15)

    plt.plot(x, sub["MAD_arecho"], marker="s", linestyle="--", linewidth=2, label="ARECHO")
    plt.fill_between(x, sub["MAD_arecho_CI_low"], sub["MAD_arecho_CI_high"], alpha=0.15)

    plt.xlabel("Prefix Length (s)")
    plt.ylabel("Mean Absolute Deviation |Δ|")
    plt.title(f"Robustness Curve — {metric}")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_reaction_curve.pdf"))
    plt.close()

# ==========================
# GLOBAL OPTIMISM SUMMARY
# ==========================

print("\n==============================")
print(" GLOBAL OPTIMISM SUMMARY ")
print("==============================\n")

LOWER_IS_BETTER = {
    "asr_match_error_rate": True,
}

def is_lower_better(metric):
    return LOWER_IS_BETTER.get(metric, False)

for metric in metric_names:

    sub = multi_df[multi_df["metric"] == metric]
    if len(sub) == 0:
        continue

    lower_better = is_lower_better(metric)

    # Compute quality-normalized bias on the fly
    if lower_better:
        anchor_quality_bias = -sub["mean_anchor"]
        arecho_quality_bias = -sub["mean_arecho"]
    else:
        anchor_quality_bias = sub["mean_anchor"]
        arecho_quality_bias = sub["mean_arecho"]

    mean_anchor_bias = np.mean(anchor_quality_bias)
    mean_arecho_bias = np.mean(arecho_quality_bias)

    anchor_more_optimistic = np.mean(
        anchor_quality_bias > arecho_quality_bias
    )

    anchor_optimistic_prefix_ratio = np.mean(
        anchor_quality_bias > 0
    )

    print(f"Metric: {metric}")
    print(f"  Mean Quality Bias — Anchor: {mean_anchor_bias:.3f}")
    print(f"  Mean Quality Bias — ARECHO: {mean_arecho_bias:.3f}")
    print(f"  Anchor more optimistic in {anchor_more_optimistic*100:.1f}% prefixes")
    print(f"  Anchor optimistic in {anchor_optimistic_prefix_ratio*100:.1f}% prefixes")
    print()

# ==========================
# Reaction Curves (Quality-Aware Bias)
# ==========================

# True  -> lower value = better quality
# False -> higher value = better quality
LOWER_IS_BETTER = {
    "asr_match_error_rate": True,
}

def is_lower_better(metric):
    return LOWER_IS_BETTER.get(metric, False)

for metric in metric_names:

    sub = multi_df[multi_df["metric"] == metric]
    if len(sub) == 0:
        continue

    sub = sub.sort_values("prefix")
    x = sub["prefix"].values

    lower_better = is_lower_better(metric)

    # Flip sign so positive ALWAYS = optimistic (overestimates quality)
    if lower_better:
        anchor_quality_bias = -sub["anchor_bias"]
        arecho_quality_bias = -sub["arecho_bias"]
        direction_icon = "↓ (Lower Better)"
    else:
        anchor_quality_bias = sub["anchor_bias"]
        arecho_quality_bias = sub["arecho_bias"]
        direction_icon = "↑ (Higher Better)"

    plt.figure(figsize=(6.2, 4.2))

    plt.plot(x, anchor_quality_bias, marker="o", linewidth=2, label="ANCHOR")
    plt.plot(x, arecho_quality_bias, marker="s", linestyle="--", linewidth=2, label="ARECHO")

    plt.axhline(0, linestyle=":", linewidth=1)

    plt.xlabel("Prefix Length (s)")
    plt.ylabel("Quality Bias (Optimistic + / Pessimistic -)")
    plt.title(f"Directional Quality Bias — {metric} {direction_icon}")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_quality_bias_curve.pdf"))
    plt.close()
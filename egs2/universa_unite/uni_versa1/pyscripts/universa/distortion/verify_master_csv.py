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
OUTPUT_DIR = "metric_analysis_outputs"

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

# Automatically detect all delta metrics (non-full only)
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

        mad_anchor = np.mean(np.abs(delta_anchor))
        mad_arecho = np.mean(np.abs(delta_arecho))

        mean_anchor = np.mean(delta_anchor)
        mean_arecho = np.mean(delta_arecho)

        std_anchor = np.std(delta_anchor, ddof=1)
        std_arecho = np.std(delta_arecho, ddof=1)

        t_stat, p_value = stats.ttest_rel(delta_anchor, delta_arecho)

        diff = delta_anchor - delta_arecho
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)

        ci_anchor = bootstrap_ci(np.abs(delta_anchor), BOOTSTRAP_SAMPLES)
        ci_arecho = bootstrap_ci(np.abs(delta_arecho), BOOTSTRAP_SAMPLES)

        print(
            f"Prefix {p}s | "
            f"N={len(paired)} | "
            f"MAD A={mad_anchor:.3f}, R={mad_arecho:.3f} | "
            f"Mean A={mean_anchor:.3f}, R={mean_arecho:.3f} | "
            f"p={p_value:.4g}, d={cohen_d:.3f}"
        )

        all_metric_results.append({
            "metric": metric,
            "prefix": p,
            "N": len(paired),
            "MAD_anchor": mad_anchor,
            "MAD_arecho": mad_arecho,
            "MAD_anchor_CI_low": ci_anchor[0],
            "MAD_anchor_CI_high": ci_anchor[1],
            "MAD_arecho_CI_low": ci_arecho[0],
            "MAD_arecho_CI_high": ci_arecho[1],
            "mean_anchor": mean_anchor,
            "mean_arecho": mean_arecho,
            "std_anchor": std_anchor,
            "std_arecho": std_arecho,
            "p_value": p_value,
            "cohen_d": cohen_d
        })

multi_df = pd.DataFrame(all_metric_results)
multi_df.to_csv(os.path.join(OUTPUT_DIR, "multi_metric_summary.csv"), index=False)

print("\nSaved summary table to multi_metric_summary.csv")

# ==========================
# Pattern Detection
# ==========================

print("\n==============================")
print(" INTERESTING PATTERN SUMMARY ")
print("==============================\n")

for metric in metric_names:

    sub = multi_df[multi_df["metric"] == metric]

    if len(sub) == 0:
        continue

    arecho_better = np.mean(sub["MAD_arecho"] < sub["MAD_anchor"])
    anchor_better = np.mean(sub["MAD_anchor"] < sub["MAD_arecho"])
    significant_ratio = np.mean(sub["p_value"] < 0.05)

    mean_direction_bias = np.mean(sub["mean_anchor"] - sub["mean_arecho"])

    print(f"Metric: {metric}")
    print(f"  ARECHO more robust in {arecho_better*100:.1f}% prefixes")
    print(f"  ANCHOR more robust in {anchor_better*100:.1f}% prefixes")
    print(f"  Significant differences in {significant_ratio*100:.1f}% prefixes")
    print(f"  Directional bias (Anchor - ARECHO): {mean_direction_bias:.3f}")

    if arecho_better > 0.75:
        print("  🔵 Strong ARECHO robustness dominance")

    if anchor_better > 0.75:
        print("  🔴 Strong ANCHOR robustness dominance")

    if abs(mean_direction_bias) > 0.3:
        print("  ⭐ Strong systematic directional bias")

    print()

# ==========================
# Generate Reaction Curves for Each Metric
# ==========================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

for metric in metric_names:

    sub = multi_df[multi_df["metric"] == metric]
    if len(sub) == 0:
        continue

    x = np.array(sorted(sub["prefix"].unique()))

    anchor_means = []
    anchor_low = []
    anchor_high = []

    arecho_means = []
    arecho_low = []
    arecho_high = []

    for p in x:
        row = sub[sub["prefix"] == p]

        anchor_means.append(row["MAD_anchor"].values[0])
        anchor_low.append(row["MAD_anchor_CI_low"].values[0])
        anchor_high.append(row["MAD_anchor_CI_high"].values[0])

        arecho_means.append(row["MAD_arecho"].values[0])
        arecho_low.append(row["MAD_arecho_CI_low"].values[0])
        arecho_high.append(row["MAD_arecho_CI_high"].values[0])

    anchor_means = np.array(anchor_means)
    anchor_low = np.array(anchor_low)
    anchor_high = np.array(anchor_high)

    arecho_means = np.array(arecho_means)
    arecho_low = np.array(arecho_low)
    arecho_high = np.array(arecho_high)

    plt.figure(figsize=(6.2, 4.2))

    plt.plot(x, anchor_means, marker="o", linewidth=2, label="ANCHOR")
    plt.fill_between(x, anchor_low, anchor_high, alpha=0.15)

    plt.plot(x, arecho_means, marker="s", linestyle="--", linewidth=2, label="ARECHO")
    plt.fill_between(x, arecho_low, arecho_high, alpha=0.15)

    plt.xlabel("Prefix Length (s)")
    plt.ylabel("Mean Absolute Deviation |Δ|")
    plt.title(f"Robustness Curve — {metric}")

    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_reaction_curve.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_reaction_curve.png"), dpi=300)
    plt.close()

print("\nSaved all reaction curves to:", OUTPUT_DIR)

print("\n==============================")
print(" ANALYSIS COMPLETE ")
print("==============================")
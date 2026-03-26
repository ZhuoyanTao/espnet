import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# ==========================
# CONFIG
# ==========================

CSV_PATH = "master_reaction_analysis.csv"
BOOTSTRAP_SAMPLES = 1000
np.random.seed(42)

# ==========================
# LOAD
# ==========================

df = pd.read_csv(CSV_PATH)
prefixes = sorted(df["prefix_seconds"].unique())

# ------------------------------------------------
# NOTE:
# This assumes your CSV already contains:
#   clean_label_dns                (full clean MOS)
#   anchor_dns                     (model predicted full MOS)
#   arecho_dns                     (baseline predicted full MOS)
#
# If you also include chunk metrics, you must have:
#   chunk_clean_dns
#   chunk_distorted_dns
# ------------------------------------------------

# ==========================
# UTILITY FUNCTIONS
# ==========================

def bootstrap_ci(values, n_boot=1000):
    boot_means = []
    n = len(values)
    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    return np.percentile(boot_means, [2.5, 97.5])

# ==========================
# GLOBAL ROBUSTNESS ANALYSIS
# ==========================

print("\n=== GLOBAL ROBUSTNESS ANALYSIS ===\n")

results = []

for p in prefixes:
    subset = df[df["prefix_seconds"] == p]

    delta_anchor = subset["delta_anchor_vs_clean"].values
    delta_arecho = subset["delta_arecho_vs_clean"].values

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

    results.append({
        "prefix": p,
        "N": len(subset),
        "MAD_anchor": mad_anchor,
        "MAD_arecho": mad_arecho,
        "p_value": p_value,
        "cohen_d": cohen_d,
        "mean_anchor": mean_anchor,
        "mean_arecho": mean_arecho,
        "MAD_anchor_CI_low": ci_anchor[0],
        "MAD_anchor_CI_high": ci_anchor[1],
        "MAD_arecho_CI_low": ci_arecho[0],
        "MAD_arecho_CI_high": ci_arecho[1],
    })

results_df = pd.DataFrame(results)
print(results_df.round(4))

# ==========================
# CHUNK DEGRADATION ANALYSIS
# ==========================

if "chunk_clean_dns" in df.columns:

    print("\n=== CHUNK DEGRADATION ANALYSIS ===\n")

    chunk_results = []

    for p in prefixes:
        subset = df[df["prefix_seconds"] == p]

        chunk_delta = subset["chunk_distorted_dns"] - subset["chunk_clean_dns"]

        mad_chunk = np.mean(np.abs(chunk_delta))
        mean_chunk = np.mean(chunk_delta)

        ci_chunk = bootstrap_ci(np.abs(chunk_delta), BOOTSTRAP_SAMPLES)

        chunk_results.append({
            "prefix": p,
            "N": len(subset),
            "MAD_chunk": mad_chunk,
            "mean_chunk": mean_chunk,
            "MAD_chunk_CI_low": ci_chunk[0],
            "MAD_chunk_CI_high": ci_chunk[1],
        })

    chunk_df = pd.DataFrame(chunk_results)
    print(chunk_df.round(4))

# ==========================
# MODEL–CHUNK ALIGNMENT
# ==========================

if "chunk_clean_dns" in df.columns:

    print("\n=== MODEL–CHUNK ALIGNMENT (Correlation) ===\n")

    for p in prefixes:
        subset = df[df["prefix_seconds"] == p]

        chunk_delta = subset["chunk_distorted_dns"] - subset["chunk_clean_dns"]
        model_delta = subset["delta_anchor_vs_clean"]

        r, p_corr = stats.pearsonr(chunk_delta, model_delta)

        print(f"Prefix {p}s: r = {r:.3f}, p = {p_corr:.4f}")

# ==========================
# PLOTS
# ==========================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})

# ---- Global robustness curve ----

plt.figure(figsize=(6.2, 4.2))

x = np.array(prefixes)

anchor_means = results_df["MAD_anchor"].values
arecho_means = results_df["MAD_arecho"].values

anchor_low = results_df["MAD_anchor_CI_low"].values
anchor_high = results_df["MAD_anchor_CI_high"].values

arecho_low = results_df["MAD_arecho_CI_low"].values
arecho_high = results_df["MAD_arecho_CI_high"].values

plt.plot(x, anchor_means, marker="o", linewidth=2, label="ANCHOR")
plt.fill_between(x, anchor_low, anchor_high, alpha=0.15)

plt.plot(x, arecho_means, marker="s", linestyle="--", linewidth=2, label="ARECHO")
plt.fill_between(x, arecho_low, arecho_high, alpha=0.15)

plt.xlabel("Prefix Length (s)")
plt.ylabel("Mean Absolute Deviation $|\\Delta|$")
plt.xticks(x)
plt.ylim(bottom=0)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("global_robustness_curve.pdf")
plt.show()

# ---- Optional chunk degradation curve ----

if "chunk_clean_dns" in df.columns:

    plt.figure(figsize=(6.2, 4.2))

    chunk_means = chunk_df["MAD_chunk"].values
    chunk_low = chunk_df["MAD_chunk_CI_low"].values
    chunk_high = chunk_df["MAD_chunk_CI_high"].values

    plt.plot(x, chunk_means, marker="o", linewidth=2, label="Chunk Degradation")
    plt.fill_between(x, chunk_low, chunk_high, alpha=0.15)

    plt.xlabel("Prefix Length (s)")
    plt.ylabel("Chunk DNS Degradation")
    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("chunk_degradation_curve.pdf")
    plt.show()

print("\nAll analysis complete.")


# ==========================
# MULTI-PAGE PDF REPORT
# ==========================

with PdfPages("full_analysis_report.pdf") as pdf:

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
    })

    x = np.array(prefixes)

    anchor_means = results_df["MAD_anchor"].values
    arecho_means = results_df["MAD_arecho"].values

    anchor_low = results_df["MAD_anchor_CI_low"].values
    anchor_high = results_df["MAD_anchor_CI_high"].values

    arecho_low = results_df["MAD_arecho_CI_low"].values
    arecho_high = results_df["MAD_arecho_CI_high"].values

    # ---- Page 1: Global Robustness ----
    plt.figure(figsize=(6.2, 4.2))

    plt.plot(x, anchor_means, marker="o", linewidth=2, label="ANCHOR")
    plt.fill_between(x, anchor_low, anchor_high, alpha=0.15)

    plt.plot(x, arecho_means, marker="s", linestyle="--", linewidth=2, label="ARECHO")
    plt.fill_between(x, arecho_low, arecho_high, alpha=0.15)

    plt.xlabel("Prefix Length (s)")
    plt.ylabel("Mean Absolute Deviation $|\\Delta|$")
    plt.title("Global Robustness vs Prefix Length")
    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.legend(frameon=False)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # ---- Page 2: Chunk Degradation (if available) ----
    if "chunk_clean_dns" in df.columns:

        plt.figure(figsize=(6.2, 4.2))

        chunk_means = chunk_df["MAD_chunk"].values
        chunk_low = chunk_df["MAD_chunk_CI_low"].values
        chunk_high = chunk_df["MAD_chunk_CI_high"].values

        plt.plot(x, chunk_means, marker="o", linewidth=2, label="Chunk Degradation")
        plt.fill_between(x, chunk_low, chunk_high, alpha=0.15)

        plt.xlabel("Prefix Length (s)")
        plt.ylabel("Chunk DNS Degradation")
        plt.title("True Chunk-Level Degradation")
        plt.xticks(x)
        plt.ylim(bottom=0)
        plt.legend(frameon=False)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # ---- Page 3: Model–Chunk Alignment at 2s ----
        subset = df[df["prefix_seconds"] == 2]

        chunk_delta = subset["chunk_distorted_dns"] - subset["chunk_clean_dns"]
        model_delta = subset["delta_anchor_vs_clean"]

        r, p_corr = stats.pearsonr(chunk_delta, model_delta)

        plt.figure(figsize=(5.5, 4.5))
        plt.scatter(chunk_delta, model_delta, alpha=0.6)
        plt.xlabel("Chunk DNS Degradation")
        plt.ylabel("Model Δ (Anchor)")
        plt.title(f"Model vs True Degradation (2s)\n r={r:.3f}, p={p_corr:.4f}")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print("\n✅ Saved full_analysis_report.pdf")
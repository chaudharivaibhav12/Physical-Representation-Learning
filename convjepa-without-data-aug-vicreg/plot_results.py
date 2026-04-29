"""
Generate report figures comparing Conv-JEPA Baseline vs Conv-JEPA w/o Augmentation.
Run: python plot_results.py
Outputs: figures/ directory with PNG files.

Data sources:
  Baseline epoch progression → probe_epoch18/19/20/21/22.json
  Baseline summary           → probe_baseline_final.json      (epoch_20.pt)
  No-aug summary             → final_main_no_aug.json         (best.pt)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

os.makedirs("figures", exist_ok=True)

RANDOM_BASELINE = 1.0

# ── Source: probe_epoch{N}.json ─────────────────────────────────────────────
EPOCHS        = [18,     19,     20,     21,     22    ]
LP_VAL_NORM   = [0.2381, 0.2566, 0.1764, 0.3112, 0.3796]
LP_TEST_NORM  = [0.2577, 0.2924, 0.1887, 0.2556, 0.5384]
KNN_VAL_NORM  = [0.5161, 0.5212, 0.3719, 0.3792, 0.3921]
KNN_TEST_NORM = [0.5754, 0.6980, 0.4188, 0.3925, 0.4268]

# ── Source: probe_baseline_final.json (Conv-JEPA Baseline, epoch_20.pt) ─────
BASELINE = {
    "linear_probe": {
        "val":  {"mse_normalized_avg": 0.17642, "mse_normalized_alpha": 0.14479, "mse_normalized_zeta": 0.20805},
        "test": {"mse_normalized_avg": 0.18866, "mse_normalized_alpha": 0.18322, "mse_normalized_zeta": 0.19410},
    },
    "knn": {
        "val":  {"mse_normalized_avg": 0.37191, "mse_normalized_alpha": 0.29578, "mse_normalized_zeta": 0.44803, "best_k": 10},
        "test": {"mse_normalized_avg": 0.41876, "mse_normalized_alpha": 0.43672, "mse_normalized_zeta": 0.40080},
    },
}

# ── Source: final_main_no_aug.json (Conv-JEPA w/o Aug, best.pt) ─────────────
NO_AUG = {
    "linear_probe": {
        "val":  {"mse_normalized_avg": 0.16773, "mse_normalized_alpha": 0.02764, "mse_normalized_zeta": 0.30781},
        "test": {"mse_normalized_avg": 0.13734, "mse_normalized_alpha": 0.03006, "mse_normalized_zeta": 0.24463},
    },
    "knn": {
        "val":  {"mse_normalized_avg": 0.21233, "mse_normalized_alpha": 0.09309, "mse_normalized_zeta": 0.33156, "best_k": 3},
        "test": {"mse_normalized_avg": 0.41936, "mse_normalized_alpha": 0.06515, "mse_normalized_zeta": 0.77357},
    },
}


# ── Figure 1: Baseline epoch progression ────────────────────────────────────
# Source: probe_epoch18/19/20/21/22.json

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.plot(EPOCHS, LP_TEST_NORM,  "o-",  color="#2563eb", label="Linear Probe (test)", linewidth=2, markersize=6)
ax.plot(EPOCHS, KNN_TEST_NORM, "s--", color="#16a34a", label="kNN (test)",           linewidth=2, markersize=6)
ax.plot(EPOCHS, LP_VAL_NORM,   "o:",  color="#93c5fd", label="Linear Probe (val)",   linewidth=1.5, markersize=5)
ax.plot(EPOCHS, KNN_VAL_NORM,  "s:",  color="#86efac", label="kNN (val)",            linewidth=1.5, markersize=5)

ax.axvline(20, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(20.1, 0.68, "best epoch\n(epoch 20)", fontsize=9, color="gray", va="top")
ax.axhline(RANDOM_BASELINE, color="red", linestyle=":", linewidth=1.2, alpha=0.7, label="Random baseline (1.0)")

ax.set_xlabel("Training Epoch")
ax.set_ylabel("Normalized MSE")
ax.set_title("Conv-JEPA Baseline — Representation Quality Across Epochs")
ax.set_xticks(EPOCHS)
ax.set_ylim(0, 1.1)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig1_epoch_progression.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig1_epoch_progression.png  [source: probe_epoch18-22.json]")


# ── Figure 2: Baseline vs No-Aug — overall test MSE ─────────────────────────
# Source: probe_baseline_final.json + final_main_no_aug.json

fig, ax = plt.subplots(figsize=(7, 4))

models   = ["Conv-JEPA\nBaseline", "Conv-JEPA\nw/o Aug"]
lp_test  = [BASELINE["linear_probe"]["test"]["mse_normalized_avg"],
            NO_AUG["linear_probe"]["test"]["mse_normalized_avg"]]
knn_test = [BASELINE["knn"]["test"]["mse_normalized_avg"],
            NO_AUG["knn"]["test"]["mse_normalized_avg"]]

x = np.arange(len(models))
w = 0.35
b1 = ax.bar(x - w/2, lp_test,  w, label="Linear Probe", color="#2563eb", edgecolor="white")
b2 = ax.bar(x + w/2, knn_test, w, label="kNN",          color="#16a34a", edgecolor="white")

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

ax.axhline(RANDOM_BASELINE, color="red", linestyle=":", linewidth=1.2, alpha=0.7, label="Random baseline (1.0)")
ax.set_ylabel("Normalized Test MSE")
ax.set_title("Baseline vs w/o Augmentation — Overall Test Performance")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.65)
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig2_model_comparison.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig2_model_comparison.png  [source: probe_baseline_final.json, final_main_no_aug.json]")


# ── Figure 3: Alpha vs Zeta breakdown — both models, both methods ────────────
# Source: probe_baseline_final.json + final_main_no_aug.json

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

params = ["α (activity)", "ζ (alignment)"]

for ax, (method_key, method_label, knn_k_b, knn_k_n) in zip(
    axes,
    [("linear_probe", "Linear Probe", None, None),
     ("knn",          "kNN",          10,   3)]
):
    base_vals   = [BASELINE[method_key]["test"]["mse_normalized_alpha"],
                   BASELINE[method_key]["test"]["mse_normalized_zeta"]]
    no_aug_vals = [NO_AUG[method_key]["test"]["mse_normalized_alpha"],
                   NO_AUG[method_key]["test"]["mse_normalized_zeta"]]

    x = np.arange(len(params))
    w = 0.35
    b1 = ax.bar(x - w/2, base_vals,   w, label="Baseline",  color="#2563eb", edgecolor="white")
    b2 = ax.bar(x + w/2, no_aug_vals, w, label="w/o Aug",   color="#f59e0b", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

    knn_str = f" (k={knn_k_b}/{knn_k_n})" if knn_k_b else ""
    ax.set_title(f"{method_label}{knn_str}")
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("Normalized Test MSE")
    ax.axhline(RANDOM_BASELINE, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

fig.suptitle("Per-Parameter Breakdown: Baseline vs w/o Augmentation (Test Set)", fontsize=12)
fig.tight_layout()
fig.savefig("figures/fig3_alpha_zeta_breakdown.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig3_alpha_zeta_breakdown.png  [source: probe_baseline_final.json, final_main_no_aug.json]")


# ── Figure 4: Summary table — both models ────────────────────────────────────
# Source: probe_baseline_final.json + final_main_no_aug.json

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

headers = ["Model", "Method", "Val MSE (norm.)", "Test MSE (norm.)",
           "Test α MSE (norm.)", "Test ζ MSE (norm.)"]
rows = [
    ["Baseline",  "Linear Probe",
     f"{BASELINE['linear_probe']['val']['mse_normalized_avg']:.4f}",
     f"{BASELINE['linear_probe']['test']['mse_normalized_avg']:.4f}",
     f"{BASELINE['linear_probe']['test']['mse_normalized_alpha']:.4f}",
     f"{BASELINE['linear_probe']['test']['mse_normalized_zeta']:.4f}"],
    ["Baseline",  "kNN (k=10)",
     f"{BASELINE['knn']['val']['mse_normalized_avg']:.4f}",
     f"{BASELINE['knn']['test']['mse_normalized_avg']:.4f}",
     f"{BASELINE['knn']['test']['mse_normalized_alpha']:.4f}",
     f"{BASELINE['knn']['test']['mse_normalized_zeta']:.4f}"],
    ["w/o Aug",   "Linear Probe",
     f"{NO_AUG['linear_probe']['val']['mse_normalized_avg']:.4f}",
     f"{NO_AUG['linear_probe']['test']['mse_normalized_avg']:.4f}",
     f"{NO_AUG['linear_probe']['test']['mse_normalized_alpha']:.4f}",
     f"{NO_AUG['linear_probe']['test']['mse_normalized_zeta']:.4f}"],
    ["w/o Aug",   "kNN (k=3)",
     f"{NO_AUG['knn']['val']['mse_normalized_avg']:.4f}",
     f"{NO_AUG['knn']['test']['mse_normalized_avg']:.4f}",
     f"{NO_AUG['knn']['test']['mse_normalized_alpha']:.4f}",
     f"{NO_AUG['knn']['test']['mse_normalized_zeta']:.4f}"],
    ["—", "Random baseline", "~1.0000", "~1.0000", "~1.0000", "~1.0000"],
]

tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.75)

for j in range(len(headers)):
    tbl[0, j].set_facecolor("#1e3a5f")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Baseline rows — light blue
for j in range(len(headers)):
    tbl[1, j].set_facecolor("#dbeafe")
    tbl[2, j].set_facecolor("#dbeafe")
# No-aug rows — light amber
for j in range(len(headers)):
    tbl[3, j].set_facecolor("#fef3c7")
    tbl[4, j].set_facecolor("#fef3c7")
# Random baseline — light red
for j in range(len(headers)):
    tbl[5, j].set_facecolor("#fee2e2")

ax.set_title("Conv-JEPA Evaluation Summary — Baseline vs w/o Augmentation", pad=12, fontsize=11)
fig.tight_layout()
fig.savefig("figures/fig4_summary_table.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig4_summary_table.png  [source: probe_baseline_final.json, final_main_no_aug.json]")

print("\nAll figures saved to figures/")

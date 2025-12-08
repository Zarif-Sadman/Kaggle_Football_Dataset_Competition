import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# GLOBAL STYLE SETTINGS 

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['figure.dpi'] = 200

# Bold tick labels globally
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2


# 1. Aggregate Metrics

models = ["LogReg", "Random Forest", "LightGBM", "XGBoost", "MLP", "GRU", "CNN1D"]

log_loss = {
    "LogReg": 1.0376,
    "Random Forest": 1.0165,
    "LightGBM": 1.0328,
    "XGBoost": 1.0099,
    "MLP": 1.0765,
    "GRU": 1.0300,
    "CNN1D": 1.0285,
}

accuracy = {
    "LogReg": 0.4581,
    "Random Forest": 0.4952,
    "LightGBM": 0.4581,
    "XGBoost": 0.4993,
    "MLP": 0.4674,
    "GRU": 0.4565,
    "CNN1D": 0.4680,
}



# 2. Horizontal Bar Charts 

log_vals = [log_loss[m] for m in models]
acc_vals = [accuracy[m] for m in models]
y_pos = np.arange(len(models))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# LOG LOSS PLOT
ax = axes[0]
ax.barh(y_pos, log_vals, color="royalblue")
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontweight="bold")
ax.invert_yaxis()
ax.set_xlabel("Log Loss", fontweight="bold")
ax.set_title("Test Log Loss by Model", fontweight="bold")

for i, v in enumerate(log_vals):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")

# ACCURACY PLOT
ax = axes[1]
ax.barh(y_pos, acc_vals, color="seagreen")
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontweight="bold")
ax.invert_yaxis()
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set_title("Test Accuracy by Model", fontweight="bold")
ax.set_xlim(0.44, 0.51)

for i, v in enumerate(acc_vals):
    ax.text(v + 0.0005, i, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")

fig.tight_layout()

# SAVE AS PDF
fig.savefig("acc.pdf", bbox_inches="tight")
plt.show()


# 3. CONFUSION MATRICES 

cm_xgb = np.array([
    [5888,   20, 1219],
    [3031,   31, 1188],
    [2851,   19, 2386],
])

cm_gru = np.array([
    [3367, 2543, 1217],
    [1269, 1899, 1082],
    [1021, 1908, 2327],
])

labels = ["home", "draw", "away"]

def row_norm(cm):
    return cm / cm.sum(axis=1, keepdims=True)

norm_xgb = row_norm(cm_xgb)
norm_gru = row_norm(cm_gru)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(wspace=0.3, right=0.90)

def plot_cm(ax, cm, cm_norm, title):
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    for i in range(3):
        for j in range(3):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold")

    return im

im1 = plot_cm(axes[0], cm_xgb, norm_xgb, "XGBoost – Test Confusion Matrix")
im2 = plot_cm(axes[1], cm_gru, norm_gru, "GRU – Test Confusion Matrix")

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label("Row-Normalized Percentage", fontweight="bold")

fig.savefig("Confusion.pdf", bbox_inches="tight")
plt.show()


# 4. PER-CLASS RECALL COMPARISON 

recall = {
    "LogReg": {"home": 0.4839, "draw": 0.4268, "away": 0.4482},
    "Random Forest": {"home": 0.7627, "draw": 0.0871, "away": 0.4623},
    "LightGBM": {"home": 0.4748, "draw": 0.4351, "away": 0.4540},
    "XGBoost": {"home": 0.8262, "draw": 0.0073, "away": 0.4540},
    "MLP": {"home": 0.6694, "draw": 0.1864, "away": 0.4207},
    "GRU": {"home": 0.4724, "draw": 0.4468, "away": 0.4427},
    "CNN1D": {"home": 0.5596, "draw": 0.3341, "away": 0.4521},
}

sel_models = ["LogReg", "Random Forest", "XGBoost", "GRU", "CNN1D"]
class_labels = ["home", "draw", "away"]

x = np.arange(len(class_labels))
width = 0.14

fig, ax = plt.subplots(figsize=(10, 5))

for idx, model in enumerate(sel_models):
    offsets = (idx - (len(sel_models)-1)/2) * width
    vals = [recall[model][c] for c in class_labels]
    ax.bar(x + offsets, vals, width, label=model, linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(class_labels, fontweight="bold")
ax.set_ylabel("Recall", fontweight="bold")
ax.set_ylim(0, 0.9)
ax.set_title("Per-Class Recall Comparison", fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig("Recall.pdf", bbox_inches="tight")
plt.show()

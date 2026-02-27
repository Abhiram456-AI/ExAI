import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy.stats import gaussian_kde


DPI = 300
BOOTSTRAP_ITERATIONS = 1000

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})


def compute_ground_truth(row):
    return 1 if "LB" in row["image_path"] else 0


def bootstrap_auc(true_labels, scores):
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.choice(len(true_labels), len(true_labels), replace=True)
        if len(np.unique(true_labels[indices])) < 2:
            continue
        fpr, tpr, _ = roc_curve(true_labels[indices], scores[indices])
        aucs.append(auc(fpr, tpr))
    return np.percentile(aucs, [2.5, 97.5])


def run_performance_evaluation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(current_dir, "validation_results.csv")

    if not os.path.exists(results_path):
        print("validation_results.csv not found.")
        return

    df = pd.read_csv(results_path)

    if "risk_score" not in df.columns:
        print("risk_score column missing.")
        return

    df["true_label"] = df.apply(compute_ground_truth, axis=1)
    true_labels = df["true_label"].values
    risk_scores = df["risk_score"].values

    # ================= ROC + AUC =================
    fpr, tpr, thresholds = roc_curve(true_labels, risk_scores)
    roc_auc = auc(fpr, tpr)
    ci_low, ci_high = bootstrap_auc(true_labels, risk_scores)

    print("\n================ ROC ANALYSIS =================")
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print("==============================================\n")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "roc_curve.png"), dpi=DPI)
    plt.savefig(os.path.join(current_dir, "roc_curve.pdf"))
    plt.close()

    if "crop" in df.columns:
        plt.figure(figsize=(6, 6))
        for crop in df["crop"].unique():
            crop_df = df[df["crop"] == crop]
            if len(crop_df["true_label"].unique()) < 2:
                continue
            fpr_c, tpr_c, _ = roc_curve(crop_df["true_label"], crop_df["risk_score"])
            auc_c = auc(fpr_c, tpr_c)
            plt.plot(fpr_c, tpr_c, linewidth=2, label=f"{crop} (AUC={auc_c:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Per-Crop ROC Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, "roc_per_crop.png"), dpi=DPI)
        plt.savefig(os.path.join(current_dir, "roc_per_crop.pdf"))
        plt.close()

    # ================= Precision-Recall =================
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, risk_scores)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2, label=f"AUC-PR = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "precision_recall_curve.png"), dpi=DPI)
    plt.savefig(os.path.join(current_dir, "precision_recall_curve.pdf"))
    plt.close()

    # ================= Calibration Curve =================
    prob_true, prob_pred = calibration_curve(true_labels, risk_scores, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "calibration_curve.png"), dpi=DPI)
    plt.savefig(os.path.join(current_dir, "calibration_curve.pdf"))
    plt.close()

    # ================= KDE Risk Distribution =================
    harmful_scores = df[df["true_label"] == 1]["risk_score"]
    non_harmful_scores = df[df["true_label"] == 0]["risk_score"]

    kde_harmful = gaussian_kde(harmful_scores)
    kde_non = gaussian_kde(non_harmful_scores)
    x_range = np.linspace(0, 1, 500)

    plt.figure(figsize=(7, 5))
    plt.plot(x_range, kde_harmful(x_range), linestyle='-', linewidth=2, label="Harmful (LB)")
    plt.plot(x_range, kde_non(x_range), linestyle='--', linewidth=2, label="Non-Harmful (NA)")
    plt.xlabel("Risk Score")
    plt.ylabel("Density")
    plt.title("Risk Score Density (KDE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "risk_density_kde.png"), dpi=DPI)
    plt.savefig(os.path.join(current_dir, "risk_density_kde.pdf"))
    plt.close()

    # ================= Confusion Matrix =================
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    predicted_labels = (risk_scores >= optimal_threshold).astype(int)

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, zero_division=0)
    rec = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    bal_acc = balanced_accuracy_score(true_labels, predicted_labels)

    print("================ PERFORMANCE REPORT =================")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Balanced Accuracy   : {bal_acc:.4f}")
    print(f"Precision           : {prec:.4f}")
    print(f"Recall              : {rec:.4f}")
    print(f"F1 Score            : {f1:.4f}")
    print("=====================================================\n")

    plt.figure(figsize=(6, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Greys')
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Non-Harmful", "Harmful"])
    plt.yticks([0, 1], ["Non-Harmful", "Harmful"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=12)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "confusion_matrix_normalized.png"), dpi=DPI)
    plt.savefig(os.path.join(current_dir, "confusion_matrix_normalized.pdf"))
    plt.close()

    metrics_summary = pd.DataFrame([{
        "AUC": roc_auc,
        "AUC_CI_Low": ci_low,
        "AUC_CI_High": ci_high,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Balanced_Accuracy": bal_acc,
        "Optimal_Threshold": optimal_threshold
    }])

    metrics_summary.to_csv(os.path.join(current_dir, "metrics_summary_ieee.csv"), index=False)

    print("Publication-ready plots generated successfully.")


if __name__ == "__main__":
    run_performance_evaluation()

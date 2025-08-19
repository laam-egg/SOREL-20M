from pefe_agent.config import *
import numpy as np
from numpy import ndarray
from typing import Any, Callable
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from .simple_stats import simple_stats
import os

def evaluate(figure_image_file_prefix, y_test, y_probs, threshold=0.5, roc_curve_func=roc_curve):
    # type: (str, ndarray, ndarray, float, Callable[[ndarray, ndarray], tuple[float, float, ndarray]]) -> dict[str, dict[str, int|float|str]]
    stats = simple_stats(y_probs, y_test, threshold)

    ROC_FILE_PATH = os.path.abspath(f"{figure_image_file_prefix}_IEF_ROC.png")
    DET_FILE_PATH = os.path.abspath(f"{figure_image_file_prefix}_IEF_DET.png")
    
    fpr, tpr, thresholds = roc_curve_func(y_test, y_probs)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # random chance line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(ROC_FILE_PATH, dpi=300, bbox_inches='tight')

    # Plot DET
    # DET curve: Detection Error Tradeoff curve
    fnr = 1 - tpr
    plt.figure()
    plt.plot(fpr, fnr, color='blue', label=f"DET curve")
    plt.plot([0, 1], [1, 0], color="gray", linestyle="--")  # random chance line
    plt.xscale("log")
    plt.yscale("log")
    ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    plt.xticks(ticks, [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    plt.yticks(ticks, [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve")
    plt.legend(loc="lower right")
    plt.savefig(DET_FILE_PATH, dpi=300, bbox_inches='tight')

    stats["roc"] = {
        "auc": roc_auc,
        "plot_path": ROC_FILE_PATH,
    }

    stats["det"] = {
        "plot_path": DET_FILE_PATH,
    }

    return stats

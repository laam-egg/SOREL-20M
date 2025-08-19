import lmdb
from pefe_agent.config import *
import msgpack
import msgpack_numpy
import numpy as np
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from tqdm import trange

msgpack_numpy.patch()

def roc_curve_custom(y_test, y_probs):
    thresholds = np.linspace(0, 1, num=100)
    fpr = []
    tpr = []
    for t in thresholds:
        y_pred = (y_probs >= t)
        fpr.append(np.sum((y_test == 0) & (y_pred == 1)) / np.sum(y_test == 0))
        tpr.append(np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1))
    return fpr, tpr, thresholds

def compute_and_plot(y_test, y_probs, roc_curve_func=roc_curve):
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
    plt.savefig(f"IEF_ROC_{roc_curve_func.__name__}.png", dpi=300, bbox_inches='tight')

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
    plt.savefig(f"IEF_DET_{roc_curve_func.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show()

from ..models.abstract_model import AbstractModel
from typing import Type

def ief(model_class, model_path, threshold):
    # type: (Type[AbstractModel], str, float) -> None

    print(f"Step 1/4: Loading model: {model_class.__name__}...", flush=True)
    model = model_class()
    print(f"Step 1/4: Loading model: from: {model_path}", flush=True)
    model.load(model_path)

    print(f"Step 2/4: Loading data...", flush=True)
    db = lmdb.open(config['self']['lmdb_path'], readonly=True, lock=False, map_size=1024 * 1024 * 1024 * 1024) # 1 TB
    y_test = []
    X_test = []
    num_entries = 0
    with db.begin() as txn:
        stat_info = txn.stat()
        num_entries = stat_info['entries']

        cursor = txn.cursor()
        PROGRESS = trange(num_entries)
        for k, v in cursor:
            PROGRESS.update()
            payload = msgpack.unpackb(v, raw=False)

            label = payload['lb']       # int
            y_test.append(label)

            features = payload['ef']    # numpy array
            X_test.append(features)

    assert PROGRESS.n == num_entries
    PROGRESS.close()

    print(f"Step 2/4: Loading data: Realigning...", flush=True)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Step 3/4: Running model inference...", flush=True)
    y_probs = model.predict(X_test)

    print(f"Step 4/4: Computing and plotting evaluation metrics: Computing simple stats...", flush=True)
    from .simple_stats import simple_stats
    simple_stats(y_probs, y_test, threshold)

    print(f"Step 4/4: Computing and plotting evaluation metrics: Using roc_curve_custom...", flush=True)
    compute_and_plot(y_test, y_probs, roc_curve_func=roc_curve_custom)
    
    print(f"Step 4/4: Computing and plotting evaluation metrics: Using sklearn.metrics.roc_curve...", flush=True)
    compute_and_plot(y_test, y_probs, roc_curve_func=roc_curve)

    print("DONE.")

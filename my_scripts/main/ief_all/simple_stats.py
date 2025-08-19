import numpy as np
from numpy import ndarray

def simple_stats(probs, labels, threshold):
    # type: (ndarray, ndarray, float) -> dict[str, dict[str, int|float]]
    TOTAL_HITS = 0
    TOTAL_MISSES = 0
    TOTAL_COUNT = 0

    TOTAL_TP = 0
    TOTAL_TN = 0
    TOTAL_FP = 0
    TOTAL_FN = 0

    TOTAL_TRUE = 0
    TOTAL_FALSE = 0
    try:
        frame_output = (probs >= threshold).astype(int)

        hits = np.sum(frame_output == labels)
        misses = np.sum(frame_output != labels)
        tp = np.sum((frame_output == 1) & (labels == 1))
        fp = np.sum((frame_output == 1) & (labels == 0))
        fn = np.sum((frame_output == 0) & (labels == 1))
        tn = np.sum((frame_output == 0) & (labels == 0))

        frame_size = len(probs)
        assert hits + misses == frame_size

        num_true = np.sum((labels == 1))
        num_false = np.sum((labels == 0))
        assert num_true + num_false == frame_size

        TOTAL_HITS += hits
        TOTAL_MISSES += misses
        TOTAL_COUNT += frame_size
        TOTAL_TP += tp
        TOTAL_TN += tn
        TOTAL_FP += fp
        TOTAL_FN += fn
        TOTAL_TRUE += num_true
        TOTAL_FALSE += num_false
    finally:
        print("")

    accuracy = TOTAL_HITS / TOTAL_COUNT if TOTAL_COUNT > 0 else 0.0
    precision = TOTAL_TP / (TOTAL_TP + TOTAL_FP) if (TOTAL_TP + TOTAL_FP) > 0 else 0.0
    recall = TOTAL_TP / (TOTAL_TP + TOTAL_FN) if (TOTAL_TP + TOTAL_FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    assert TOTAL_HITS + TOTAL_MISSES == TOTAL_COUNT
    return {
        "dataset": {
            "total_count": TOTAL_COUNT,
            "malware_count": TOTAL_TRUE,
            "benign_count": TOTAL_FALSE,
        },

        "simple_stats": [
            {
                "threshold": threshold,
                "total_hits": TOTAL_HITS,
                "total_misses": TOTAL_MISSES,
                "accuracy": accuracy,
                "TP": TOTAL_TP,
                "TN": TOTAL_TN,
                "FP": TOTAL_FP,
                "FN": TOTAL_FN,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ],
    }

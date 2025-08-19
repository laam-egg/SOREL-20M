import numpy as np
from numpy import ndarray

def simple_stats(probs, labels, threshold):
    # type: (ndarray, ndarray, float) -> None
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

    print("============== DATASET ================")
    print(f"TOTAL: {TOTAL_COUNT}")
    print(f"    = {TOTAL_TRUE} malware samples")
    print(f"    + {TOTAL_FALSE} benign samples")
    print()
    print("============= INFERENCE ===============")
    print(f"TOTAL HITS: {TOTAL_HITS}")
    print(f"TOTAL MISSES: {TOTAL_MISSES}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"TP: {TOTAL_TP}, TN: {TOTAL_TN}, FP: {TOTAL_FP}, FN: {TOTAL_FN}")
    print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1-score: {f1:.2%}")

    assert TOTAL_HITS + TOTAL_MISSES == TOTAL_COUNT
    print("No anomalies found.")
    print("=======================================")
    print("")

from . import *
import argparse
from typing import Callable

RUN_OPTIONS = {
    "LGBM": infer_lgbm,
    "FFNN": infer_ffnn,
    "pefe": pefe,
    "IEF_LGBM": IEF_LGBM,
    "IEF_FFNN": IEF_FFNN,
    "IEF_ALL": IEF_ALL,
    "IEF_ALL_VIZ": IEF_ALL_VIZ,
} # type: dict[str, Callable[[ArgparseNamespace], None]]

def main():
    parser = argparse.ArgumentParser(description="Run/evaluate SOREL-20M models")
    parser.add_argument("run_option", help=f"One of: {', '.join(RUN_OPTIONS)}")
    parser.add_argument("--pe-file", help="(LGBM, FFNN) Path to the PE file", default="/home/lam/Desktop/Viettel/LESS_DATA/Trojan.ATA_virussign.com_422cd1e311479e0bc0cb9bd9b4f106cf.exe")
    parser.add_argument("--lgb-model", help="(LGBM, IEF_LGBM) Path to LightGBM model file (.model)", default="../MODELS/lightGBM/seed0/lightgbm.model")
    parser.add_argument("--ffnn-model", help="(FFNN, IEF_FFNN) Path to FFNN model file (.pth)", default="../MODELS/FFNN/seed0/epoch_1.pt")
    parser.add_argument("--threshold", type=float, help="(IEF_LGBM, IEF_FFNN, IEF_ALL) Threshold of prediction when evaluating a model.", default=0.5)
    parser.add_argument("--lgb-models-dir", help="(IEF_ALL) Path to the LightGBM models directory", default="../MODELS/lightGBM/")
    parser.add_argument("--ffnn-models-dir", help="(IEF_ALL) Path to the FFNN models directory", default="../MODELS/FFNN/")
    parser.add_argument("--results-dir", help="(IEF_ALL, IEF_ALL_VIZ) Path to the results directory", default="../RESULTS/ALL")
    args = parser.parse_args()

    r = args.run_option
    try:
        func = RUN_OPTIONS[r]
    except KeyError:
        print(f"ERROR: Run option not implemented: {r}")
    else:
        func(args)

if __name__ == "__main__":
    main()

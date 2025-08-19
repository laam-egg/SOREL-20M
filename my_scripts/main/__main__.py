from . import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run/evaluate SOREL-20M models")
    parser.add_argument("run_option", help="Either 'LGBM' or 'FFNN'.")
    parser.add_argument("--pe-file", help="Path to the PE file", default="/home/lam/Desktop/Viettel/LESS_DATA/Trojan.ATA_virussign.com_422cd1e311479e0bc0cb9bd9b4f106cf.exe")
    parser.add_argument("--lgb-model", help="Path to LightGBM model file (.model)", default="../MODELS/lightGBM/seed0/lightgbm.model")
    parser.add_argument("--ffnn-model", help="Path to FFNN model file (.pth)", default="../MODELS/FFNN/seed0/epoch_1.pt")
    parser.add_argument("--threshold", type=float, help="Threshold of prediction when evaluating a model.", default=0.5)
    args = parser.parse_args()

    r = args.run_option
    if r == "LGBM":
        infer_lgbm(args)
    elif r == "FFNN":
        infer_ffnn(args)
    elif r == "pefe":
        pefe(args)
    elif r == "IEF_LGBM":
        IEF_LGBM(args)
    elif r == "IEF_FFNN":
        IEF_FFNN(args)
    else:
        print(f"Run option not implemented: {r}")

if __name__ == "__main__":
    main()

import argparse
from argparse import Namespace as ArgparseNamespace

def infer_lgbm(args):
    # type: (ArgparseNamespace) -> None
    from .models import SOREL20M_LGBM
    model = SOREL20M_LGBM()
    model.load(args.lgb_model)
    prob = model.predict_single_file(args.pe_file)
    print(f"On file: {args.pe_file}")
    print(f"LightGBM prediction: {prob:.6f}")

def infer_ffnn(args):
    # type: (ArgparseNamespace) -> None
    from .models import SOREL20M_FFNN
    model = SOREL20M_FFNN()
    model.load(args.ffnn_model)
    prob = model.predict_single_file(args.pe_file)
    print(f"On file: {args.pe_file}")
    print(f"FFNN prediction: {prob:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Run/evaluate SOREL-20M models")
    parser.add_argument("run_option", help="Either 'LGBM' or 'FFNN'.")
    parser.add_argument("--pe-file", help="Path to the PE file", default="/home/lam/Desktop/Viettel/LESS_DATA/Trojan.ATA_virussign.com_422cd1e311479e0bc0cb9bd9b4f106cf.exe")
    parser.add_argument("--lgb-model", help="Path to LightGBM model file (.model)", default="../MODELS/lightGBM/seed0/lightgbm.model")
    parser.add_argument("--ffnn-model", help="Path to FFNN model file (.pth)", default="../MODELS/FFNN/seed0/epoch_1.pt")
    args = parser.parse_args()

    if args.run_option == "LGBM":
        infer_lgbm(args)
    elif args.run_option == "FFNN":
        infer_ffnn(args)
    else:
        print(f"Run option not implemented: {args.run_option}")

if __name__ == "__main__":
    main()

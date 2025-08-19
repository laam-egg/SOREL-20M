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

def pefe(args):
    # type: (ArgparseNamespace) -> None
    from .pefe import run_pefe
    run_pefe()

def IEF_LGBM(args):
    # type: (ArgparseNamespace) -> None
    from .ief import ief
    from .models import SOREL20M_LGBM
    ief(SOREL20M_LGBM, args.lgb_model, args.threshold)

def IEF_FFNN(args):
    # type: (ArgparseNamespace) -> None
    from .ief import ief
    from .models import SOREL20M_FFNN
    ief(SOREL20M_FFNN, args.ffnn_model, args.threshold)

from argparse import Namespace as ArgparseNamespace
from typing import Type

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

def IEF_ALL(args):
    # type: (ArgparseNamespace) -> None

    from .models import SOREL20M_LGBM, SOREL20M_FFNN

    # NOTE: Stuff in ief_all aren't needed
    # anymore in favor of the new pefe-ief
    # module.

        # from .ief_all import ief_all
        # ief_all(
        #     {
        #         str(args.lgb_models_dir): SOREL20M_LGBM,
        #         str(args.ffnn_models_dir): SOREL20M_FFNN,
        #     },
        #     args.threshold,
        #     args.results_dir,
        # )

    models_dirs_and_classes = {
        str(args.lgb_models_dir): SOREL20M_LGBM,
        str(args.ffnn_models_dir): SOREL20M_FFNN,
    }

    thresholds = [*map(float, args.thresholds.split(","))]

    results_dir = args.results_dir
    
    from pprint import pprint
    print("===================================")
    print("LOADED OPTIONS:")
    pprint(dict(
        models_dirs_and_classes=models_dirs_and_classes,
        thresholds=thresholds,
        results_dir=results_dir,
    ))
    print("===================================")






    from pefe_ief import IEF
    from pefe_ief.models.abstract_model import AbstractModel
    from pefe_ief.dataset import PEFELMDBDataset
    import re

    ief = IEF(results_dir)


    def is_model_checkpoint_file(model_class, file_path):
        # type: (Type[AbstractModel], str) -> str
        if not (file_path.endswith(".pt") or file_path.endswith(".model")):
            print(f"NOTE: Skipping file as it does not appear to be a model/checkpoint file: {file_path}")
            return False
        return True
    
    def get_model_checkpoint_name(model_class, model_type_name, checkpoint_path):
        # type: (Type[AbstractModel], str, str) -> str | None
        p = checkpoint_path.lower()

        # FFNN
        maTch = re.search(r'seed\d+[\\/]+epoch\_\d+', p)
        if maTch:
            return model_type_name + "+" + maTch.group(0)
        
        # LGBM
        maTch = re.search(r'(seed\d+)[\\/]+(.*)\.model', p)
        if maTch:
            # Default LGBM model checkpoint path:
            # .../seed<N>/lightgbm.model where N is an integer
            # so we could omit the "lightgbm.model" part
            if maTch.group(2) == "lightgbm":
                return model_type_name + "+" + maTch.group(1)
            else:
                # non-default file name so preserve detail!
                return model_type_name + "+" + maTch.group(0)
        
        return None
    
    from pefe_agent.config import config
    X_test, y_test = PEFELMDBDataset().read(config['self']['lmdb_path'])

    ief.run(
        models_dirs_and_classes=models_dirs_and_classes,
        is_model_checkpoint_file=is_model_checkpoint_file,
        get_model_checkpoint_name=get_model_checkpoint_name,
        X_test=X_test, y_test=y_test,
        config=IEF.EvaluationConfig(
            thresholds=thresholds
        )
    )

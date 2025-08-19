# SOREL-20M Instructions

[Original README](./README-original.md)

- [SOREL-20M Instructions](#sorel-20m-instructions)
  - [Setup](#setup)
    - [Conda Environment](#conda-environment)
    - [EMBERv2 FE Patch](#emberv2-fe-patch)
    - [Download datasets and models](#download-datasets-and-models)
  - [Evaluate on SOREL-20M's own dataset](#evaluate-on-sorel-20ms-own-dataset)
  - [Interesting part: Infer any PE file yourself](#interesting-part-infer-any-pe-file-yourself)
  - [Troubleshooting](#troubleshooting)
    - [ValueError: Samples cannot be a single string. The input must be an iterable over iterables of strings.](#valueerror-samples-cannot-be-a-single-string-the-input-must-be-an-iterable-over-iterables-of-strings)

## Setup

### Conda Environment

Consult the original README.

**For FFNN models only:** If you want to use CPU
in evaluating FFNN models (or you don't have a
CUDA 12.2+ capable GPU) then:

```sh
conda env create -f environment-cpu.yml
```

Otherwise, use the GPU:

```sh
conda env create -f environment-gpu.yml
```

With older/newer versions of CUDA installed,
you might need to tweak versions of packages
in `environment-gpu.yml`.

### EMBERv2 FE Patch

**After creating the conda environment,**
**be sure to apply the following patch**
for EMBERv2 feature extractor (credit
<https://github.com/elastic/ember/issues/103#issuecomment-1623975101>):

1. Open file `<your_conda_root>/envs/sorel/lib/python3.8/site-packages/ember/features.py`
2. Change the line 192 from

    ```python
    entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
    ```

    to

    ```python
    entry_name_hashed = FeatureHasher(50, input_type="string").transform([ [raw_obj['entry']] ]).toarray()[0]
    ```

### Download datasets and models

Download the models and data using aws CLI
(guide in original README).

Be sure to activate the environment before proceeding.

```sh
conda activate sorel
```

## Evaluate on SOREL-20M's own dataset

**TL;DR:** I usually skip this part.

Using SOREL-20M's own dataset? Not particularly
interesting. You might want to run the commands
to see if there is any problem in environment
installation. Otherwise, I don't find value in
running these commands.

First, configure the variables in `$PROJECT_ROOT/config.py`.
Then:

1. For evaluating LightGBM models:

    ```sh
    cd $PROJECT_ROOT
    mkdir -p ./RESULTS/LGBM
    python evaluate.py evaluate_lgb ./MODELS/lightGBM/seed0/lightgbm.model ./RESULTS/LGBM --remove_missing_features=shas_missing_ember_features.json
    ```

2. For evaluating FFNN models:

    ```sh
    cd $PROJECT_ROOT
    mkdir -p ./RESULTS/FFNN
    python evaluate.py evaluate_network ./RESULTS/FFNN ./MODELS/FFNN/seed0/epoch_10.pt --remove_missing_features=shas_missing_ember_features.json
    ```

    Tested on machine with CUDA 12.4 (Driver 550.163.01).

## Interesting part: Infer any PE file yourself

```sh
cd $PROJECT_ROOT
cd my_scripts
```

Displaying detailed usage:

```sh
python -m main --help
```

Example usage:

1. Inference:

    ```sh
    python -m main LGBM \\
        --pe-file=/path/to/your/PE/file \\
        --lgb-model=/path/to/downloaded/LGBM/model/file/e.g./.../lightGBM/seed0/lightgbm.model
    
    python -m main FFNN \\
        --pe-file=/path/to/your/PE/file
        --ffnn-model=/path/to/downloaded/LGBM/model/file/e.g./.../FFNN/seed0/epoch_1.pt
    ```

## Troubleshooting

### ValueError: Samples cannot be a single string. The input must be an iterable over iterables of strings.

Looks like this:

```plain
Traceback (most recent call last):
  ...
  File ".../envs/sorel/lib/python3.8/site-packages/sklearn/feature_extraction/_hash.py", line 172, in transform
    raise ValueError(
ValueError: Samples can not be a single string. The input must be an iterable over iterables of strings.
```

**Solution:** Apply [the patch as detailed here](#emberv2-fe-patch).

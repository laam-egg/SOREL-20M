# SOREL-20M Instructions

[Original README](./README-original.md)

- [SOREL-20M Instructions](#sorel-20m-instructions)
  - [Setup](#setup)
    - [Conda Environment](#conda-environment)
    - [EMBERv2 FE Patch](#emberv2-fe-patch)
    - [Download datasets and models](#download-datasets-and-models)
  - [Evaluate on SOREL-20M's own dataset](#evaluate-on-sorel-20ms-own-dataset)
  - [Interesting part: Infer any PE file yourself](#interesting-part-infer-any-pe-file-yourself)
  - [Decoupling Feature Extraction and Inference](#decoupling-feature-extraction-and-inference)
    - [Step 1: Feature Extraction](#step-1-feature-extraction)
    - [Step 2: Inference on Extracted Features](#step-2-inference-on-extracted-features)
    - [Step 3: Comparing Models](#step-3-comparing-models)
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
    python -m main LGBM \
        --pe-file=/path/to/your/PE/file \ 
        --lgb-model=/path/to/downloaded/LGBM/model/file/e.g./.../lightGBM/seed0/lightgbm.model
    
    python -m main FFNN \
        --pe-file=/path/to/your/PE/file \ 
        --ffnn-model=/path/to/downloaded/LGBM/model/file/e.g./.../FFNN/seed0/epoch_1.pt
    ```

## Decoupling Feature Extraction and Inference

The previous part showed how to infer one PE file at a time.
But how about inferring multiple PE files?

Decoupling Feature Extraction and Inference would be helpful
when dealing with extremely large set of PE files.

Extracting features from hundreds of thousands of PE files
would often take several hours. Meanwhile, running model
inference on the extracted features usually takes seconds.

Therefore, you would want to extract features from them
*for once* - then just run inference, assess models...
much faster afterwards.

### Step 1: Feature Extraction

Configure and run the [`pefe-loader`](https://github.com/pefe-system/pefe-loader)
so that it iterates over directories of benign and
malware PE files.

Then, in `$PROJECT_ROOT/my_scripts`, copy `config.example.json`
to a new file named `config.json` also in the same
directory. Configure the variables as appropriate.
The loader host and port shall be the same as what
you have configured in `pefe-loader` previously.

At this point, make sure you have run `pefe-loader`
with the loader type `SpreadFileLoader`. Then, run the
feature extraction agent:

```sh
conda activate sorel
cd $PROJECT_ROOT
cd my_scripts
python -m main pefe
```

You should run the above in multiple terminals
so that multiple feature extraction agents are
created and participate in parallel processing!

### Step 2: Inference on Extracted Features

```sh
cd $PROJECT_ROOT
cd my_scripts
```

then

```sh
python -m main IEF_LGBM \ 
    --lgb-model=/path/to/downloaded/LGBM/model/file/e.g./.../lightGBM/seed0/lightgbm.model
```

or

```sh
python -m main IEF_FFNN \ 
    --ffnn-model=/path/to/downloaded/LGBM/model/file/e.g./.../FFNN/seed0/epoch_1.pt
```

(IEF stands for "Infer on Extracted Features")

This would also evaluate the specified model using
various metrics e.g. F1 and AUC scores, ROC and
DET curves.

The graphics will also be saved as image files
into the current working directory.

### Step 3: Comparing Models

The final boss! This will:

- Scan directories containing multiple model/checkpoint files
    (just like SOREL-20M's baseline checkpoints directories),
- Load the models,
- Evaluate them on the LMDB dataset (which are the features
    that we extracted from PE files earlier),
- Render all those models' evaluation results as json and
    charts.

```sh
cd $PROJECT_ROOT
mkdir -p ./RESULTS/ALL
cd my_scripts

python -m main IEF_ALL \ 
    --lgb-models-dir=/path/to/downloaded/LGBM/models/directory/e.g./.../lightGBM/ \ 
    --ffnn-models-dir=/path/to/downloaded/FFNN/models/directory/e.g./.../FFNN/ \ 
    --results-dir=../RESULTS/ALL
```

The results are stored in, of course, the results dir.
In this case, it is `$PROJECT_ROOT/RESULTS/ALL`.

**We use a different Python environment for viewing**
**these results.**

```sh
cd $PROJECT_ROOT
cd my_scripts/ief_all_viz
conda env create -f environment.yml
conda activate sorel_ief_all_viz
jupyter nbextension enable --py widgetsnbextension
```

To preview raw results:

```sh
python preview.py ../../RESULTS/ALL/
```

To visualize/render the results:

```sh
jupyter notebook --url=127.0.0.1 --port=8888
```

then open a browser, navigate to `localhost:8888`,
choose to open the file `ief_all_viz.ipynb`. Run
all cells.

It is not recommended to run this notebook in
VSCode as the kernel version is no longer supported.
I have also experienced weird bugs there myself,
e.g. tqdm does not update, rendering images is slow
as hell (or never finishes), and HTML elements
are unexpectedly rendered in duplicate.

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

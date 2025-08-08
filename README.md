# SOREL-20M Instructions

[Original README](./README-original.md)

Download the models and data using aws CLI
(guide in original README), then

```sh
mkdir ./RESULTS
python evaluate.py evaluate_lgb ./MODELS/lightGBM/seed0/lightgbm.model ./RESULTS/
# or, for immediate inference (but might throw out exception on data with missing features):
python evaluate.py evaluate_lgb ./MODELS/lightGBM/seed0/lightgbm.model ./RESULTS/ --remove_missing_features=False
```

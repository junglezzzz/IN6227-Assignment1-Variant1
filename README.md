IN6227 Assignment 1 — Variant 1

This project compares logistic regression and random forest on the course-provided binary classification dataset. Data inspection, three rounds of stratified cross-validation, and final test evaluation are kept in separate scripts. Model selection uses `train.csv`; `test.csv` is scored only after the final settings are fixed.

Place the course-provided files at `dataset/train.csv` and `dataset/test.csv`. The scripts require Python 3, NumPy, pandas, and scikit-learn. `compare_models.py` provides shared functions for the weighted-round scripts; it does not need to be run separately to reproduce the reported workflow.


Run

```bash
python -m pip install numpy pandas scikit-learn
python inspect_data.py
python compare_models_round1_weighted.py
python compare_models_round2_weighted.py
python compare_models_round3_weights.py
python evaluate_final.py --output-dir results/reproduction
```

Outputs are saved under `results/`. The final evaluation will not overwrite an existing report, so use a new output directory if running it again.


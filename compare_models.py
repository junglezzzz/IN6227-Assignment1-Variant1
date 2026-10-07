import argparse
import csv
import warnings
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError as exc:
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


TARGET = "label"
NUMERIC = [
    "aptitude_score", "performance_score", "stability_index", "load_ratio",
    "index_weight", "activity_duration", "composite_rank",
]
CORRELATED = ["performance_score", "stability_index", "composite_rank"]
SEED = 42
FOLDS = 5

HERE = Path(__file__).resolve().parent
LOCAL_TRAIN = HERE / "dataset" / "train.csv"
DESKTOP_TRAIN = Path("/Users/jungle/Desktop/NTU/Assignment/mining1/dataset/train.csv")
DEFAULT_TRAIN = LOCAL_TRAIN if LOCAL_TRAIN.is_file() else DESKTOP_TRAIN


def load_training(path):
    data = pd.read_csv(path, keep_default_na=False)
    data = data.replace(r"^\s*$", np.nan, regex=True)
    if TARGET not in data:
        raise ValueError("train.csv must contain a 'label' column.")

    labels = data[TARGET]
    invalid = sorted(set(labels.dropna()) - {"yes", "no"})
    if invalid:
        raise ValueError(f"Unexpected label values: {invalid}")

    data = data.loc[labels.isin(["yes", "no"])].copy()
    features = [column for column in data if column != TARGET]
    missing_columns = sorted(set(NUMERIC) - set(features))
    if missing_columns:
        raise ValueError(f"Missing numerical columns: {missing_columns}")
    if len(features) != 15:
        raise ValueError(f"Expected 15 predictors; found {len(features)}.")

    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="raise")
    categorical = [column for column in features if column not in NUMERIC]
    X = data[features]
    y = data[TARGET].eq("yes").astype(int)
    return X, y, features, categorical


def feature_plans(features):
    plans = {"A": features}
    for name, keep in zip("BCD", CORRELATED):
        plans[name] = [column for column in features
                       if column not in CORRELATED or column == keep]
    return plans


def configurations():
    for plan in "ABCD":
        for c in (0.1, 1.0, 10.0):
            yield {"model": "Logistic regression", "plan": plan, "C": c,
                   "trees": "", "depth": "", "leaf": ""}

    for plan in "ABCD":
        for trees, depth, leaf in product((200, 500), (10, 20, None), (1, 5)):
            yield {"model": "Random forest", "plan": plan, "C": "",
                   "trees": trees, "depth": depth, "leaf": leaf}


def make_pipeline(setting, columns, categorical):
    numeric = [column for column in columns if column in NUMERIC]
    text = [column for column in columns if column in categorical]
    numeric_steps = [("fill", SimpleImputer(strategy="median"))]
    if setting["model"] == "Logistic regression":
        numeric_steps.append(("scale", StandardScaler()))
        classifier = LogisticRegression(
            C=setting["C"], solver="lbfgs", max_iter=1000, tol=1e-4,
            class_weight=None,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=setting["trees"], max_depth=setting["depth"],
            min_samples_leaf=setting["leaf"], class_weight=None,
            random_state=SEED, n_jobs=2,
        )

    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), text),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", classifier)])


def evaluate(setting, X, y, columns, categorical, folds):
    scores = {key: [] for key in ("f1", "train_f1", "precision", "recall", "accuracy")}
    start = perf_counter()

    for train_idx, valid_idx in folds:
        model = make_pipeline(setting, columns, categorical)
        X_train, X_valid = X.iloc[train_idx][columns], X.iloc[valid_idx][columns]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(
                f"Logistic regression did not converge for plan {setting['plan']} "
                f"and C={setting['C']}. Check max_iter before using its scores."
            )

        predicted = model.predict(X_valid)
        scores["f1"].append(f1_score(y_valid, predicted, pos_label=1))
        scores["precision"].append(precision_score(y_valid, predicted, pos_label=1,
                                                    zero_division=0))
        scores["recall"].append(recall_score(y_valid, predicted, pos_label=1))
        scores["accuracy"].append(accuracy_score(y_valid, predicted))
        scores["train_f1"].append(f1_score(y_train, model.predict(X_train), pos_label=1))

    result = dict(setting)
    for fold, score in enumerate(scores["f1"], start=1):
        result[f"fold_{fold}_f1"] = score
    result.update({
        "mean_f1": float(np.mean(scores["f1"])),
        "sd_f1": float(np.std(scores["f1"], ddof=1)),
        "mean_train_f1": float(np.mean(scores["train_f1"])),
        "mean_precision": float(np.mean(scores["precision"])),
        "mean_recall": float(np.mean(scores["recall"])),
        "mean_accuracy": float(np.mean(scores["accuracy"])),
        "seconds": perf_counter() - start,
    })
    return result


def parameter_text(row):
    if row["model"] == "Logistic regression":
        return f"C={row['C']:g}"
    depth = "None" if row["depth"] in (None, "") else row["depth"]
    return f"trees={row['trees']}, depth={depth}, leaf={row['leaf']}"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def summary(results, row_count, csv_name):
    lines = ["# Five-fold cross-validation", "",
             markdown_table(["Setting", "Value"], [
                 ["Source", "train.csv only"],
                 ["Rows with a label", f"{row_count:,}"],
                 ["Folds", "5 stratified folds, shuffled, seed 42"],
                 ["Positive class", "yes"],
                 ["Main comparison score", "Mean validation F1"],
                 ["Logistic regression", "L2 regularisation, lbfgs, max_iter=1000, tol=0.0001"],
                 ["Random forest", "n_jobs=2, seed 42"],
             ]), "", "## Highest mean F1 in this grid", ""]

    best_by_model = [max((row for row in results if row["model"] == model),
                         key=lambda row: row["mean_f1"])
                     for model in ("Logistic regression", "Random forest")]
    lines += [markdown_table(
        ["Model", "Features", "Parameters", "Mean F1", "F1 SD", "Train F1"],
        [[row["model"], row["plan"], parameter_text(row),
          f"{row['mean_f1']:.4f}", f"{row['sd_f1']:.4f}",
          f"{row['mean_train_f1']:.4f}"] for row in best_by_model]),
        "", "## Best-scoring parameters within each feature plan", ""]

    best_by_plan = [max((row for row in results if row["model"] == model
                        and row["plan"] == plan), key=lambda row: row["mean_f1"])
                    for model in ("Logistic regression", "Random forest")
                    for plan in "ABCD"]
    lines += [markdown_table(
        ["Model", "Features", "Parameters", "Mean F1", "F1 SD", "Train F1",
         "Precision", "Recall", "Accuracy"],
        [[row["model"], row["plan"], parameter_text(row),
          f"{row['mean_f1']:.4f}", f"{row['sd_f1']:.4f}",
          f"{row['mean_train_f1']:.4f}", f"{row['mean_precision']:.4f}",
          f"{row['mean_recall']:.4f}", f"{row['mean_accuracy']:.4f}"]
         for row in best_by_plan]),
        "", "## Feature plans", "",
        markdown_table(["Plan", "Correlated columns retained", "Original predictors"], [
            ["A", "performance_score, stability_index, composite_rank", 15],
            ["B", "performance_score", 13],
            ["C", "stability_index", 13],
            ["D", "composite_rank", 13],
        ]), "", f"All 60 configurations and their five fold scores: [{csv_name}]({csv_name})."]
    return "\n".join(lines) + "\n"


def save_results(output_dir, results, row_count):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        ending = "" if suffix == 1 else f"_{suffix}"
        report = output_dir / f"cv_summary{ending}.md"
        detail = output_dir / f"cv_results{ending}.csv"
        if not report.exists() and not detail.exists():
            break
        suffix += 1

    columns = ["model", "plan", "C", "trees", "depth", "leaf"]
    columns += [f"fold_{fold}_f1" for fold in range(1, FOLDS + 1)]
    columns += ["mean_f1", "sd_f1", "mean_train_f1", "mean_precision",
                "mean_recall", "mean_accuracy", "seconds"]
    with detail.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            values = {}
            for key, value in row.items():
                if key == "depth" and row["model"] == "Random forest" and value is None:
                    values[key] = "None"
                elif key == "C" and value != "":
                    values[key] = f"{value:g}"
                elif isinstance(value, float):
                    values[key] = f"{value:.6f}"
                else:
                    values[key] = value
            writer.writerow(values)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(summary(results, row_count, detail.name))
    return report, detail


def main():
    parser = argparse.ArgumentParser(description="Compare the planned models on train.csv.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if SKLEARN_IMPORT_ERROR is not None:
        raise SystemExit(
            "scikit-learn is missing. Install it in the active environment with: "
            "python -m pip install scikit-learn"
        )
    if not args.train.is_file():
        parser.error(f"Training file not found: {args.train}")

    X, y, features, categorical = load_training(args.train)
    plans = feature_plans(features)
    folds = list(StratifiedKFold(n_splits=FOLDS, shuffle=True,
                                 random_state=SEED).split(X, y))
    settings = list(configurations())
    results = []
    for number, setting in enumerate(settings, start=1):
        result = evaluate(setting, X, y, plans[setting["plan"]], categorical, folds)
        results.append(result)
        print(f"{number:2d}/{len(settings)}  {setting['model']}, "
              f"plan {setting['plan']}, {parameter_text(setting)}  "
              f"F1={result['mean_f1']:.4f}", flush=True)

    report, detail = save_results(args.output_dir, results, len(y))
    print(f"Summary: {report}")
    print(f"All configurations: {detail}")


if __name__ == "__main__":
    main()

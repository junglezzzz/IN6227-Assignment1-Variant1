import argparse
import csv
import warnings
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

import compare_models as base

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
MODELS = ("Logistic regression", "Random forest")
PLANS = "ABCD"
WEIGHTS = (None, "balanced")
LOGISTIC_C = (0.1, 1.0, 10.0)
FOREST_TREES = (100, 300)
FOREST_DEPTH = (10, 20, None)
FOREST_LEAF = (1, 5)


def configurations():
    for plan, c, weight in product(PLANS, LOGISTIC_C, WEIGHTS):
        yield {
            "model": MODELS[0], "plan": plan, "C": c,
            "trees": "", "depth": "", "leaf": "", "class_weight": weight,
        }
    for plan, trees, depth, leaf, weight in product(
        PLANS, FOREST_TREES, FOREST_DEPTH, FOREST_LEAF, WEIGHTS
    ):
        yield {
            "model": MODELS[1], "plan": plan, "C": "",
            "trees": trees, "depth": depth, "leaf": leaf,
            "class_weight": weight,
        }


def make_pipeline(setting, columns, categorical):
    numeric = [column for column in columns if column in base.NUMERIC]
    text = [column for column in columns if column in categorical]
    numeric_steps = [("fill", SimpleImputer(strategy="median"))]

    if setting["model"] == MODELS[0]:
        numeric_steps.append(("scale", StandardScaler()))
        classifier = LogisticRegression(
            C=setting["C"], solver="lbfgs", max_iter=1000, tol=1e-4,
            class_weight=setting["class_weight"],
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=setting["trees"], max_depth=setting["depth"],
            min_samples_leaf=setting["leaf"],
            class_weight=setting["class_weight"],
            random_state=base.SEED, n_jobs=2,
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
    scores = {
        key: [] for key in ("f1", "train_f1", "precision", "recall", "accuracy")
    }
    started = perf_counter()

    for train_idx, valid_idx in folds:
        model = make_pipeline(setting, columns, categorical)
        X_train = X.iloc[train_idx][columns]
        X_valid = X.iloc[valid_idx][columns]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(
                f"Logistic regression did not converge: plan={setting['plan']}, "
                f"C={setting['C']}, class_weight={setting['class_weight']}."
            )

        validation_prediction = model.predict(X_valid)
        training_prediction = model.predict(X_train)
        scores["f1"].append(f1_score(y_valid, validation_prediction))
        scores["train_f1"].append(f1_score(y_train, training_prediction))
        scores["precision"].append(
            precision_score(y_valid, validation_prediction, zero_division=0)
        )
        scores["recall"].append(recall_score(y_valid, validation_prediction))
        scores["accuracy"].append(accuracy_score(y_valid, validation_prediction))

    result = dict(setting)
    for fold_number, score in enumerate(scores["f1"], start=1):
        result[f"fold_{fold_number}_f1"] = score
    result.update({
        "mean_f1": float(np.mean(scores["f1"])),
        "sd_f1": float(np.std(scores["f1"], ddof=1)),
        "mean_train_f1": float(np.mean(scores["train_f1"])),
        "mean_precision": float(np.mean(scores["precision"])),
        "mean_recall": float(np.mean(scores["recall"])),
        "mean_accuracy": float(np.mean(scores["accuracy"])),
        "seconds": perf_counter() - started,
    })
    return result


def setting_text(row):
    if row["model"] == MODELS[0]:
        return f"C={row['C']:g}"
    depth = "None" if row["depth"] is None else row["depth"]
    return f"trees={row['trees']}, depth={depth}, leaf={row['leaf']}"


def weight_text(weight):
    return "None" if weight is None else weight


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def summary_row(row):
    return [
        row["model"], row["plan"], setting_text(row),
        weight_text(row["class_weight"]), f"{row['mean_f1']:.4f}",
        f"{row['sd_f1']:.4f}", f"{row['mean_train_f1']:.4f}",
        f"{row['mean_train_f1'] - row['mean_f1']:.4f}",
        f"{row['mean_precision']:.4f}", f"{row['mean_recall']:.4f}",
        f"{row['mean_accuracy']:.4f}",
    ]


def build_summary(results, row_count, csv_name):
    best_by_model = [
        max((row for row in results if row["model"] == model),
            key=lambda row: row["mean_f1"])
        for model in MODELS
    ]
    best_by_plan_weight = [
        max((row for row in results
             if row["model"] == model
             and row["plan"] == plan
             and row["class_weight"] == weight),
            key=lambda row: row["mean_f1"])
        for model in MODELS for plan in PLANS for weight in WEIGHTS
    ]
    headers = [
        "Model", "Features", "Parameters", "Class weight", "Mean F1",
        "F1 SD", "Train F1", "Train-validation gap", "Precision",
        "Recall", "Accuracy",
    ]
    return "\n".join([
        "# First-round cross-validation",
        "",
        markdown_table(["Setting", "Value"], [
            ["Source", "train.csv only"],
            ["Rows with a label", f"{row_count:,}"],
            ["Validation", "5 stratified folds, shuffled, seed 42"],
            ["Positive class", "yes"],
            ["Main comparison score", "Mean validation F1"],
            ["Logistic regression", "plans A-D; C=0.1, 1, 10; weight=None, balanced"],
            ["Random forest", "plans A-D; trees=100, 300; depth=10, 20, None; leaf=1, 5; weight=None, balanced"],
            ["Total configurations", str(len(results))],
        ]),
        "",
        "## Highest mean F1 by model",
        "",
        markdown_table(headers, [summary_row(row) for row in best_by_model]),
        "",
        "## Highest mean F1 by feature plan and class weight",
        "",
        markdown_table(headers, [summary_row(row) for row in best_by_plan_weight]),
        "",
        "## Feature plans",
        "",
        markdown_table(
            ["Plan", "Correlated columns retained", "Original predictors"],
            [
                ["A", "performance_score, stability_index, composite_rank", 15],
                ["B", "performance_score", 13],
                ["C", "stability_index", 13],
                ["D", "composite_rank", 13],
            ],
        ),
        "",
        f"All configurations and fold scores: [{csv_name}]({csv_name}).",
        "",
    ])


def save_results(output_dir, results, row_count):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        ending = "" if suffix == 1 else f"_{suffix}"
        report = output_dir / f"round1_weighted_summary{ending}.md"
        detail = output_dir / f"round1_weighted_results{ending}.csv"
        if not report.exists() and not detail.exists():
            break
        suffix += 1

    columns = ["model", "plan", "C", "trees", "depth", "leaf", "class_weight"]
    columns += [f"fold_{fold}_f1" for fold in range(1, base.FOLDS + 1)]
    columns += [
        "mean_f1", "sd_f1", "mean_train_f1", "mean_precision",
        "mean_recall", "mean_accuracy", "seconds",
    ]
    with detail.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            values = {}
            for key, value in row.items():
                if key in ("depth", "class_weight") and value is None:
                    values[key] = "None"
                elif key == "C" and value != "":
                    values[key] = f"{value:g}"
                elif isinstance(value, float):
                    values[key] = f"{value:.6f}"
                else:
                    values[key] = value
            writer.writerow(values)

    with report.open("x", encoding="utf-8") as handle:
        handle.write(build_summary(results, row_count, detail.name))
    return report, detail


def main():
    parser = argparse.ArgumentParser(
        description="First-round search with class weighting on train.csv."
    )
    parser.add_argument("--train", type=Path, default=base.DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if not args.train.is_file():
        parser.error(f"Training file not found: {args.train}")

    X, y, features, categorical = base.load_training(args.train)
    plans = base.feature_plans(features)
    folds = list(StratifiedKFold(
        n_splits=base.FOLDS, shuffle=True, random_state=base.SEED
    ).split(X, y))
    settings = list(configurations())
    if len(settings) != 120:
        raise RuntimeError(f"Expected 120 configurations, found {len(settings)}.")

    results = []
    for number, setting in enumerate(settings, start=1):
        result = evaluate(
            setting, X, y, plans[setting["plan"]], categorical, folds
        )
        results.append(result)
        print(
            f"{number:3d}/{len(settings)}  {setting['model']}, "
            f"plan {setting['plan']}, {setting_text(setting)}, "
            f"class_weight={weight_text(setting['class_weight'])}, "
            f"F1={result['mean_f1']:.4f}",
            flush=True,
        )

    report, detail = save_results(args.output_dir, results, len(y))
    print(f"Summary: {report}")
    print(f"All configurations: {detail}")


if __name__ == "__main__":
    main()

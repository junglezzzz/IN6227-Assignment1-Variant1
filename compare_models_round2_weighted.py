import argparse
import csv
from itertools import product
from pathlib import Path

import compare_models as data
import compare_models_round1_weighted as first_round


HERE = Path(__file__).resolve().parent
WEIGHTS = (
    ("None", None),
    ("yes=1.5", {0: 1, 1: 1.5}),
    ("yes=2.5", {0: 1, 1: 2.5}),
    ("balanced", "balanced"),
)
LOGISTIC_PLANS = ("A", "C")
LOGISTIC_C = (0.1, 1.0, 10.0)
FOREST_PLANS = ("A", "D")
FOREST_TREES = (50, 100)
FOREST_PAIRS = tuple(product((15, 20, 25, None), (3, 5, 10))) + ((20, 1),)

COLUMNS = [
    "model", "plan", "C", "trees", "depth", "leaf", "class_weight",
    "fold_1_f1", "fold_2_f1", "fold_3_f1", "fold_4_f1", "fold_5_f1",
    "mean_f1", "sd_f1", "mean_train_f1", "mean_precision",
    "mean_recall", "mean_accuracy", "seconds",
]


def configurations():
    for plan, c, (weight_name, weight) in product(
        LOGISTIC_PLANS, LOGISTIC_C, WEIGHTS
    ):
        yield {
            "model": "Logistic regression", "plan": plan, "C": c,
            "trees": "", "depth": "", "leaf": "",
            "class_weight": weight, "weight_name": weight_name,
        }
    for plan, trees, (depth, leaf), (weight_name, weight) in product(
        FOREST_PLANS, FOREST_TREES, FOREST_PAIRS, WEIGHTS
    ):
        yield {
            "model": "Random forest", "plan": plan, "C": "",
            "trees": trees, "depth": depth, "leaf": leaf,
            "class_weight": weight, "weight_name": weight_name,
        }


def csv_row(result):
    row = {}
    for key in COLUMNS:
        value = result[key]
        if key == "class_weight":
            row[key] = result["weight_name"]
        elif key == "depth" and result["model"] == "Random forest" and value is None:
            row[key] = "None"
        elif key == "C" and value != "":
            row[key] = f"{value:g}"
        elif isinstance(value, float):
            row[key] = f"{value:.6f}"
        else:
            row[key] = value
    return row


def output_paths(output_dir):
    suffix = 1
    while True:
        ending = "" if suffix == 1 else f"_{suffix}"
        stem = f"round2_weighted{ending}"
        report = output_dir / f"{stem}_summary.md"
        detail = output_dir / f"{stem}_results.csv"
        partial = output_dir / f"{stem}_results.partial.csv"
        if partial.exists() or (not report.exists() and not detail.exists()):
            return report, detail, partial
        suffix += 1


def read_partial(partial, settings):
    if not partial.exists():
        with partial.open("x", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=COLUMNS).writeheader()
        return []

    with partial.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Partial results have unexpected columns.")
        rows = list(reader)
    if len(rows) > len(settings):
        raise ValueError("Partial results contain too many configurations.")
    keys = ("model", "plan", "C", "trees", "depth", "leaf", "class_weight")
    for number, (row, setting) in enumerate(zip(rows, settings), start=1):
        expected = csv_row({**setting, **{
            "fold_1_f1": 0.0, "fold_2_f1": 0.0, "fold_3_f1": 0.0,
            "fold_4_f1": 0.0, "fold_5_f1": 0.0, "mean_f1": 0.0,
            "sd_f1": 0.0, "mean_train_f1": 0.0,
            "mean_precision": 0.0, "mean_recall": 0.0,
            "mean_accuracy": 0.0, "seconds": 0.0,
        }})
        if any(row[key] != str(expected[key]) for key in keys):
            raise ValueError(f"Partial results differ from setting {number}.")
    return rows


def setting_text(row):
    if row["model"] == "Logistic regression":
        return f"C={float(row['C']):g}"
    return (
        f"trees={row['trees']}, depth={row['depth']}, leaf={row['leaf']}"
    )


def summary_row(row):
    gap = float(row["mean_train_f1"]) - float(row["mean_f1"])
    return [
        row["model"], row["plan"], setting_text(row), row["class_weight"],
        f"{float(row['mean_f1']):.4f}", f"{float(row['sd_f1']):.4f}",
        f"{float(row['mean_train_f1']):.4f}", f"{gap:.4f}",
        f"{float(row['mean_precision']):.4f}",
        f"{float(row['mean_recall']):.4f}",
        f"{float(row['mean_accuracy']):.4f}",
    ]


def build_summary(rows, row_count, csv_name):
    models = ("Logistic regression", "Random forest")
    plans_by_model = {
        "Logistic regression": LOGISTIC_PLANS,
        "Random forest": FOREST_PLANS,
    }
    best_models = [
        max((row for row in rows if row["model"] == model),
            key=lambda row: float(row["mean_f1"]))
        for model in models
    ]
    best_groups = [
        max((row for row in rows
             if row["model"] == model
             and row["plan"] == plan
             and row["class_weight"] == weight_name),
            key=lambda row: float(row["mean_f1"]))
        for model in models
        for plan in plans_by_model[model]
        for weight_name, _ in WEIGHTS
    ]
    headers = [
        "Model", "Features", "Parameters", "Class weight", "Mean F1",
        "F1 SD", "Train F1", "Train-validation gap", "Precision",
        "Recall", "Accuracy",
    ]
    return "\n".join([
        "# Second-round cross-validation",
        "",
        first_round.markdown_table(["Setting", "Value"], [
            ["Source", "train.csv only"],
            ["Rows with a label", f"{row_count:,}"],
            ["Validation", "5 stratified folds, shuffled, seed 42"],
            ["Positive class", "yes"],
            ["Main comparison score", "Mean validation F1"],
            ["Logistic regression", "plans A, C; C=0.1, 1, 10"],
            ["Random forest", "plans A, D; trees=50, 100"],
            ["Forest depth and leaf", "depth=15, 20, 25, None with leaf=3, 5, 10; plus depth=20, leaf=1"],
            ["Class weights", "None; yes=1.5; yes=2.5; balanced"],
            ["Configurations", str(len(rows))],
        ]),
        "",
        "## Highest mean F1 by model",
        "",
        first_round.markdown_table(
            headers, [summary_row(row) for row in best_models]
        ),
        "",
        "## Highest mean F1 by feature plan and class weight",
        "",
        first_round.markdown_table(
            headers, [summary_row(row) for row in best_groups]
        ),
        "",
        "## Feature plans",
        "",
        first_round.markdown_table(
            ["Plan", "Correlated columns retained", "Original predictors"],
            [
                ["A", "performance_score, stability_index, composite_rank", 15],
                ["C", "stability_index", 13],
                ["D", "composite_rank", 13],
            ],
        ),
        "",
        f"All configurations and fold scores: [{csv_name}]({csv_name}).",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Run the planned second-round cross-validation on train.csv."
    )
    parser.add_argument("--train", type=Path, default=data.DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if not args.train.is_file():
        parser.error(f"Training file not found: {args.train}")

    settings = list(configurations())
    if len(FOREST_PAIRS) != 13 or len(settings) != 232:
        raise RuntimeError("Expected 13 forest pairs and 232 configurations.")
    X, y, features, categorical = data.load_training(args.train)
    plans = data.feature_plans(features)
    folds = list(data.StratifiedKFold(
        n_splits=data.FOLDS, shuffle=True, random_state=data.SEED
    ).split(X, y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, detail, partial = output_paths(args.output_dir)
    completed = read_partial(partial, settings)
    if completed:
        print(f"Resuming from {len(completed)} completed configurations.", flush=True)

    for index in range(len(completed), len(settings)):
        setting = settings[index]
        result = first_round.evaluate(
            setting, X, y, plans[setting["plan"]], categorical, folds
        )
        row = csv_row(result)
        with partial.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=COLUMNS).writerow(row)
        completed.append(row)
        print(
            f"{index + 1:3d}/{len(settings)}  {setting['model']}, "
            f"plan {setting['plan']}, {setting_text(row)}, "
            f"weight={setting['weight_name']}, F1={result['mean_f1']:.4f}",
            flush=True,
        )

    if report.exists() or detail.exists():
        raise FileExistsError("Final result filename is already in use.")
    partial.rename(detail)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(build_summary(completed, len(y), detail.name))
    print(f"Summary: {report}")
    print(f"All configurations: {detail}")


if __name__ == "__main__":
    main()

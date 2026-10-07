import argparse
import csv
from pathlib import Path

import compare_models as data
import compare_models_round1_weighted as models


HERE = Path(__file__).resolve().parent
YES_WEIGHTS = (1.5, 1.75, 2.0, 2.25, 2.5)

COLUMNS = [
    "model", "plan", "C", "trees", "depth", "leaf", "yes_weight",
    "fold_1_f1", "fold_2_f1", "fold_3_f1", "fold_4_f1", "fold_5_f1",
    "mean_f1", "sd_f1", "mean_train_f1", "train_validation_gap",
    "mean_precision", "mean_recall", "mean_accuracy", "seconds",
]


def configurations():
    for yes_weight in YES_WEIGHTS:
        yield {
            "model": "Logistic regression", "plan": "A", "C": 10.0,
            "trees": "", "depth": "", "leaf": "",
            "class_weight": {0: 1, 1: yes_weight},
            "yes_weight": yes_weight,
        }
    for yes_weight in YES_WEIGHTS:
        yield {
            "model": "Random forest", "plan": "D", "C": "",
            "trees": 100, "depth": 20, "leaf": 5,
            "class_weight": {0: 1, 1: yes_weight},
            "yes_weight": yes_weight,
        }


def result_row(result):
    row = {
        "model": result["model"],
        "plan": result["plan"],
        "C": "" if result["C"] == "" else f"{result['C']:g}",
        "trees": result["trees"],
        "depth": result["depth"],
        "leaf": result["leaf"],
        "yes_weight": f"{result['yes_weight']:g}",
    }
    for fold in range(1, data.FOLDS + 1):
        row[f"fold_{fold}_f1"] = f"{result[f'fold_{fold}_f1']:.6f}"
    for metric in (
        "mean_f1", "sd_f1", "mean_train_f1", "mean_precision",
        "mean_recall", "mean_accuracy", "seconds",
    ):
        row[metric] = f"{result[metric]:.6f}"
    row["train_validation_gap"] = (
        f"{result['mean_train_f1'] - result['mean_f1']:.6f}"
    )
    return row


def result_table(rows, model):
    headers = [
        "Yes weight", "Mean F1", "F1 SD", "Train F1",
        "Train-validation gap", "Precision", "Recall", "Accuracy",
    ]
    values = [
        [
            row["yes_weight"],
            f"{float(row['mean_f1']):.4f}",
            f"{float(row['sd_f1']):.4f}",
            f"{float(row['mean_train_f1']):.4f}",
            f"{float(row['train_validation_gap']):.4f}",
            f"{float(row['mean_precision']):.4f}",
            f"{float(row['mean_recall']):.4f}",
            f"{float(row['mean_accuracy']):.4f}",
        ]
        for row in rows if row["model"] == model
    ]
    return models.markdown_table(headers, values)


def build_summary(rows, row_count, detail_name):
    return "\n".join([
        "# Third-round class-weight comparison",
        "",
        models.markdown_table(["Setting", "Value"], [
            ["Source", "train.csv only"],
            ["Rows with a label", f"{row_count:,}"],
            ["Validation", "5 stratified folds, shuffled, seed 42"],
            ["Positive class", "yes"],
            ["Logistic regression", "plan A, C=10"],
            ["Random forest", "plan D, trees=100, depth=20, minimum leaf size=5"],
            ["No class weight", "1"],
            ["Yes class weights", "1.5, 1.75, 2, 2.25, 2.5"],
            ["Main comparison score", "Mean validation F1"],
        ]),
        "",
        "## Logistic regression",
        "",
        result_table(rows, "Logistic regression"),
        "",
        "## Random forest",
        "",
        result_table(rows, "Random forest"),
        "",
        f"Fold scores and run times: [{detail_name}]({detail_name}).",
        "",
    ])


def save_results(output_dir, results, row_count):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        ending = "" if suffix == 1 else f"_{suffix}"
        report = output_dir / f"round3_weight_summary{ending}.md"
        detail = output_dir / f"round3_weight_results{ending}.csv"
        if not report.exists() and not detail.exists():
            break
        suffix += 1

    rows = [result_row(result) for result in results]
    with detail.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(build_summary(rows, row_count, detail.name))
    return report, detail


def main():
    parser = argparse.ArgumentParser(
        description="Compare class weights for both selected classifiers."
    )
    parser.add_argument("--train", type=Path, default=data.DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if not args.train.is_file():
        parser.error(f"Training file not found: {args.train}")

    X, y, features, categorical = data.load_training(args.train)
    plans = data.feature_plans(features)
    folds = list(data.StratifiedKFold(
        n_splits=data.FOLDS, shuffle=True, random_state=data.SEED
    ).split(X, y))

    results = []
    settings = list(configurations())
    if len(settings) != 10:
        raise RuntimeError(f"Expected 10 configurations, found {len(settings)}.")
    for number, setting in enumerate(settings, start=1):
        result = models.evaluate(
            setting, X, y, plans[setting["plan"]], categorical, folds
        )
        results.append(result)
        print(
            f"{number}/{len(settings)}  {setting['model']}, "
            f"yes_weight={setting['yes_weight']:g}, "
            f"F1={result['mean_f1']:.4f}",
            flush=True,
        )

    report, detail = save_results(args.output_dir, results, len(y))
    print(f"Summary: {report}")
    print(f"All results: {detail}")


if __name__ == "__main__":
    main()

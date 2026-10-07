import argparse
import warnings
from pathlib import Path
from time import perf_counter

import compare_models as data
import compare_models_round1_weighted as models

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
LOCAL_TEST = HERE / "dataset" / "test.csv"
DESKTOP_TEST = Path("/Users/jungle/Desktop/NTU/Assignment/mining1/dataset/test.csv")
DEFAULT_TEST = LOCAL_TEST if LOCAL_TEST.is_file() else DESKTOP_TEST

SETTINGS = [
    {
        "model": "Logistic regression", "plan": "A", "C": 10,
        "class_weight": {0: 1, 1: 2.0},
    },
    {
        "model": "Random forest", "plan": "D", "trees": 100,
        "depth": 20, "leaf": 5, "class_weight": {0: 1, 1: 2.5},
    },
]


def evaluate(setting, X_train, y_train, X_test, y_test, columns, categorical):
    pipeline = models.make_pipeline(setting, columns, categorical)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train[columns], y_train)
    fit_seconds = perf_counter() - started
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Logistic regression did not converge on the full training set.")

    predicted = pipeline.predict(X_test[columns])
    positive_index = list(pipeline.named_steps["model"].classes_).index(1)
    probability = pipeline.predict_proba(X_test[columns])[:, positive_index]
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probability)

    return {
        "name": setting["model"],
        "plan": setting["plan"],
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted),
        "f1": f1_score(y_test, predicted),
        "roc_auc": roc_auc_score(y_test, probability),
        "ap": average_precision_score(y_test, probability),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fit_seconds": fit_seconds,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


def make_pr_svg(results, positive_share):
    width, height = 720, 500
    left, right, top, bottom = 75, 35, 70, 65
    plot_width = width - left - right
    plot_height = height - top - bottom

    def point(recall, precision):
        x = left + recall * plot_width
        y = top + (1 - precision) * plot_height
        return f"{x:.1f},{y:.1f}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="75" y="35" font-family="Arial" font-size="20">Precision–recall curve</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x = left + tick * plot_width
        y = top + (1 - tick) * plot_height
        lines += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" '
            'stroke="#e5e7eb"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>',
            f'<text x="{x:.1f}" y="{height - 38}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{tick:g}</text>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{tick:g}</text>',
        ]

    baseline_y = top + (1 - positive_share) * plot_height
    lines.append(
        f'<line x1="{left}" y1="{baseline_y:.1f}" '
        f'x2="{left + plot_width}" y2="{baseline_y:.1f}" '
        'stroke="#888888" stroke-width="1.5" stroke-dasharray="6 5"/>'
    )
    colors = ("#2563eb", "#dc2626")
    for result, color in zip(results, colors):
        points = " ".join(
            point(recall, precision)
            for recall, precision in zip(
                result["recall_curve"][::-1], result["precision_curve"][::-1]
            )
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2" stroke-linejoin="round"/>'
        )

    lines += [
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 8}" '
        'text-anchor="middle" font-family="Arial" font-size="14">Recall</text>',
        f'<text x="20" y="{top + plot_height / 2:.1f}" '
        'transform="rotate(-90 20 '
        f'{top + plot_height / 2:.1f})" text-anchor="middle" '
        'font-family="Arial" font-size="14">Precision</text>',
    ]
    for index, (result, color) in enumerate(zip(results, colors)):
        y = top + 18 + index * 22
        lines += [
            f'<line x1="{left + 15}" y1="{y}" x2="{left + 45}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>',
            f'<text x="{left + 53}" y="{y + 4}" font-family="Arial" font-size="13">'
            f'{result["name"]} (AP {result["ap"]:.3f})</text>',
        ]
    lines += [
        f'<text x="{left + plot_width - 5}" y="{baseline_y - 6:.1f}" '
        'text-anchor="end" font-family="Arial" font-size="11" fill="#666666">'
        f'Yes share: {positive_share:.3f}</text>',
        '</svg>',
    ]
    return "\n".join(lines) + "\n"


def make_report(results, train_count, test_count, positive_count, curve_name):
    lines = [
        "# Final test results", "",
        f"Training rows with a label: {train_count:,}. ",
        f"Test rows with a label: {test_count:,} "
        f"({positive_count:,} yes, {test_count - positive_count:,} no).", "",
        "Predictions use the default 0.5 threshold. Precision, recall and F1 "
        "refer to `yes`. ROC-AUC and AP use predicted probabilities.", "",
        "| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | Fit time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f'| {result["name"]} | {result["plan"]} | '
            + " | ".join(
                f'{result[key]:.4f}'
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "ap")
            )
            + f' | {result["fit_seconds"]:.2f} |'
        )
    for result in results:
        lines += [
            "", f'## {result["name"]}: confusion matrix', "",
            "| Actual / predicted | no | yes |",
            "| --- | ---: | ---: |",
            f'| no | {result["tn"]:,} | {result["fp"]:,} |',
            f'| yes | {result["fn"]:,} | {result["tp"]:,} |',
        ]
    lines += ["", "## Precision–recall curve", "", f"![PR curves]({curve_name})", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the two fixed models on test.csv.")
    parser.add_argument("--train", type=Path, default=data.DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    for path in (args.train, args.test):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    report_path = args.output_dir / "final_test_results.md"
    curve_path = args.output_dir / "final_pr_curves.svg"
    for path in (report_path, curve_path):
        if path.exists():
            parser.error(f"Result already exists; will not repeat or overwrite the test evaluation: {path}")

    X_train, y_train, features, categorical = data.load_training(args.train)
    X_test, y_test, test_features, _ = data.load_training(args.test)
    if set(test_features) != set(features):
        raise ValueError("Train and test predictor columns do not match.")
    if set(y_test) != {0, 1}:
        raise ValueError("Both yes and no labels are needed for ROC-AUC and AP.")

    plans = data.feature_plans(features)
    results = [
        evaluate(setting, X_train, y_train, X_test, y_test,
                 plans[setting["plan"]], categorical)
        for setting in SETTINGS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with curve_path.open("x", encoding="utf-8") as handle:
        handle.write(make_pr_svg(results, float(y_test.mean())))
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(make_report(results, len(y_train), len(y_test),
                                 int(y_test.sum()), curve_path.name))

    print(f"Results: {report_path}")
    print(f"PR curve: {curve_path}")


if __name__ == "__main__":
    main()

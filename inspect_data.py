import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "label"
NUMERIC_COLUMNS = [
    "aptitude_score", "performance_score", "stability_index", "load_ratio",
    "index_weight", "activity_duration", "composite_rank",
]
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA = SCRIPT_DIR / "dataset"
SOURCE_DATA = Path("/Users/jungle/Desktop/NTU/Assignment/mining1/dataset")
DEFAULT_DATA = LOCAL_DATA if LOCAL_DATA.is_dir() else SOURCE_DATA


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def number(value):
    return "n/a" if pd.isna(value) else f"{value:,.3f}"


def names(values):
    return ", ".join(f"`{v}`" for v in values) if values else "None"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path):
    # Preserve category spelling and inspect blanks before numeric conversion.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if TARGET not in raw.columns:
        raise ValueError(f"{path.name} has no '{TARGET}' column.")
    blank = raw.apply(lambda col: col.str.fullmatch(r"\s*", na=False))
    frame = raw.mask(blank)
    numeric = [col for col in NUMERIC_COLUMNS if col in frame.columns]
    categorical = [col for col in frame.columns if col not in numeric + [TARGET]]
    invalid_numeric = 0
    infinite_numeric = 0
    for col in numeric:
        parsed = pd.to_numeric(frame[col], errors="coerce")
        invalid_numeric += int((frame[col].notna() & parsed.isna()).sum())
        infinite_numeric += int(np.isinf(parsed).sum())
        frame[col] = parsed

    text_columns = categorical + [TARGET]
    surrounding_spaces = sum(int((raw[c].ne("") & raw[c].ne(raw[c].str.strip())).sum())
                             for c in text_columns)
    variants = []
    for col in text_columns:
        values = frame[col].dropna().unique()
        groups = {}
        for value in values:
            groups.setdefault(value.strip().casefold(), []).append(value)
        variants.extend((col, spellings) for spellings in groups.values() if len(spellings) > 1)

    features = [c for c in frame.columns if c != TARGET]
    repeated_features = frame.duplicated(subset=features, keep=False)
    if repeated_features.any():
        label_counts = frame.loc[repeated_features].groupby(
            features, dropna=False, observed=True
        )[TARGET].nunique(dropna=True)
        conflict_groups = int(label_counts.gt(1).sum())
    else:
        conflict_groups = 0

    return {
        "frame": frame, "raw": raw, "numeric": numeric, "categorical": categorical,
        "features": features, "blank": blank,
        "invalid_numeric": invalid_numeric, "infinite_numeric": infinite_numeric,
        "surrounding_spaces": surrounding_spaces, "variants": variants,
        "invalid_labels": int((frame[TARGET].notna() & ~frame[TARGET].isin(["yes", "no"])).sum()),
        "duplicates": int(frame.duplicated().sum()),
        "feature_duplicates": int(frame.duplicated(subset=features).sum()),
        "conflicts": conflict_groups,
    }


def build_results(train, test):
    tr, te = train["frame"], test["frame"]
    numeric, categorical = train["numeric"], train["categorical"]
    valid = tr[TARGET].isin(["yes", "no"])
    labelled = tr.loc[valid]
    y = labelled[TARGET].eq("yes").astype(int)
    feature_missing = [int(d["blank"][d["features"]].sum().sum()) for d in (train, test)]
    missing_rows = [int(d["blank"][d["features"]].any(axis=1).sum()) for d in (train, test)]
    parts = ["# Data inspection"]

    parts += ["## Dataset", table(["Item", "Train", "Test"], [
        ["Rows", f"{len(tr):,}", f"{len(te):,}"],
        ["Predictor columns", len(train["features"]), len(test["features"])],
        ["Numeric predictors", len(numeric), len(test["numeric"])],
        ["Categorical predictors", len(categorical), len(test["categorical"])],
        ["Missing predictor cells", *feature_missing],
        ["Rows with a missing predictor", *missing_rows],
        ["Rows with a missing label", int(tr[TARGET].isna().sum()), int(te[TARGET].isna().sum())],
    ])]

    parts += ["### Columns and missing values", table(
        ["Column", "Type", "Train missing", "Test missing"],
        [[f"`{c}`", "Target" if c == TARGET else "Numeric" if c in numeric else "Categorical",
          int(train["blank"][c].sum()), int(test["blank"][c].sum()) if c in te.columns else "Column absent"]
         for c in tr.columns])]

    parts += ["## Data quality", table(["Check", "Train", "Test"], [
        ["Duplicate rows across all columns", train["duplicates"], test["duplicates"]],
        ["Duplicate predictor combinations", train["feature_duplicates"], test["feature_duplicates"]],
        ["Predictor combinations with conflicting labels", train["conflicts"], test["conflicts"]],
        ["Non-empty labels outside yes/no", train["invalid_labels"], test["invalid_labels"]],
        ["Non-numeric entries in numeric columns", train["invalid_numeric"], test["invalid_numeric"]],
        ["Infinite numeric values", train["infinite_numeric"], test["infinite_numeric"]],
        ["Text cells with surrounding whitespace", train["surrounding_spaces"], test["surrounding_spaces"]],
        ["Category groups differing only by case or spaces", len(train["variants"]), len(test["variants"])],
    ])]
    if train["variants"] or test["variants"]:
        variant_rows = [[source, column, ", ".join(spellings)]
                        for source, data in [("Train", train), ("Test", test)]
                        for column, spellings in data["variants"]]
        parts += ["### Spelling variants", table(["File", "Column", "Variants"], variant_rows)]
    placeholders = []
    for col in categorical:
        for value in ["Unknown", "Other"]:
            a, b = int(tr[col].eq(value).sum()), int(te[col].eq(value).sum()) if col in te else 0
            if a or b:
                placeholders.append([f"`{col}`", value, a, b])
    if placeholders:
        parts += ["### Literal Unknown/Other values",
                  table(["Column", "Recorded value", "Train count", "Test count"], placeholders)]

    counts = labelled[TARGET].value_counts()
    test_labelled = te.loc[te[TARGET].isin(["yes", "no"])]
    test_counts = test_labelled[TARGET].value_counts()
    parts += ["## Distributions", "### Labels", table(
        ["Label", "Train count", "Train share", "Test count", "Test share"],
        [[label, f"{int(counts.get(label, 0)):,}", f"{counts.get(label, 0) / len(labelled):.2%}",
          f"{int(test_counts.get(label, 0)):,}",
          f"{test_counts.get(label, 0) / len(test_labelled):.2%}" if len(test_labelled) else "n/a"]
         for label in ["no", "yes"]])]

    stats = []
    for col in numeric:
        s = tr[col].replace([np.inf, -np.inf], np.nan).dropna()
        q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        count = int(((s < lower) | (s > upper)).sum())
        stats.append([f"`{col}`", number(s.min()), number(s.mean()), number(median),
                      number(s.std()), number(s.max()), f"{count:,} ({count/len(s):.2%})"])
    parts += ["### Numeric values in train.csv", table(
        ["Feature", "Min", "Mean", "Median", "Std. dev.", "Max", "Outside 1.5 x IQR"], stats)]
    rows = []
    for col in categorical:
        vc = tr[col].value_counts()
        top = vc.index[0]
        rows.append([f"`{col}`", len(vc), top, f"{int(vc.iloc[0]):,}", f"{vc.iloc[0]/vc.sum():.2%}",
                     int(vc.lt(100).sum())])
    parts += ["### Categorical values in train.csv", table(
        ["Feature", "Categories", "Most common", "Count", "Share", "Categories with <100 records"], rows)]

    parts += ["## Feature relationships", "### Numeric feature pairs"]
    corr = tr[numeric].corr()
    pairs = [(a, b, float(corr.loc[a, b])) for a, b in combinations(numeric, 2)
             if abs(corr.loc[a, b]) >= 0.5]
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    parts.append(table(["Feature 1", "Feature 2", "Pearson r (|r| >= 0.50)"],
                       [[f"`{a}`", f"`{b}`", f"{r:.3f}"] for a, b, r in pairs]))

    label_rows = []
    for col in numeric:
        label_rows.append([f"`{col}`", number(labelled.loc[y.eq(0), col].mean()),
                           number(labelled.loc[y.eq(1), col].mean()), f"{labelled[col].corr(y):.3f}"])
    parts += ["### Numeric features and the label", table(
        ["Feature", "Mean for no", "Mean for yes", "Pearson r with yes=1"], label_rows)]

    category_rows = []
    purity = {}
    for col in categorical:
        ct = pd.crosstab(labelled[col], labelled[TARGET]).reindex(columns=["no", "yes"], fill_value=0)
        purity[col] = float(ct.max(axis=1).sum() / ct.to_numpy().sum())
        rates = ct.assign(n=ct.sum(axis=1))
        rates["rate"] = rates["yes"] / rates["n"]
        eligible = rates.loc[rates["n"].ge(100)].sort_values("rate")
        if eligible.empty:
            continue
        low, high = eligible.index[0], eligible.index[-1]
        category_rows.append([f"`{col}`", low,
                              f"{eligible.loc[low, 'rate']:.2%} (n={int(eligible.loc[low, 'n']):,})", high,
                              f"{eligible.loc[high, 'rate']:.2%} (n={int(eligible.loc[high, 'n']):,})"])
    parts += ["### Categorical features and the label", table(
        ["Feature", "Lowest yes-rate category (n >= 100)", "Yes rate", "Highest yes-rate category (n >= 100)", "Yes rate"],
        category_rows)]

    parts += ["## Train and test compatibility"]
    shared = [c for c in train["features"] if c in test["features"]]
    common_features = tr[shared].drop_duplicates().merge(te[shared].drop_duplicates(), on=shared)
    unseen = {c: sorted(set(te[c].dropna()) - set(tr[c].dropna()))
              for c in categorical if c in te}
    mismatch = [c for c in shared if (c in numeric) != (c in test["numeric"])]
    compatibility = [
        ["Column names and order", "Match" if list(tr.columns) == list(te.columns) else "Differ"],
        ["Numeric/categorical types", "Match" if not mismatch else names(mismatch)],
        ["Categories occurring only in test", names([f"{c}: {v}" for c, v in unseen.items() if v])],
        ["Shared predictor combinations", len(common_features)],
    ]
    outside_ranges = []
    for col in numeric:
        if col not in te:
            continue
        outside = int(((te[col] < tr[col].min()) | (te[col] > tr[col].max())).sum())
        if outside:
            outside_ranges.append(f"{col}: {outside}")
    compatibility.append(["Test numeric values outside observed training ranges", "; ".join(outside_ranges) or "None"])
    parts.append(table(["Check", "Result"], compatibility))

    parts += ["## Uninformative columns and possible leakage"]
    constants = [c for c in train["features"] if tr[c].nunique(dropna=True) <= 1]
    identical = [f"{a} / {b}" for a, b in combinations(train["features"], 2) if tr[a].equals(tr[b])]
    id_like = []
    for col in train["features"]:
        s = tr[col].dropna()
        fraction = s.nunique() / len(s) if len(s) else 0
        if fraction >= 0.95 and (col in categorical or s.is_monotonic_increasing or s.is_monotonic_decreasing):
            id_like.append(col)
    label_copies = [c for c in categorical if tr[c].equals(tr[TARGET])]
    deterministic = [c for c, p in purity.items() if p == 1.0]
    separating = []
    for col in numeric:
        no, yes = labelled.loc[y.eq(0), col].dropna(), labelled.loc[y.eq(1), col].dropna()
        if len(no) and len(yes) and (no.max() < yes.min() or yes.max() < no.min()):
            separating.append(col)
    parts.append(table(["Training-data check", "Result"], [
        ["Constant or entirely missing predictors", names(constants)],
        ["Exactly identical predictor columns", names(identical)],
        ["Possible identifiers under the uniqueness/order check", names(id_like)],
        ["Categorical predictors identical to label", names(label_copies)],
        ["Categorical predictors whose categories perfectly determine the observed label", names(deterministic)],
        ["Numeric predictors separating all observed yes/no values with one threshold", names(separating)],
    ]))
    return "\n\n".join(parts) + "\n"


def save_new_file(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    index = 2
    while True:
        try:
            with candidate.open("x", encoding="utf-8") as handle:
                handle.write(contents)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
            index += 1


def main():
    parser = argparse.ArgumentParser(description="Inspect train/test CSV files and save English data tables.")
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA / "train.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_DATA / "test.csv")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "results" / "data_checks.md")
    args = parser.parse_args()
    for path in [args.train, args.test]:
        if not path.is_file():
            parser.error(f"File not found: {path}. Use --train and --test to provide the CSV paths.")
    fingerprints = [digest(args.train), digest(args.test)]
    train, test = load_data(args.train), load_data(args.test)
    if not train["numeric"] or not train["categorical"]:
        parser.error("This script expects the numeric and categorical columns in the IN6227 dataset.")
    if set(train["features"]) != set(test["features"]):
        parser.error("Predictor columns differ between train and test. Check the CSV headers.")
    if not train["frame"][TARGET].isin(["yes", "no"]).any():
        parser.error("No valid yes/no labels were found in the training file.")
    contents = build_results(train, test)
    if fingerprints != [digest(args.train), digest(args.test)]:
        raise RuntimeError("A source CSV changed during inspection. Run again on stable input files.")
    output = save_new_file(args.output.resolve(), contents)
    print(f"Training rows: {len(train['frame']):,}; test rows: {len(test['frame']):,}")
    print(f"Predictors: {len(train['numeric'])} numeric, {len(train['categorical'])} categorical")
    print(f"English results saved to: {output}")


if __name__ == "__main__":
    main()

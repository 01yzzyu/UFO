"""Aggregate scored instances into result tables.

Produces, per model and protocol:
  - per-task accuracy
  - per-category accuracy (mean over that category's instances)
  - overall average

Outputs both a tidy CSV and a LaTeX table laid out like the paper's main table
(State Determination / Reconstruction / Augmentation x Direct/Textual/Visual/Joint).
"""

import csv

from .config import (
    CATEGORIES,
    CATEGORY_DISPLAY,
    PROTOCOLS,
    TASK_TO_CATEGORY,
)


def _tag(model_id):
    return model_id.replace("/", "_").replace(":", "_").replace("\\", "_")


def accuracy_by_task(instances, model_id, protocols=None):
    """Return {protocol: {task: (acc, n)}} for one model."""
    protocols = protocols or list(PROTOCOLS)
    tag = _tag(model_id)
    acc = {p: {t: [0.0, 0] for t in TASK_TO_CATEGORY} for p in protocols}
    for it in instances:
        task = it.get("task")
        if task not in TASK_TO_CATEGORY:
            continue
        for p in protocols:
            sk = f"score_{p}_{tag}"
            if sk in it:
                acc[p][task][0] += float(it[sk])
                acc[p][task][1] += 1
    out = {}
    for p in protocols:
        out[p] = {t: (s / n if n else None, n) for t, (s, n) in acc[p].items()}
    return out


def accuracy_by_category(instances, model_id, protocols=None):
    """Return {protocol: {category: (acc, n)}} computed over instances."""
    protocols = protocols or list(PROTOCOLS)
    tag = _tag(model_id)
    agg = {p: {c: [0.0, 0] for c in CATEGORIES} for p in protocols}
    for it in instances:
        cat = it.get("category")
        if cat not in CATEGORIES:
            continue
        for p in protocols:
            sk = f"score_{p}_{tag}"
            if sk in it:
                agg[p][cat][0] += float(it[sk])
                agg[p][cat][1] += 1
    out = {}
    for p in protocols:
        out[p] = {c: (s / n if n else None, n) for c, (s, n) in agg[p].items()}
    return out


def overall_accuracy(instances, model_id, protocols=None):
    protocols = protocols or list(PROTOCOLS)
    tag = _tag(model_id)
    out = {}
    for p in protocols:
        s, n = 0.0, 0
        for it in instances:
            sk = f"score_{p}_{tag}"
            if sk in it:
                s += float(it[sk])
                n += 1
        out[p] = (s / n if n else None, n)
    return out


def _fmt(acc):
    return f"{acc * 100:.2f}" if acc is not None else "-"


def write_csv(path, per_model_task, protocols):
    """per_model_task: {model_name: {protocol: {task: (acc, n)}}}"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["model", "protocol"] + list(TASK_TO_CATEGORY.keys()) + ["overall"]
        w.writerow(header)
        for model, by_proto in per_model_task.items():
            for p in protocols:
                row = [model, p]
                vals = []
                for t in TASK_TO_CATEGORY:
                    acc, _ = by_proto.get(p, {}).get(t, (None, 0))
                    row.append(_fmt(acc))
                    if acc is not None:
                        vals.append(acc)
                row.append(_fmt(sum(vals) / len(vals) if vals else None))
                w.writerow(row)


def to_latex_main(per_model_category, models_in_order, protocols=None):
    """Paper-style main table: categories x protocols, one row per model.

    per_model_category: {model_name: {protocol: {category: (acc, n)}}}
    """
    protocols = protocols or list(PROTOCOLS)
    cols_per_cat = len(protocols)
    n_cat = len(CATEGORIES)

    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{",
             r"\begin{tabular}{l" + "c" * (cols_per_cat * (n_cat + 1)) + "}",
             r"\toprule"]

    # header row 1: category spans + Average
    head1 = [r"\multirow{2}{*}{\textbf{Model}}"]
    for c in CATEGORIES:
        head1.append(r"\multicolumn{%d}{c}{\textbf{%s}}" % (cols_per_cat,
                                                            CATEGORY_DISPLAY[c]))
    head1.append(r"\multicolumn{%d}{c}{\textbf{Average}}" % cols_per_cat)
    lines.append(" & ".join(head1) + r" \\")

    # header row 2: protocol labels repeated
    proto_labels = " & ".join(r"\textit{%s}" % p.capitalize() for p in protocols)
    lines.append(" & " + " & ".join([proto_labels] * (n_cat + 1)) + r" \\")
    lines.append(r"\midrule")

    for model in models_in_order:
        by_proto = per_model_category.get(model, {})
        cells = [model]
        # categories
        for c in CATEGORIES:
            for p in protocols:
                acc, _ = by_proto.get(p, {}).get(c, (None, 0))
                cells.append(_fmt(acc))
        # average across categories
        for p in protocols:
            accs = []
            for c in CATEGORIES:
                acc, _ = by_proto.get(p, {}).get(c, (None, 0))
                if acc is not None:
                    accs.append(acc)
            cells.append(_fmt(sum(accs) / len(accs) if accs else None))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"}",
              r"\caption{UFO main results: accuracy (\%) across state "
              r"determination, reconstruction, and augmentation under the "
              r"direct, textual, visual, and joint protocols.}",
              r"\label{tab:ufo_main}", r"\end{table*}"]
    return "\n".join(lines)


def summary_markdown(per_model_overall, models_in_order, protocols=None):
    """A compact, human-readable Markdown table of overall accuracy.

    per_model_overall: {model_name: {protocol: (acc, n)}}
    Returns a Markdown string (also nice to print to the console).
    """
    protocols = protocols or list(PROTOCOLS)
    header = "| Model | " + " | ".join(p.capitalize() for p in protocols) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(protocols)) + " |"
    rows = [header, sep]
    for model in models_in_order:
        by_proto = per_model_overall.get(model, {})
        cells = [model]
        for p in protocols:
            acc, _ = by_proto.get(p, (None, 0))
            cells.append(_fmt(acc))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)

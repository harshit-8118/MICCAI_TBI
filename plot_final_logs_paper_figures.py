from __future__ import annotations

import argparse
import csv
import math
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependency. Run this script in the same Python environment "
        "used for visualize_test_evaluations.py, or install matplotlib and numpy."
    ) from exc

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependency. Run this script in the same Python environment "
        "used for visualize_test_evaluations.py, or install matplotlib and numpy."
    ) from exc


NUMERIC_COLUMNS = [
    "threshold",
    "mincc",
    "test loss",
    "dice all",
    "hd95 all",
    "assd all",
    "dice positive",
    "hd95 positive",
    "assd positive",
    "dice micro",
    "hd95 micro",
    "assd micro",
    "dice small",
    "hd95 small",
    "assd small",
    "dice large",
    "hd95 large",
    "assd large",
    "dice empty",
    "hd95 empty",
    "assd empty",
    "missed positive",
    "false positive",
]

DICE_GROUPS = [
    ("all", "dice all"),
    ("positive", "dice positive"),
    ("micro", "dice micro"),
    ("small", "dice small"),
    ("large", "dice large"),
    ("empty", "dice empty"),
]

HD95_GROUPS = [
    ("all", "hd95 all"),
    ("positive", "hd95 positive"),
    ("micro", "hd95 micro"),
    ("small", "hd95 small"),
    ("large", "hd95 large"),
    ("empty", "hd95 empty"),
]

ASSD_GROUPS = [
    ("all", "assd all"),
    ("positive", "assd positive"),
    ("micro", "assd micro"),
    ("small", "assd small"),
    ("large", "assd large"),
    ("empty", "assd empty"),
]


@dataclass
class FinalLogRow:
    benchmark: str
    values: dict[str, float]
    row_index: int
    short_name: str
    family: str
    setting_label: str


def _to_float(value: str) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def _rank(values: list[float], higher_is_better: bool) -> list[int]:
    valid = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    valid.sort(key=lambda item: item[1], reverse=higher_is_better)
    ranks = [len(values) + 1 for _ in values]
    last_value = None
    last_rank = 0
    for position, (index, value) in enumerate(valid, start=1):
        if last_value is not None and math.isclose(value, last_value, rel_tol=1e-12, abs_tol=1e-12):
            rank = last_rank
        else:
            rank = position
            last_value = value
            last_rank = rank
        ranks[index] = rank
    return ranks


def _short_benchmark_name(name: str) -> str:
    lowered = name.lower()
    if "single_ddp" in lowered and "no_tta" in lowered:
        return "ddp_kpcyjb66 no-TTA"
    if "single_ddp" in lowered and "tta" in lowered:
        return "ddp_kpcyjb66 TTA"
    if "best_val_f0" in lowered and "no_tta" in lowered:
        return "e5ohnz5w no-TTA"
    if "best_val_f0" in lowered and "tta" in lowered:
        return "e5ohnz5w TTA"
    if "ensemble_2_models" in lowered:
        if "kpcyjb66_no_tta" in lowered:
            ddp_mode = "no-TTA"
        elif "kpcyjb66_tta" in lowered:
            ddp_mode = "TTA"
        else:
            ddp_mode = "?"
        if "e5ohnz5w_no_tta" in lowered:
            e5_mode = "no-TTA"
        elif "e5ohnz5w_tta" in lowered:
            e5_mode = "TTA"
        else:
            e5_mode = "?"
        return f"Hybrid ddp {ddp_mode} / e5 {e5_mode}"
    if "tta_ddp" in lowered and "kpcyjb66_tta" in lowered:
        return "Hybrid ddp TTA / e5 TTA"
    if "tta_ddp" in lowered and "no_tta" in lowered:
        return "Hybrid ddp no-TTA / e5 TTA"
    return name.replace("_", " ")


def _model_family(short_name: str) -> str:
    if short_name.startswith("ddp_kpcyjb66"):
        return "ddp_kpcyjb66"
    if short_name.startswith("e5ohnz5w"):
        return "e5ohnz5w"
    if short_name.startswith("Hybrid ddp no-TTA / e5 no-TTA"):
        return "Hybrid ddp no-TTA / e5 no-TTA"
    if short_name.startswith("Hybrid ddp no-TTA / e5 TTA"):
        return "Hybrid ddp no-TTA / e5 TTA"
    if short_name.startswith("Hybrid ddp TTA / e5 no-TTA"):
        return "Hybrid ddp TTA / e5 no-TTA"
    if short_name.startswith("Hybrid ddp TTA / e5 TTA"):
        return "Hybrid ddp TTA / e5 TTA"
    return "Other"


def _setting_label(short_name: str, values: dict[str, float]) -> str:
    threshold = _fmt(values.get("threshold", float("nan")), 3)
    mincc = _fmt(values.get("mincc", float("nan")), 0)
    return f"{short_name}\nthr={threshold}, mincc={mincc}"


def _wrap_labels(labels: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for label in labels:
        parts = []
        for line in label.splitlines():
            parts.extend(textwrap.wrap(line, width=width, break_long_words=False) or [""])
        wrapped.append("\n".join(parts))
    return wrapped


def read_final_logs(path: Path) -> list[FinalLogRow]:
    rows: list[FinalLogRow] = []
    current_benchmark = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_index, raw in enumerate(reader, start=2):
            benchmark = str(raw.get("BENCHMARKS", "")).strip()
            if benchmark:
                current_benchmark = benchmark
            if not str(raw.get("threshold", "")).strip():
                continue
            values = {column: _to_float(raw.get(column, "")) for column in NUMERIC_COLUMNS}
            short_name = _short_benchmark_name(current_benchmark)
            family = _model_family(short_name)
            rows.append(
                FinalLogRow(
                    benchmark=current_benchmark,
                    values=values,
                    row_index=raw_index,
                    short_name=short_name,
                    family=family,
                    setting_label=_setting_label(short_name, values),
                )
            )
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def add_ranks(rows: list[FinalLogRow]) -> None:
    rank_specs = [
        ("positive", "dice positive", "hd95 positive", "assd positive"),
        ("all", "dice all", "hd95 all", "assd all"),
    ]
    for prefix, dice_key, hd95_key, assd_key in rank_specs:
        dice_ranks = _rank([row.values[dice_key] for row in rows], higher_is_better=True)
        hd95_ranks = _rank([row.values[hd95_key] for row in rows], higher_is_better=False)
        assd_ranks = _rank([row.values[assd_key] for row in rows], higher_is_better=False)
        for row, dice_rank, hd95_rank, assd_rank in zip(rows, dice_ranks, hd95_ranks, assd_ranks):
            row.values[f"{prefix} dice rank"] = float(dice_rank)
            row.values[f"{prefix} hd95 rank"] = float(hd95_rank)
            row.values[f"{prefix} assd rank"] = float(assd_rank)
            row.values[f"{prefix} rank sum"] = float(dice_rank + hd95_rank + assd_rank)


def _rank_sort_key(row: FinalLogRow) -> tuple[float, float, float, float, float, float]:
    return (
        row.values["positive rank sum"],
        row.values["missed positive"],
        row.values["false positive"],
        -row.values["dice positive"],
        row.values["hd95 positive"],
        row.values["assd positive"],
    )


def _best_rows_by_benchmark(rows: list[FinalLogRow]) -> list[FinalLogRow]:
    best_by_benchmark: dict[str, FinalLogRow] = {}
    for row in rows:
        current = best_by_benchmark.get(row.benchmark)
        if current is None or _rank_sort_key(row) < _rank_sort_key(current):
            best_by_benchmark[row.benchmark] = row
    preferred_order = [
        "ddp_kpcyjb66 no-TTA",
        "ddp_kpcyjb66 TTA",
        "e5ohnz5w no-TTA",
        "e5ohnz5w TTA",
        "Hybrid ddp no-TTA / e5 no-TTA",
        "Hybrid ddp no-TTA / e5 TTA",
        "Hybrid ddp TTA / e5 no-TTA",
        "Hybrid ddp TTA / e5 TTA",
    ]
    return sorted(
        best_by_benchmark.values(),
        key=lambda row: preferred_order.index(row.short_name) if row.short_name in preferred_order else len(preferred_order),
    )


def _model_type(row: FinalLogRow) -> str:
    return "Ensemble" if "Ensemble" in row.short_name or "Hybrid" in row.short_name else "Single"


def _tta_mode(row: FinalLogRow) -> str:
    if "ddp no-TTA / e5 no-TTA" in row.short_name:
        return "none"
    if "ddp no-TTA / e5 TTA" in row.short_name:
        return "mixed"
    if "ddp TTA / e5 no-TTA" in row.short_name:
        return "mixed"
    if "ddp TTA / e5 TTA" in row.short_name:
        return "tta"
    return "tta" if row.short_name.endswith(" TTA") else "none"


def write_clean_csv(rows: list[FinalLogRow], output_path: Path) -> None:
    fieldnames = [
        "row_index",
        "benchmark",
        "short_name",
        "family",
        "setting_label",
        *NUMERIC_COLUMNS,
        "positive dice rank",
        "positive hd95 rank",
        "positive assd rank",
        "positive rank sum",
        "all dice rank",
        "all hd95 rank",
        "all assd rank",
        "all rank sum",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "row_index": row.row_index,
                "benchmark": row.benchmark,
                "short_name": row.short_name,
                "family": row.family,
                "setting_label": row.setting_label.replace("\n", " | "),
            }
            out.update(row.values)
            writer.writerow(out)


def _metric_score(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    finite = np.isfinite(values)
    scores = np.full(values.shape, np.nan, dtype=float)
    if not finite.any():
        return scores
    low = np.nanmin(values)
    high = np.nanmax(values)
    if math.isclose(float(low), float(high), rel_tol=1e-12, abs_tol=1e-12):
        scores[finite] = 0.5
        return scores
    if higher_is_better:
        scores[finite] = (values[finite] - low) / (high - low)
    else:
        scores[finite] = (high - values[finite]) / (high - low)
    return scores


def _plot_metric_heatmap(
    rows: list[FinalLogRow],
    metrics: list[tuple[str, str, bool, int]],
    output_path: Path,
    title: str,
    label_width: int,
) -> None:
    labels = _wrap_labels([row.setting_label for row in rows], label_width)
    raw = np.array([[row.values[key] for _, key, _, _ in metrics] for row in rows], dtype=float)
    score_columns = []
    for col_index, (_, _, higher_is_better, _) in enumerate(metrics):
        score_columns.append(_metric_score(raw[:, col_index], higher_is_better))
    scores = np.vstack(score_columns).T

    height = max(5.0, 0.62 * len(rows) + 1.7)
    width = max(9.0, 1.25 * len(metrics) + 5.0)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(scores, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for label, _, _, _ in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="both", length=0)
    for row_index in range(raw.shape[0]):
        for col_index in range(raw.shape[1]):
            value = raw[row_index, col_index]
            if not math.isfinite(value):
                text = ""
            else:
                digits = metrics[col_index][3]
                text = _fmt(value, digits)
            color = "white" if scores[row_index, col_index] >= 0.68 else "#222222"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("Darker cells are better within the shown rows")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="relative goodness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_summary(rows: list[FinalLogRow], output_dir: Path, top_n: int, label_width: int) -> None:
    selected = sorted(rows, key=_rank_sort_key)[:top_n]
    metrics = [
        ("Dice+", "dice positive", True, 3),
        ("HD95+", "hd95 positive", False, 2),
        ("ASSD+", "assd positive", False, 2),
        ("Dice empty", "dice empty", True, 3),
        ("Missed+", "missed positive", False, 0),
        ("Empty FP", "false positive", False, 0),
        ("Rank sum", "positive rank sum", False, 0),
    ]
    _plot_metric_heatmap(
        selected,
        metrics,
        output_dir / "plot_1_rank_summary_positive_metrics.png",
        "Global comparison by challenge-style positive metrics",
        label_width,
    )


def plot_tradeoff_scatter(rows: list[FinalLogRow], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    markers = {
        "ddp_kpcyjb66": "o",
        "e5ohnz5w": "s",
        "Hybrid ddp no-TTA / e5 no-TTA": "D",
        "Hybrid ddp no-TTA / e5 TTA": "v",
        "Hybrid ddp TTA / e5 no-TTA": "P",
        "Hybrid ddp TTA / e5 TTA": "^",
        "Other": "X",
    }
    colors_by_family = {
        "ddp_kpcyjb66": "#4c78a8",
        "e5ohnz5w": "#f58518",
        "Hybrid ddp no-TTA / e5 no-TTA": "#54a24b",
        "Hybrid ddp no-TTA / e5 TTA": "#ff9da6",
        "Hybrid ddp TTA / e5 no-TTA": "#e45756",
        "Hybrid ddp TTA / e5 TTA": "#b279a2",
        "Other": "#79706e",
    }
    false_positives = np.array([row.values["false positive"] for row in rows], dtype=float)
    sizes = 60 + 24 * false_positives
    families = [
        "ddp_kpcyjb66",
        "e5ohnz5w",
        "Hybrid ddp no-TTA / e5 no-TTA",
        "Hybrid ddp no-TTA / e5 TTA",
        "Hybrid ddp TTA / e5 no-TTA",
        "Hybrid ddp TTA / e5 TTA",
        "Other",
    ]
    for family in families:
        group = [index for index, row in enumerate(rows) if row.family == family]
        if not group:
            continue
        missed = np.array([rows[index].values["missed positive"] for index in group], dtype=float)
        ax.scatter(
            [rows[index].values["assd positive"] for index in group],
            [rows[index].values["dice positive"] for index in group],
            s=[sizes[index] for index in group],
            color=colors_by_family.get(family, "#79706e"),
            marker=markers.get(family, "X"),
            edgecolor="black",
            linewidths=0.7 + 0.25 * missed,
            alpha=0.82,
            label=family,
        )

    x_values = np.array([row.values["assd positive"] for row in rows], dtype=float)
    y_values = np.array([row.values["dice positive"] for row in rows], dtype=float)
    x_pad = max(1.0, 0.12 * (float(np.nanmax(x_values)) - float(np.nanmin(x_values))))
    y_pad = max(0.004, 0.18 * (float(np.nanmax(y_values)) - float(np.nanmin(y_values))))
    ax.set_xlim(float(np.nanmin(x_values)) - x_pad, float(np.nanmax(x_values)) + x_pad)
    ax.set_ylim(float(np.nanmin(y_values)) - y_pad, float(np.nanmax(y_values)) + y_pad)

    ranked_rows = sorted(rows, key=_rank_sort_key)
    rank_by_identity = {id(row): index for index, row in enumerate(ranked_rows, start=1)}
    x_mid = float(np.nanmean(ax.get_xlim()))
    y_mid = float(np.nanmean(ax.get_ylim()))
    for row in _best_rows_by_benchmark(rows):
        x = row.values["assd positive"]
        y = row.values["dice positive"]
        x_offset = -72 if x > x_mid else 10
        y_offset = -42 if y > y_mid else 10
        ha = "right" if x > x_mid else "left"
        va = "top" if y > y_mid else "bottom"
        ax.annotate(
            f"#{rank_by_identity[id(row)]} {row.short_name}\n"
            f"thr={_fmt(row.values['threshold'], 3)}, mincc={_fmt(row.values['mincc'], 0)}"
            f"\nHD95+={_fmt(row.values['hd95 positive'], 1)}",
            xy=(x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7,
            ha=ha,
            va=va,
            arrowprops={"arrowstyle": "-", "lw": 0.45, "alpha": 0.45},
        )
    ax.set_title("Global positive-lesion tradeoff: Dice vs ASSD", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("ASSD on lesion-containing scans (lower is better)")
    ax.set_ylabel("Dice on lesion-containing scans (higher is better)")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(loc="upper right", title="Model family", frameon=True)
    legend_handles = legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles
    for handle in legend_handles:
        handle.set_sizes([70])
    ax.text(
        0.99,
        0.02,
        "Marker size = empty false positives; thicker border = more missed positives",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    for path in [
        output_dir / "plot_2_positive_tradeoff_scatter.png",
        output_dir / "plot_2_positive_dice_vs_assd_scatter.png",
    ]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dice_hd95_all_scatter(rows: list[FinalLogRow], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    colors = {"Single": "#4c78a8", "Ensemble": "#e45756"}
    markers = {"none": "o", "tta": "^", "mixed": "P"}
    for model_type in ["Single", "Ensemble"]:
        for tta_mode in ["none", "tta", "mixed"]:
            group = [row for row in rows if _model_type(row) == model_type and _tta_mode(row) == tta_mode]
            if not group:
                continue
            ax.scatter(
                [row.values["dice all"] for row in group],
                [row.values["hd95 all"] for row in group],
                s=[70 + 24 * row.values["false positive"] for row in group],
                color=colors[model_type],
                marker=markers[tta_mode],
                edgecolor="black",
                linewidths=[0.8 + 0.25 * row.values["missed positive"] for row in group],
                alpha=0.82,
                label=f"{model_type}, {tta_mode}",
            )

    ranked_rows = sorted(rows, key=lambda row: (row.values["all rank sum"], row.values["missed positive"], row.values["false positive"]))
    rank_by_identity = {id(row): index for index, row in enumerate(ranked_rows, start=1)}
    x_values = np.array([row.values["dice all"] for row in rows], dtype=float)
    y_values = np.array([row.values["hd95 all"] for row in rows], dtype=float)
    x_pad = max(0.003, 0.10 * (float(np.nanmax(x_values)) - float(np.nanmin(x_values))))
    y_pad = max(1.0, 0.10 * (float(np.nanmax(y_values)) - float(np.nanmin(y_values))))
    ax.set_xlim(float(np.nanmin(x_values)) - x_pad, float(np.nanmax(x_values)) + x_pad)
    ax.set_ylim(float(np.nanmin(y_values)) - y_pad, float(np.nanmax(y_values)) + y_pad)
    x_mid = float(np.nanmean(ax.get_xlim()))
    y_mid = float(np.nanmean(ax.get_ylim()))
    for row in _best_rows_by_benchmark(rows):
        x = row.values["dice all"]
        y = row.values["hd95 all"]
        ax.annotate(
            f"#{rank_by_identity[id(row)]} {row.short_name}\n"
            f"thr={_fmt(row.values['threshold'], 3)}, mincc={_fmt(row.values['mincc'], 0)}",
            xy=(x, y),
            xytext=(-72 if x > x_mid else 10, 10 if y > y_mid else -36),
            textcoords="offset points",
            fontsize=7,
            ha="right" if x > x_mid else "left",
            va="bottom" if y > y_mid else "top",
            arrowprops={"arrowstyle": "-", "lw": 0.45, "alpha": 0.45},
        )
    ax.set_title("Dice All vs HD95 All: overlap versus boundary error", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dice All (higher is better)")
    ax.set_ylabel("HD95 All (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", title="Model type, TTA", frameon=True)
    ax.text(0.01, 0.02, "Best direction: right and down", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    fig.tight_layout()
    for path in [
        output_dir / "plot_2b_dice_all_vs_hd95_all_scatter.png",
        output_dir / "plot_2b_positive_dice_vs_hd95_scatter.png",
    ]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_contributions(rows: list[FinalLogRow], output_dir: Path, top_n: int, label_width: int) -> None:
    selected = sorted(rows, key=lambda row: row.values["positive rank sum"])[:top_n]
    labels = _wrap_labels([row.setting_label for row in selected], label_width)
    y = np.arange(len(selected))
    dice = np.array([row.values["positive dice rank"] for row in selected], dtype=float)
    hd95 = np.array([row.values["positive hd95 rank"] for row in selected], dtype=float)
    assd = np.array([row.values["positive assd rank"] for row in selected], dtype=float)
    height = max(5.0, 0.55 * len(selected) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(y, dice, color="#4c78a8", label="Dice rank")
    ax.barh(y, hd95, left=dice, color="#f58518", label="HD95 rank")
    ax.barh(y, assd, left=dice + hd95, color="#54a24b", label="ASSD rank")
    for index, row in enumerate(selected):
        total = row.values["positive rank sum"]
        ax.text(total + 0.35, index, f"sum={_fmt(total, 0)}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Global rank contribution on lesion-containing scans", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Lower total rank is better")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_3_rank_contributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_panels(rows: list[FinalLogRow], output_dir: Path, top_n: int, label_width: int) -> None:
    selected = sorted(rows, key=_rank_sort_key)[:top_n]
    panel_specs = [
        ("Dice by lesion group", DICE_GROUPS, "Blues", True, output_dir / "plot_4a_dice_groups.png"),
        ("HD95 by lesion group (lower is better)", HD95_GROUPS, "YlOrRd_r", False, output_dir / "plot_4b_hd95_groups.png"),
        ("ASSD by lesion group (lower is better)", ASSD_GROUPS, "YlOrRd_r", False, output_dir / "plot_4c_assd_groups.png"),
    ]
    for title, groups, cmap, higher_is_better, path in panel_specs:
        labels = _wrap_labels([row.setting_label for row in selected], label_width)
        data = np.array([[row.values[column] for _, column in groups] for row in selected], dtype=float)
        height = max(5.0, 0.62 * len(selected) + 1.7)
        fig, ax = plt.subplots(figsize=(10.0, height))
        if higher_is_better:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, max(1.0, float(np.nanpercentile(data, 95)))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([label for label, _ in groups], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(selected)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.tick_params(axis="both", length=0)
        threshold = np.nanmean(data)
        for row_index in range(data.shape[0]):
            for col_index in range(data.shape[1]):
                value = data[row_index, col_index]
                color = "white" if value > threshold and not higher_is_better else "#222222"
                if higher_is_better and value > 0.72:
                    color = "white"
                ax.text(col_index, row_index, _fmt(value, 3 if higher_is_better else 2), ha="center", va="center", fontsize=8, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Dice" if higher_is_better else "distance")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_best_per_benchmark(rows: list[FinalLogRow], output_dir: Path, label_width: int) -> None:
    best_by_benchmark: dict[str, FinalLogRow] = {}
    for row in rows:
        current = best_by_benchmark.get(row.benchmark)
        if current is None or _rank_sort_key(row) < _rank_sort_key(current):
            best_by_benchmark[row.benchmark] = row
    selected = sorted(best_by_benchmark.values(), key=_rank_sort_key)
    labels = _wrap_labels([row.setting_label for row in selected], label_width)
    x = np.arange(len(selected))
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 1.2 * len(selected)), 8.0), sharex=True)
    axes[0].bar(x - 0.18, [row.values["dice positive"] for row in selected], width=0.36, label="Dice positive", color="#4c78a8")
    axes[0].bar(x + 0.18, [row.values["dice empty"] for row in selected], width=0.36, label="Dice empty", color="#72b7b2")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Dice")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("Best setting from each benchmark block", fontsize=14, fontweight="bold", pad=12)

    axes[1].bar(x - 0.2, [row.values["hd95 positive"] for row in selected], width=0.2, label="HD95 positive", color="#f58518")
    axes[1].bar(x, [row.values["assd positive"] for row in selected], width=0.2, label="ASSD positive", color="#e45756")
    axes[1].bar(x + 0.2, [row.values["false positive"] for row in selected], width=0.2, label="Empty FP", color="#54a24b")
    axes[1].set_ylabel("Distance / count")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_5_best_per_benchmark_block.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_variant_summary(rows: list[FinalLogRow], output_dir: Path, label_width: int) -> None:
    """One compact paper-facing comparison after removing redundant thresholds."""
    best_by_benchmark: dict[str, FinalLogRow] = {}
    for row in rows:
        current = best_by_benchmark.get(row.benchmark)
        if current is None or _rank_sort_key(row) < _rank_sort_key(current):
            best_by_benchmark[row.benchmark] = row

    preferred_order = [
        "ddp_kpcyjb66 no-TTA",
        "ddp_kpcyjb66 TTA",
        "e5ohnz5w no-TTA",
        "e5ohnz5w TTA",
        "Hybrid ddp no-TTA / e5 no-TTA",
        "Hybrid ddp no-TTA / e5 TTA",
        "Hybrid ddp TTA / e5 no-TTA",
        "Hybrid ddp TTA / e5 TTA",
    ]
    selected = sorted(
        best_by_benchmark.values(),
        key=lambda row: preferred_order.index(row.short_name) if row.short_name in preferred_order else len(preferred_order),
    )
    labels = _wrap_labels([row.setting_label for row in selected], label_width)
    x = np.arange(len(selected))

    fig, axes = plt.subplots(3, 1, figsize=(max(10.0, 1.35 * len(selected)), 10.5), sharex=True)
    axes[0].bar(x, [row.values["dice positive"] for row in selected], color="#4c78a8", width=0.62)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Dice+")
    axes[0].set_title("Best representative setting per checkpoint/TTA family", fontsize=14, fontweight="bold", pad=12)
    axes[0].grid(axis="y", alpha=0.25)
    for idx, row in enumerate(selected):
        axes[0].text(idx, row.values["dice positive"] + 0.015, _fmt(row.values["dice positive"], 3), ha="center", va="bottom", fontsize=8)

    axes[1].bar(x - 0.18, [row.values["hd95 positive"] for row in selected], width=0.36, color="#f58518", label="HD95+")
    axes[1].bar(x + 0.18, [row.values["assd positive"] for row in selected], width=0.36, color="#e45756", label="ASSD+")
    axes[1].set_ylabel("Distance")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].bar(x - 0.18, [row.values["missed positive"] for row in selected], width=0.36, color="#b279a2", label="Missed positive")
    axes[2].bar(x + 0.18, [row.values["false positive"] for row in selected], width=0.36, color="#54a24b", label="Empty false positive")
    axes[2].set_ylabel("Cases")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(loc="upper right")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "plot_6_model_tta_ensemble_representatives.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_headline_grouped_bars(rows: list[FinalLogRow], output_dir: Path, label_width: int) -> None:
    selected = _best_rows_by_benchmark(rows)
    labels = _wrap_labels([row.short_name for row in selected], label_width)
    x = np.arange(len(selected))
    width = 0.25
    metrics = [
        ("Dice All", "dice all", "#4c78a8"),
        ("Dice Positive", "dice positive", "#f58518"),
        ("Dice Empty", "dice empty", "#54a24b"),
    ]
    fig, ax = plt.subplots(figsize=(max(10.0, 1.35 * len(selected)), 5.8))
    for offset, (label, key, color) in zip([-width, 0, width], metrics):
        values = [row.values[key] for row in selected]
        ax.bar(x + offset, values, width=width, label=label, color=color)
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.012, _fmt(value, 3), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title("Headline comparison: overlap metrics by model family", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_0_headline_grouped_dice_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_mincc_heatmap(
    rows: list[FinalLogRow],
    output_path: Path,
    metric_key: str,
    title: str,
    cmap: str,
    higher_is_better: bool,
    digits: int,
) -> None:
    benchmarks = []
    for row in rows:
        if row.benchmark not in benchmarks:
            benchmarks.append(row.benchmark)
    thresholds = sorted({row.values["threshold"] for row in rows if math.isfinite(row.values["threshold"])})
    minccs = sorted({row.values["mincc"] for row in rows if math.isfinite(row.values["mincc"])})
    ncols = 2
    nrows = math.ceil(len(benchmarks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, max(4.0, 2.7 * nrows)), squeeze=False)
    all_values = np.array([row.values[metric_key] for row in rows], dtype=float)
    if higher_is_better:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    for ax, benchmark in zip(axes.flat, benchmarks):
        data = np.full((len(minccs), len(thresholds)), np.nan, dtype=float)
        for row in rows:
            if row.benchmark != benchmark:
                continue
            y = minccs.index(row.values["mincc"])
            x = thresholds.index(row.values["threshold"])
            data[y, x] = row.values[metric_key]
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(_short_benchmark_name(benchmark), fontsize=9, fontweight="bold")
        ax.set_xticks(np.arange(len(thresholds)))
        ax.set_xticklabels([_fmt(v, 2) for v in thresholds], fontsize=8)
        ax.set_yticks(np.arange(len(minccs)))
        ax.set_yticklabels([_fmt(v, 0) for v in minccs], fontsize=8)
        ax.set_xlabel("threshold")
        ax.set_ylabel("mincc")
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                value = data[y, x]
                if math.isfinite(value):
                    ax.text(x, y, _fmt(value, digits), ha="center", va="center", fontsize=7)
    for ax in axes.flat[len(benchmarks) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.10, top=0.90, wspace=0.35, hspace=0.55)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_mincc_heatmaps(rows: list[FinalLogRow], output_dir: Path) -> None:
    _plot_threshold_mincc_heatmap(
        rows,
        output_dir / "plot_7_threshold_mincc_dice_all_heatmaps.png",
        "dice all",
        "Sparse threshold x mincc sensitivity: Dice All",
        "Blues",
        True,
        3,
    )
    _plot_threshold_mincc_heatmap(
        rows,
        output_dir / "plot_8_threshold_mincc_false_positive_heatmaps.png",
        "false positive",
        "Sparse threshold x mincc sensitivity: empty false positives",
        "YlOrRd",
        False,
        0,
    )


def plot_error_tradeoff_lines(rows: list[FinalLogRow], output_dir: Path) -> None:
    benchmarks = []
    for row in rows:
        if row.benchmark not in benchmarks:
            benchmarks.append(row.benchmark)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharex=True)
    for benchmark in benchmarks:
        group = sorted([row for row in rows if row.benchmark == benchmark], key=lambda row: (row.values["threshold"], row.values["mincc"]))
        if not group:
            continue
        x = [row.values["threshold"] for row in group]
        label = _short_benchmark_name(benchmark)
        axes[0].plot(x, [row.values["missed positive"] for row in group], marker="o", linewidth=1.4, label=label)
        axes[1].plot(x, [row.values["false positive"] for row in group], marker="o", linewidth=1.4, label=label)
        for ax, key in zip(axes, ["missed positive", "false positive"]):
            for row in group:
                ax.text(row.values["threshold"], row.values[key] + 0.08, f"c{_fmt(row.values['mincc'], 0)}", fontsize=7, ha="center")
    axes[0].set_title("Missed positives")
    axes[1].set_title("Empty false positives")
    for ax in axes:
        ax.set_xlabel("threshold")
        ax.set_ylabel("case count")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Error trade-off across threshold/mincc choices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_9_error_tradeoff_lines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_radar(rows: list[FinalLogRow], output_dir: Path) -> None:
    selected = [row for row in _best_rows_by_benchmark(rows) if _model_type(row) == "Ensemble"]
    if not selected:
        return
    axes_labels = ["small", "micro", "large", "positive", "empty"]
    keys = ["dice small", "dice micro", "dice large", "dice positive", "dice empty"]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
    for row in selected:
        values = [row.values[key] for key in keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.4, label=row.short_name)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Hybrid/ensemble robustness across lesion-size groups", fontsize=14, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=7)
    fig.tight_layout()
    for path in [
        output_dir / "plot_10_size_group_radar.png",
        output_dir / "plot_10_hybrid_ensemble_size_group_radar.png",
    ]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_size_distribution(rows: list[FinalLogRow], output_dir: Path) -> None:
    labels = ["micro", "small", "large"]
    keys = ["dice micro", "dice small", "dice large"]
    data = [[row.values[key] for row in rows if math.isfinite(row.values[key])] for key in keys]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_alpha(0.28)
    for index, values in enumerate(data, start=1):
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(np.array(jitter) + index, values, s=22, color="#e45756", alpha=0.75, edgecolor="black", linewidth=0.3)
    ax.set_title("Dice distribution by lesion-size group across configurations", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_11_size_group_dice_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_summary(rows: list[FinalLogRow], output_dir: Path) -> None:
    best_by_name: dict[str, FinalLogRow] = {}
    for row in rows:
        current = best_by_name.get(row.short_name)
        if current is None or _rank_sort_key(row) < _rank_sort_key(current):
            best_by_name[row.short_name] = row

    tta_pairs = [
        ("ddp_kpcyjb66", best_by_name.get("ddp_kpcyjb66 no-TTA"), best_by_name.get("ddp_kpcyjb66 TTA")),
        ("e5ohnz5w", best_by_name.get("e5ohnz5w no-TTA"), best_by_name.get("e5ohnz5w TTA")),
    ]
    tta_pairs = [(name, base, tta) for name, base, tta in tta_pairs if base is not None and tta is not None]
    best_single = min(
        [row for row in best_by_name.values() if _model_type(row) == "Single"],
        key=_rank_sort_key,
        default=None,
    )
    best_ensemble = min(
        [row for row in best_by_name.values() if _model_type(row) == "Ensemble"],
        key=_rank_sort_key,
        default=None,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    if tta_pairs:
        labels = [name for name, _, _ in tta_pairs]
        y = np.arange(len(labels))
        dice_delta = [tta.values["dice all"] - base.values["dice all"] for _, base, tta in tta_pairs]
        hd95_improvement = [base.values["hd95 all"] - tta.values["hd95 all"] for _, base, tta in tta_pairs]
        axes[0].barh(y - 0.18, dice_delta, height=0.36, label="Delta Dice All", color="#4c78a8")
        axes[0].barh(y + 0.18, hd95_improvement, height=0.36, label="HD95 All improvement", color="#f58518")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels)
        axes[0].set_title("TTA impact")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="x", alpha=0.25)
    else:
        axes[0].axis("off")

    if best_single is not None and best_ensemble is not None:
        labels = ["Best single", "Best ensemble"]
        x = np.arange(2)
        axes[1].bar(x - 0.18, [best_single.values["dice positive"], best_ensemble.values["dice positive"]], width=0.36, label="Dice+", color="#4c78a8")
        axes[1].bar(x + 0.18, [best_single.values["assd positive"], best_ensemble.values["assd positive"]], width=0.36, label="ASSD+", color="#e45756")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_title("Best single vs best ensemble")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].text(0, 0.02, best_single.short_name, transform=axes[1].get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
        axes[1].text(1, 0.02, best_ensemble.short_name, transform=axes[1].get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    else:
        axes[1].axis("off")
    fig.suptitle("Ablation summary: TTA and ensemble value", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_12_tta_ensemble_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready figures from FINAL LOGS.csv.")
    parser.add_argument("--csv", type=Path, default=Path("FINAL LOGS.csv"), help="Path to FINAL LOGS.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures") / "final_logs", help="Directory for figures and cleaned CSV.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top rows to show in compact heatmaps.")
    parser.add_argument("--label-width", type=int, default=34, help="Wrap y-axis labels to this many characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_final_logs(args.csv)
    add_ranks(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_clean_csv(rows, args.output_dir / "cleaned_final_logs_with_ranks.csv")
    sorted_rows = sorted(rows, key=_rank_sort_key)
    write_clean_csv(sorted_rows, args.output_dir / "ranked_final_logs.csv")

    top_n = max(1, min(args.top_n, len(rows)))
    plot_headline_grouped_bars(rows, args.output_dir, args.label_width)
    plot_rank_summary(rows, args.output_dir, top_n, args.label_width)
    plot_tradeoff_scatter(rows, args.output_dir)
    plot_dice_hd95_all_scatter(rows, args.output_dir)
    plot_rank_contributions(rows, args.output_dir, top_n, args.label_width)
    plot_group_panels(rows, args.output_dir, top_n, args.label_width)
    plot_best_per_benchmark(rows, args.output_dir, args.label_width)
    plot_model_variant_summary(rows, args.output_dir, args.label_width)
    plot_threshold_mincc_heatmaps(rows, args.output_dir)
    plot_error_tradeoff_lines(rows, args.output_dir)
    plot_robustness_radar(rows, args.output_dir)
    plot_size_distribution(rows, args.output_dir)
    plot_ablation_summary(rows, args.output_dir)

    print(f"Read {len(rows)} metric rows from {args.csv}")
    print(f"Wrote figures and CSV summaries to {args.output_dir}")
    print("Hybrid labels use order: ddp_kpcyjb66 / e5ohnz5w")
    print("Best positive-rank setting:")
    best = sorted_rows[0]
    print(f"  {best.setting_label.replace(chr(10), ' | ')}")
    print(
        "  "
        f"dice+={best.values['dice positive']:.4f}, "
        f"hd95+={best.values['hd95 positive']:.2f}, "
        f"assd+={best.values['assd positive']:.2f}, "
        f"missed+={best.values['missed positive']:.0f}, "
        f"empty_fp={best.values['false positive']:.0f}"
    )


if __name__ == "__main__":
    main()

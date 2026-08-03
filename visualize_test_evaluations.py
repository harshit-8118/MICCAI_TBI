from __future__ import annotations

import argparse
import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    # "best_tr_f1_kpcyjb66_no_tta",
    # "best_tr_f1_kpcyjb66_tta",
    # "best_tr_714109_no_tta",
    # "best_tr_714109_tta",
    # "single_ddp_fft_finetuned_kpycyjb66_no_tta",
    # "single_ddp_fft_finetuned_kpycyjb66_tta",
    # "ep280_core_halo_growth_sweep",
    # "ensemble_4_models_no_tta_192",
    # "ensemble_4_models_tta_192",
    # "ensemble_4_models_no_tta",
    # "ensemble_4_models_tta",
    # "best_val_f1_ueamq6m8_no_tta",
    # "ep280_core05_halo005_fast",
    # "ensemble_2_models_e5ohnz5w_ddp_kpcyjb66_no_tta",
    "ensemble_2_models_e5ohnz5w_tta_ddp_kpcyjb66_no_tta_hybrid",
    "ensemble_2_stgy_2_e5ohnz5w_tta_ddp_kpcyjb66_no_tta",
    # "best_tr_all_ddp_128_ue6v2mtj_no_tta",
    # "best_val_f0_e5ohnz5w_no_tta",
    # "best_val_f0_e5ohnz5w_tta",
]

DICE_METRICS = [
    ("balanced", "test_dice_balanced"),
    ("all", "test_dice_all"),
    ("positive", "test_dice_positive"),
    ("gt50", "test_dice_gt50"),
    # ("micro", "test_dice_micro"),
    ("empty", "test_dice_empty"),
    # ("very tiny", "test_dice_very_tiny"),
    ("tiny", "test_dice_tiny"),
    ("small", "test_dice_small"),
    ("large", "test_dice_large"),
]

LESION_GROUPS = [
    ("all", "all"),
    ("positive", "positive"),
    ("gt50", "gt50"),
    # ("micro", "micro"),
    ("empty", "empty"),
    # ("very tiny", "very_tiny"),
    ("tiny", "tiny"),
    ("small", "small"),
    ("large", "large"),
]

SWEEP_METRICS = [
    ("balanced", "test_dice_balanced"),
    ("positive", "test_dice_positive"),
    ("gt50", "test_dice_gt50"),
    # ("micro", "test_dice_micro"),
    ("empty", "test_dice_empty"),
    # ("very tiny", "test_dice_very_tiny"),
    ("tiny", "test_dice_tiny"),
    ("small", "test_dice_small"),
    ("large", "test_dice_large"),
]
from __future__ import annotations

import argparse
import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    # "best_tr_f1_kpcyjb66_no_tta",
    # "best_tr_f1_kpcyjb66_tta",
    # "best_tr_714109_no_tta",
    # "best_tr_714109_tta",
    # "single_ddp_fft_finetuned_kpycyjb66_no_tta",
    # "single_ddp_fft_finetuned_kpycyjb66_tta",
    # "ep280_core_halo_growth_sweep",
    "ensemble_2_models_e5ohnz5w_ddp_kpcyjb66_no_tta",
    # "ensemble_4_models_no_tta_192",
    # "ensemble_4_models_tta_192",
    # "ensemble_4_models_no_tta",
    # "ensemble_4_models_tta",
    # "best_val_f1_ueamq6m8_no_tta",
    # "ep280_core05_halo005_fast",
    "ensemble_2_models_e5ohnz5w_tta_ddp_kpcyjb66_no_tta_hybrid",
    "ensemble_2_stgy_2_e5ohnz5w_tta_ddp_kpcyjb66_no_tta",
    # "best_tr_all_ddp_128_ue6v2mtj_no_tta",
    # "best_val_f0_e5ohnz5w_no_tta",
    # "best_val_f0_e5ohnz5w_tta",
]

DICE_METRICS = [
    ("balanced", "test_dice_balanced"),
    ("all", "test_dice_all"),
    ("positive", "test_dice_positive"),
    ("gt50", "test_dice_gt50"),
    # ("micro", "test_dice_micro"),
    ("empty", "test_dice_empty"),
    # ("very tiny", "test_dice_very_tiny"),
    ("tiny", "test_dice_tiny"),
    ("small", "test_dice_small"),
    ("large", "test_dice_large"),
]

LESION_GROUPS = [
    ("all", "all"),
    ("positive", "positive"),
    ("gt50", "gt50"),
    # ("micro", "micro"),
    ("empty", "empty"),
    # ("very tiny", "very_tiny"),
    ("tiny", "tiny"),
    ("small", "small"),
    ("large", "large"),
]

SWEEP_METRICS = [
    ("balanced", "test_dice_balanced"),
    ("positive", "test_dice_positive"),
    ("gt50", "test_dice_gt50"),
    # ("micro", "test_dice_micro"),
    ("empty", "test_dice_empty"),
    # ("very tiny", "test_dice_very_tiny"),
    ("tiny", "test_dice_tiny"),
    ("small", "test_dice_small"),
    ("large", "test_dice_large"),
]

ERROR_METRICS = [
    ("missed +", "n_missed_positive"),
    ("empty FP", "n_empty_false_positive"),
]

HD95_METRICS = [(label, f"test_hd95_{suffix}") for label, suffix in LESION_GROUPS]
ASSD_METRICS = [(label, f"test_assd_{suffix}") for label, suffix in LESION_GROUPS]

SURFACE_METRICS = [
    *[(f"HD95 {label}", column) for label, column in HD95_METRICS],
    *[(f"ASSD {label}", column) for label, column in ASSD_METRICS],
    ("rank proxy", "ranking_proxy_sum"),
    ("comparison rank", "comparison_ranking_proxy_sum"),
]

SETTING_COLUMNS = [
    ("thr", "threshold", 3),
    ("mincc", "min_component_voxels", 0),
    ("core", "core_threshold", 3),
    ("halo", "halo_threshold", 3),
    ("grow", "max_growth_ratio", 2),
    ("dil", "dilation_radius", 0),
    ("dilcc", "dilate_min_component_voxels", 0),
    ("corecc", "core_min_component_voxels", 0),
    ("maxvox", "max_grown_component_voxels", 0),
]


@dataclass(frozen=True)
class SummaryRow:
    run: str
    path: Path
    values: dict[str, float | str]


def _to_number(value: str) -> float:
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _format_value(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _get(row: SummaryRow, key: str) -> float:
    value = row.values.get(key, float("nan"))
    if isinstance(value, str):
        return _to_number(value)
    return float(value)


def _has_metric(rows: list[SummaryRow], key: str) -> bool:
    return any(math.isfinite(_get(row, key)) for row in rows)


def _lower_is_better(metric: str, select_mode: str = "auto") -> bool:
    if select_mode == "min":
        return True
    if select_mode == "max":
        return False
    lowered = metric.lower()
    return any(token in lowered for token in ["loss", "hd95", "assd", "rank", "missed", "false_positive"])


def _rank_values(values: list[float], higher_is_better: bool) -> list[int]:
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


def add_combined_ranking_proxy(rows: list[SummaryRow]) -> None:
    if not (_has_metric(rows, "test_dice_all") and _has_metric(rows, "test_hd95_all") and _has_metric(rows, "test_assd_all")):
        return
    dice_ranks = _rank_values([_get(row, "test_dice_all") for row in rows], higher_is_better=True)
    hd95_ranks = _rank_values([_get(row, "test_hd95_all") for row in rows], higher_is_better=False)
    assd_ranks = _rank_values([_get(row, "test_assd_all") for row in rows], higher_is_better=False)
    for row, dice_rank, hd95_rank, assd_rank in zip(rows, dice_ranks, hd95_ranks, assd_ranks):
        row.values["comparison_dice_rank"] = float(dice_rank)
        row.values["comparison_hd95_rank"] = float(hd95_rank)
        row.values["comparison_assd_rank"] = float(assd_rank)
        row.values["comparison_ranking_proxy_sum"] = float(dice_rank + hd95_rank + assd_rank)


def add_selected_row_ranks(rows: list[SummaryRow]) -> None:
    """Ranks recalculated only within the rows currently shown in a plot/table."""
    if not rows:
        return
    dice_ranks = _rank_values([_get(row, "test_dice_all") for row in rows], higher_is_better=True)
    hd95_ranks = _rank_values([_get(row, "test_hd95_all") for row in rows], higher_is_better=False)
    assd_ranks = _rank_values([_get(row, "test_assd_all") for row in rows], higher_is_better=False)
    selected_sums: list[float] = []
    for row, dice_rank, hd95_rank, assd_rank in zip(rows, dice_ranks, hd95_ranks, assd_ranks):
        rank_sum = float(dice_rank + hd95_rank + assd_rank)
        row.values["selected_dice_rank"] = float(dice_rank)
        row.values["selected_hd95_rank"] = float(hd95_rank)
        row.values["selected_assd_rank"] = float(assd_rank)
        row.values["selected_rank_sum"] = rank_sum
        selected_sums.append(rank_sum)
    selected_cmp_ranks = _rank_values(selected_sums, higher_is_better=False)
    for row, rank in zip(rows, selected_cmp_ranks):
        row.values["selected_cmp_rank"] = float(rank)


def _sort_value(value: float) -> float:
    return value if math.isfinite(value) else math.inf


def _sort_key(row: SummaryRow) -> tuple[object, ...]:
    return (
        row.run,
        _sort_value(_get(row, "threshold")),
        _sort_value(_get(row, "min_component_voxels")),
        _sort_value(_get(row, "halo_threshold")),
        _sort_value(_get(row, "max_growth_ratio")),
        _sort_value(_get(row, "dilation_radius")),
    )


def _read_summary(path: Path, run_name: str) -> list[SummaryRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[SummaryRow] = []
        for raw in reader:
            values: dict[str, float | str] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                if key in {"run", "source_path"}:
                    values[key] = str(value)
                else:
                    number = _to_number(value)
                    values[key] = number if math.isfinite(number) else str(value or "")
            rows.append(SummaryRow(run=run_name, path=path, values=values))
    return rows


def _discover_run_names(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    default_existing = [name for name in DEFAULT_RUNS if (root / name / "summary.csv").exists()]
    if default_existing:
        return default_existing
    return sorted(path.name for path in root.iterdir() if (path / "summary.csv").exists())


def load_all_rows(root: Path, run_names: list[str] | None) -> list[SummaryRow]:
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")
    names = _discover_run_names(root, run_names)
    if not names:
        raise FileNotFoundError(f"No run folders with summary.csv found under {root}")

    rows: list[SummaryRow] = []
    missing: list[str] = []
    for name in names:
        summary_path = root / name / "summary.csv"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        rows.extend(_read_summary(summary_path, name))
    if missing:
        raise FileNotFoundError("Missing summary.csv files:\n" + "\n".join(missing))
    if not rows:
        raise ValueError(f"No rows loaded from {root}")
    return sorted(rows, key=_sort_key)


def choose_best_rows(rows: list[SummaryRow], select_metric: str, select_mode: str = "auto") -> list[SummaryRow]:
    by_run: dict[str, list[SummaryRow]] = {}
    for row in rows:
        by_run.setdefault(row.run, []).append(row)

    lower_is_better = _lower_is_better(select_metric, select_mode)

    def key_for(row: SummaryRow) -> tuple[float, float, float, float, float]:
        primary = _get(row, select_metric)
        if not math.isfinite(primary):
            primary_score = -math.inf
        else:
            primary_score = -primary if lower_is_better else primary
        return (
            primary_score,
            _get(row, "test_dice_positive"),
            _get(row, "test_dice_gt50"),
            -_get(row, "n_missed_positive"),
            -_get(row, "n_empty_false_positive"),
        )

    best: list[SummaryRow] = []
    for run, run_rows in by_run.items():
        best_row = max(run_rows, key=key_for)
        best.append(best_row)
    return sorted(best, key=lambda row: row.run)


def _all_columns(rows: list[SummaryRow]) -> list[str]:
    preferred = [
        "run",
        "source_path",
        "postprocess",
        "tta",
        *[column for _, column, _ in SETTING_COLUMNS],
        "test_loss",
        *[column for _, column in DICE_METRICS],
        *[column for _, column in SURFACE_METRICS],
        *[column for _, column in ERROR_METRICS],
        "ranking_proxy_dice_rank",
        "ranking_proxy_hd95_rank",
        "ranking_proxy_assd_rank",
        "comparison_dice_rank",
        "comparison_hd95_rank",
        "comparison_assd_rank",
        "comparison_ranking_proxy_sum",
        "selected_dice_rank",
        "selected_hd95_rank",
        "selected_assd_rank",
        "selected_rank_sum",
        "selected_cmp_rank",
        "n_cases",
        "n_positive",
        "n_gt50",
        "n_empty",
        "n_very_tiny",
        "n_tiny",
        "n_small",
        "n_large",
    ]
    discovered = sorted({key for row in rows for key in row.values})
    columns = [column for column in preferred if column == "run" or column == "source_path" or column in discovered]
    columns.extend(column for column in discovered if column not in columns)
    return columns


def write_rows(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _all_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out: dict[str, object] = {"run": row.run, "source_path": str(row.path)}
            out.update(row.values)
            writer.writerow(out)


def _short_run_name(name: str) -> str:
    return name.replace("single_", "").replace("_no_tta", "\nno TTA").replace("_tta", "\nTTA").replace("_", " ")


def _available_metric_specs(rows: list[SummaryRow], specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(label, column) for label, column in specs if _has_metric(rows, column)]


def _metric_matrix(rows: list[SummaryRow], specs: list[tuple[str, str]]) -> np.ndarray:
    return np.array([[_get(row, column) for _, column in specs] for row in rows], dtype=float)


def _capped_vmax(matrix: np.ndarray, percentile: float = 95.0) -> float:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 1.0
    cap = float(np.nanpercentile(finite, percentile))
    if not math.isfinite(cap) or cap <= 0:
        cap = float(np.nanmax(finite))
    return max(cap, 1e-6)


def _format_setting_param(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return ""
    if digits == 0 or abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _setting_label(row: SummaryRow, include_run: bool = True) -> str:
    parts: list[str] = []
    for label, column, digits in SETTING_COLUMNS:
        value = _get(row, column)
        if math.isfinite(value):
            parts.append(f"{label}={_format_setting_param(value, digits)}")
    if isinstance(row.values.get("postprocess"), str) and row.values.get("postprocess"):
        parts.append(str(row.values["postprocess"]))
    first_line = _short_run_name(row.run).replace("\n", " ") if include_run else ""
    chunks = [" ".join(parts[:4]), " ".join(parts[4:])]
    lines = [first_line] if first_line else []
    lines.extend(chunk for chunk in chunks if chunk)
    return "\n".join(lines)


def _top_rows_by_rank(rows: list[SummaryRow], top_n: int, max_per_run: int | None = 5) -> list[SummaryRow]:
    rank_metric = None
    for candidate in ["comparison_ranking_proxy_sum", "ranking_proxy_sum", "test_loss"]:
        if _has_metric(rows, candidate):
            rank_metric = candidate
            break

    def finite_or(value: float, fallback: float) -> float:
        return value if math.isfinite(value) else fallback

    def key_for(row: SummaryRow) -> tuple[float, float, float, float, float]:
        if rank_metric:
            primary = _get(row, rank_metric)
            primary_score = finite_or(primary, math.inf)
        else:
            primary_score = -finite_or(_get(row, "test_dice_balanced"), -math.inf)
        return (
            primary_score,
            -finite_or(_get(row, "test_dice_all"), -math.inf),
            finite_or(_get(row, "test_hd95_all"), math.inf),
            finite_or(_get(row, "test_assd_all"), math.inf),
            finite_or(_get(row, "n_empty_false_positive"), math.inf),
        )

    sorted_rows = sorted(rows, key=key_for)
    if max_per_run is None or max_per_run <= 0:
        return sorted_rows[: max(1, top_n)]

    selected: list[SummaryRow] = []
    selected_per_run: dict[str, int] = {}
    for row in sorted_rows:
        if selected_per_run.get(row.run, 0) >= max_per_run:
            continue
        selected.append(row)
        selected_per_run[row.run] = selected_per_run.get(row.run, 0) + 1
        if len(selected) >= max(1, top_n):
            break
    return selected


def _scaled_marker_sizes(values: list[float], minimum: float = 45.0, maximum: float = 210.0) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [minimum for _ in values]
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi):
        return [(minimum + maximum) / 2 for _ in values]
    sizes = []
    for value in values:
        if not math.isfinite(value):
            sizes.append(minimum)
        else:
            sizes.append(minimum + (maximum - minimum) * (value - lo) / (hi - lo))
    return sizes


def _heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    fmt: str = ".3f",
    annot_fontsize: float = 7.0,
    xtick_fontsize: float = 8.0,
    ytick_fontsize: float = 8.0,
    show_ylabels: bool = True,
) -> None:
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=xtick_fontsize)
    ax.set_yticks(range(len(row_labels)))
    ytick_texts = ax.set_yticklabels(row_labels if show_ylabels else [""] * len(row_labels), fontsize=ytick_fontsize)
    for tick_text in ytick_texts:
        tick_text.set_multialignment("right")
        tick_text.set_linespacing(1.08)
    if not show_ylabels:
        ax.tick_params(axis="y", length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                text = format(float(value), fmt)
                color = "white" if (vmax is not None and value > (vmin or 0) + 0.65 * (vmax - (vmin or 0))) else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=annot_fontsize, color=color)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def plot_best_dashboard(best_rows: list[SummaryRow], output_path: Path, select_metric: str, dpi: int) -> None:
    run_labels = [_short_run_name(row.run) for row in best_rows]
    metric_labels = [label for label, _ in DICE_METRICS]
    dice_matrix = np.array([[_get(row, column) for _, column in DICE_METRICS] for row in best_rows], dtype=float)
    error_matrix = np.array([[_get(row, column) for _, column in ERROR_METRICS] for row in best_rows], dtype=float)
    has_surface = any(_has_metric(best_rows, column) for _, column in SURFACE_METRICS)
    setting_text = []
    for row in best_rows:
        parts = [
            f"thr={_format_value(_get(row, 'threshold'), 2)}",
            f"mincc={int(_get(row, 'min_component_voxels'))}",
            f"loss={_format_value(_get(row, 'test_loss'), 3)}",
        ]
        if has_surface:
            parts.append(f"hd95={_format_value(_get(row, 'test_hd95_all'), 2)}")
            parts.append(f"assd={_format_value(_get(row, 'test_assd_all'), 2)}")
            parts.append(f"rank={_format_value(_get(row, 'ranking_proxy_sum'), 0)}")
        setting_text.append("\n".join(parts))

    fig = plt.figure(figsize=(18, max(6, 0.7 * len(best_rows) + 3)))
    grid = fig.add_gridspec(1, 3, width_ratios=[5.2, 1.25, 1.25])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])

    _heatmap(
        ax0,
        dice_matrix,
        run_labels,
        metric_labels,
        f"Best Setting Per Run by {select_metric}",
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
    )
    max_error = max(1.0, float(np.nanmax(error_matrix)) if np.isfinite(error_matrix).any() else 1.0)
    _heatmap(
        ax1,
        error_matrix,
        run_labels,
        [label for label, _ in ERROR_METRICS],
        "Error Counts",
        vmin=0.0,
        vmax=max_error,
        cmap="YlOrRd",
        fmt=".0f",
    )
    ax2.axis("off")
    ax2.set_title("Chosen Setting", fontsize=11, fontweight="bold")
    for index, text in enumerate(setting_text):
        y = 1.0 - (index + 0.5) / max(len(setting_text), 1)
        ax2.text(0.05, y, text, va="center", ha="left", fontsize=9)

    fig.suptitle("Test Evaluation Overview", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _unique_sorted(rows: Iterable[SummaryRow], key: str) -> list[float]:
    values = sorted({round(_get(row, key), 10) for row in rows if math.isfinite(_get(row, key))})
    return values


def _series_by_threshold(rows: list[SummaryRow], metric: str, mincc: float) -> tuple[list[float], list[float]]:
    thresholds = _unique_sorted(rows, "threshold")
    lower_is_better = _lower_is_better(metric)
    x_values: list[float] = []
    y_values: list[float] = []
    for threshold in thresholds:
        values = [
            _get(row, metric)
            for row in rows
            if _get(row, "min_component_voxels") == mincc and abs(_get(row, "threshold") - threshold) < 1e-9
        ]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            continue
        x_values.append(threshold)
        y_values.append(min(values) if lower_is_better else max(values))
    return x_values, y_values


def _color_map(run_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(index % 10) for index, name in enumerate(run_names)}


def plot_threshold_sweeps(rows: list[SummaryRow], output_path: Path, dpi: int) -> None:
    run_names = sorted({row.run for row in rows})
    min_components = _unique_sorted(rows, "min_component_voxels")
    colors = _color_map(run_names)
    styles = ["-", "--", ":", "-."]
    style_for_mincc = {value: styles[index % len(styles)] for index, value in enumerate(min_components)}

    fig, axes = plt.subplots(3, 3, figsize=(20, 13), sharex=False, sharey=False)
    axes_flat = list(axes.ravel())
    for ax, (label, column) in zip(axes_flat, SWEEP_METRICS):
        for run in run_names:
            run_rows = [row for row in rows if row.run == run]
            for mincc in min_components:
                thresholds, values = _series_by_threshold(run_rows, column, mincc)
                if not thresholds:
                    continue
                ax.plot(
                    thresholds,
                    values,
                    color=colors[run],
                    linestyle=style_for_mincc[mincc],
                    linewidth=1.7,
                    alpha=0.85,
                )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("threshold")
        ax.set_ylabel("Dice")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)

    run_handles = [
        plt.Line2D([0], [0], color=colors[run], linewidth=2.2, label=_short_run_name(run).replace("\n", " "))
        for run in run_names
    ]
    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle=style_for_mincc[mincc], linewidth=2.0, label=f"mincc={int(mincc)}")
        for mincc in min_components
    ]
    fig.legend(handles=run_handles + style_handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Dice Metrics Across Thresholds and Min-Component Voxels", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _metric_grid(rows: list[SummaryRow], metric: str) -> tuple[np.ndarray, list[str], list[str]]:
    thresholds = _unique_sorted(rows, "threshold")
    min_components = _unique_sorted(rows, "min_component_voxels")
    matrix = np.full((len(min_components), len(thresholds)), np.nan, dtype=float)
    for i, mincc in enumerate(min_components):
        for j, threshold in enumerate(thresholds):
            selected = [
                row
                for row in rows
                if _get(row, "min_component_voxels") == mincc and abs(_get(row, "threshold") - threshold) < 1e-9
            ]
            if selected:
                values = [_get(row, metric) for row in selected if math.isfinite(_get(row, metric))]
                if values:
                    matrix[i, j] = min(values) if _lower_is_better(metric) else max(values)
    return matrix, [f"{value:g}" for value in min_components], [f"{value:g}" for value in thresholds]


def plot_run_heatmaps(rows: list[SummaryRow], output_path: Path, dpi: int) -> None:
    run_names = sorted({row.run for row in rows})
    heatmap_specs = [
        ("balanced", "test_dice_balanced", "YlGnBu", 0.0, 1.0, ".3f"),
        ("positive", "test_dice_positive", "YlGnBu", 0.0, 1.0, ".3f"),
        ("test loss", "test_loss", "magma_r", None, None, ".3f"),
        ("missed +", "n_missed_positive", "YlOrRd", None, None, ".0f"),
        ("empty FP", "n_empty_false_positive", "YlOrRd", None, None, ".0f"),
    ]
    fig, axes = plt.subplots(len(run_names), len(heatmap_specs), figsize=(21, max(4, 2.7 * len(run_names))))
    if len(run_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, run in enumerate(run_names):
        run_rows = [row for row in rows if row.run == run]
        for col_index, (title, metric, cmap, vmin, vmax, fmt) in enumerate(heatmap_specs):
            matrix, mincc_labels, threshold_labels = _metric_grid(run_rows, metric)
            if vmax is None and np.isfinite(matrix).any():
                vmax = max(1.0, float(np.nanmax(matrix)))
            axes[row_index, col_index].set_ylabel(f"{_short_run_name(run)}\nmincc", fontsize=8)
            _heatmap(
                axes[row_index, col_index],
                matrix,
                mincc_labels,
                threshold_labels,
                title if row_index == 0 else "",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                fmt=fmt,
            )
            axes[row_index, col_index].set_xlabel("threshold", fontsize=8)

    fig.suptitle("Per-Run Threshold x Min-Component Tradeoffs", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_tradeoff_scatter(rows: list[SummaryRow], output_path: Path, dpi: int) -> None:
    run_names = sorted({row.run for row in rows})
    colors = _color_map(run_names)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title in [
        (axes[0], "test_dice_balanced", "Balanced Dice vs Missed Positives"),
        (axes[1], "test_dice_positive", "Positive Dice vs Missed Positives"),
    ]:
        for run in run_names:
            run_rows = [row for row in rows if row.run == run]
            x = [_get(row, "n_missed_positive") for row in run_rows]
            y = [_get(row, metric) for row in run_rows]
            sizes = [40 + 22 * _get(row, "n_empty_false_positive") for row in run_rows]
            ax.scatter(x, y, s=sizes, color=colors[run], alpha=0.55, label=_short_run_name(run).replace("\n", " "))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("n_missed_positive")
        ax.set_ylabel(metric)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Error Tradeoff: Marker Size = Empty False Positives", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.86, 0.94])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_surface_sweeps(rows: list[SummaryRow], output_path: Path, dpi: int) -> bool:
    specs = [(label, column) for label, column in SURFACE_METRICS if _has_metric(rows, column)]
    if not specs:
        return False

    run_names = sorted({row.run for row in rows})
    min_components = _unique_sorted(rows, "min_component_voxels")
    colors = _color_map(run_names)
    styles = ["-", "--", ":", "-."]
    style_for_mincc = {value: styles[index % len(styles)] for index, value in enumerate(min_components)}

    ncols = min(3, len(specs))
    nrows = int(math.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.2 * nrows), squeeze=False)
    axes_flat = list(axes.ravel())
    for ax, (label, column) in zip(axes_flat, specs):
        for run in run_names:
            run_rows = [row for row in rows if row.run == run]
            for mincc in min_components:
                thresholds, values = _series_by_threshold(run_rows, column, mincc)
                if not thresholds:
                    continue
                ax.plot(
                    thresholds,
                    values,
                    color=colors[run],
                    linestyle=style_for_mincc[mincc],
                    linewidth=1.7,
                    alpha=0.85,
                )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("threshold")
        ax.set_ylabel("lower is better" if _lower_is_better(column) else "higher is better")
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(specs) :]:
        ax.axis("off")

    run_handles = [
        plt.Line2D([0], [0], color=colors[run], linewidth=2.2, label=_short_run_name(run).replace("\n", " "))
        for run in run_names
    ]
    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle=style_for_mincc[mincc], linewidth=2.0, label=f"mincc={int(mincc)}")
        for mincc in min_components
    ]
    fig.legend(handles=run_handles + style_handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Surface Metrics and Challenge Proxy Across Sweeps", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_best_surface_dashboard(best_rows: list[SummaryRow], output_path: Path, dpi: int) -> bool:
    hd95_specs = _available_metric_specs(best_rows, HD95_METRICS)
    assd_specs = _available_metric_specs(best_rows, ASSD_METRICS)
    if not hd95_specs and not assd_specs:
        return False

    run_labels = [_short_run_name(row.run) for row in best_rows]
    panels: list[tuple[str, list[tuple[str, str]], str]] = []
    if hd95_specs:
        panels.append(("HD95 by lesion group (lower is better)", hd95_specs, "YlOrRd"))
    if assd_specs:
        panels.append(("ASSD by lesion group (lower is better)", assd_specs, "YlOrRd"))

    fig, axes = plt.subplots(1, len(panels), figsize=(8.4 * len(panels), max(6, 0.65 * len(best_rows) + 3)))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, specs, cmap) in zip(axes, panels):
        matrix = _metric_matrix(best_rows, specs)
        _heatmap(
            ax,
            matrix,
            run_labels,
            [label for label, _ in specs],
            title,
            vmin=0.0,
            vmax=_capped_vmax(matrix),
            cmap=cmap,
            fmt=".2f",
        )

    fig.suptitle("Best Settings: Surface Metrics by Lesion Group", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def _truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _wrap_text(text: str, width: int) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    return textwrap.wrap(
        text,
        width=max(12, int(width)),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [text]


def _top_setting_label(row: SummaryRow, index: int) -> str:
    run_name = _short_run_name(row.run).replace("\n", " ")
    parts: list[str] = []
    for label, column, digits in SETTING_COLUMNS:
        value = _get(row, column)
        if not math.isfinite(value):
            continue
        if label in {"thr", "mincc", "core", "halo", "grow"} or abs(value) > 1e-9:
            parts.append(f"{label}={_format_setting_param(value, digits)}")
    postprocess = row.values.get("postprocess")
    if isinstance(postprocess, str) and postprocess and postprocess.lower() != "none":
        parts.append(postprocess)
    label_lines = _wrap_text(f"#{index:02d} {run_name}", 46)
    label_lines.extend(_wrap_text(" ".join(parts), 56))
    return "\n".join(label_lines)


def _top_row_labels(rows: list[SummaryRow]) -> list[str]:
    return [_top_setting_label(row, index) for index, row in enumerate(rows, start=1)]


def _label_line_count(labels: list[str]) -> int:
    return max(1, max((label.count("\n") + 1 for label in labels), default=1))


def _finite_panel_vmax(matrix: np.ndarray, *, cap: bool = True) -> float:
    if not np.isfinite(matrix).any():
        return 1.0
    if cap:
        return _capped_vmax(matrix)
    return max(1.0, float(np.nanmax(matrix)))


def _plot_top_metric_panels(
    rows: list[SummaryRow],
    output_path: Path,
    title: str,
    panels: list[tuple[str, list[tuple[str, str]], str, float | None, float | None, str, float, bool]],
    dpi: int,
) -> bool:
    available_panels = []
    for panel_title, specs, cmap, vmin, vmax, fmt, width_per_col, cap_vmax in panels:
        available_specs = _available_metric_specs(rows, specs)
        if not available_specs:
            continue
        matrix = _metric_matrix(rows, available_specs)
        panel_vmax = _finite_panel_vmax(matrix, cap=cap_vmax) if vmax is None else vmax
        available_panels.append((panel_title, available_specs, matrix, cmap, vmin, panel_vmax, fmt, width_per_col))

    if not available_panels:
        return False

    row_labels = _top_row_labels(rows)
    label_line_count = _label_line_count(row_labels)
    max_label_line_chars = max(
        (len(line) for label in row_labels for line in label.splitlines()),
        default=40,
    )
    widths = [max(2.4, width_per_col * len(specs)) for _, specs, *_rest, width_per_col in available_panels]
    left_label_inches = min(9.5, max(4.8, 0.085 * max_label_line_chars))
    fig_height = max(7.0, 0.42 * len(rows) * label_line_count + 3.2)
    fig_width = sum(widths) + left_label_inches + 2.2
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(1, len(available_panels), width_ratios=widths, wspace=0.38)

    for index, (panel_title, specs, matrix, cmap, vmin, vmax, fmt, _width_per_col) in enumerate(available_panels):
        ax = fig.add_subplot(grid[0, index])
        _heatmap(
            ax,
            matrix,
            row_labels,
            [label for label, _ in specs],
            panel_title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            fmt=fmt,
            annot_fontsize=6.6 if len(rows) > 18 else 7.2,
            xtick_fontsize=8.5,
            ytick_fontsize=6.8 if label_line_count > 2 else 7.4,
            show_ylabels=index == 0,
        )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.subplots_adjust(
        left=min(0.45, left_label_inches / fig_width),
        right=0.98,
        bottom=0.08,
        top=0.92,
        wspace=0.42,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_top_setting_detail_splits(top_rows: list[SummaryRow], output_dir: Path, dpi: int) -> list[Path]:
    if not top_rows:
        return []

    written: list[Path] = []
    detail_jobs = [
        (
            output_dir / "plot_7b_top_dice_groups.png",
            "Top Settings: Dice by Lesion Group",
            [("Dice groups", [(label, f"test_dice_{suffix}") for label, suffix in LESION_GROUPS], "YlGnBu", 0.0, 1.0, ".3f", 0.9, True)],
        ),
        (
            output_dir / "plot_7c_top_hd95_groups.png",
            "Top Settings: HD95 by Lesion Group (Lower is Better)",
            [("HD95 groups", HD95_METRICS, "YlOrRd", 0.0, None, ".2f", 0.9, True)],
        ),
        (
            output_dir / "plot_7d_top_assd_groups.png",
            "Top Settings: ASSD by Lesion Group (Lower is Better)",
            [("ASSD groups", ASSD_METRICS, "YlOrRd", 0.0, None, ".2f", 0.9, True)],
        ),
        (
            output_dir / "plot_7e_top_errors_and_ranks.png",
            "Top Settings: Errors and Ranks",
            [
                ("Errors", ERROR_METRICS, "YlOrRd", 0.0, None, ".0f", 0.9, False),
                (
                    "Own sweep ranks",
                    [
                        ("Dice", "ranking_proxy_dice_rank"),
                        ("HD95", "ranking_proxy_hd95_rank"),
                        ("ASSD", "ranking_proxy_assd_rank"),
                        ("sum", "ranking_proxy_sum"),
                    ],
                    "YlOrRd",
                    0.0,
                    None,
                    ".0f",
                    0.85,
                    False,
                ),
                (
                    "Global ranks",
                    [
                        ("Dice", "comparison_dice_rank"),
                        ("HD95", "comparison_hd95_rank"),
                        ("ASSD", "comparison_assd_rank"),
                        ("sum", "comparison_ranking_proxy_sum"),
                    ],
                    "YlOrRd",
                    0.0,
                    None,
                    ".0f",
                    0.85,
                    False,
                ),
                (
                    "Ranks within shown rows",
                    [
                        ("Dice", "selected_dice_rank"),
                        ("HD95", "selected_hd95_rank"),
                        ("ASSD", "selected_assd_rank"),
                        ("sum", "selected_rank_sum"),
                        ("cmp", "selected_cmp_rank"),
                    ],
                    "YlOrRd",
                    0.0,
                    None,
                    ".0f",
                    0.85,
                    False,
                ),
            ],
        ),
    ]

    for output_path, title, panels in detail_jobs:
        if _plot_top_metric_panels(top_rows, output_path, title, panels, dpi):
            written.append(output_path)
    return written


def plot_top_surface_settings(
    rows: list[SummaryRow],
    output_path: Path,
    top_n: int,
    top_per_run: int,
    dpi: int,
) -> list[SummaryRow]:
    top_rows = _top_rows_by_rank(rows, top_n, top_per_run)
    add_selected_row_ranks(top_rows)

    key_dice_specs = [
        ("all", "test_dice_all"),
        ("positive", "test_dice_positive"),
        ("gt50", "test_dice_gt50"),
        ("small", "test_dice_small"),
        ("large", "test_dice_large"),
    ]
    key_hd95_specs = [
        ("all", "test_hd95_all"),
        ("positive", "test_hd95_positive"),
        ("gt50", "test_hd95_gt50"),
        ("small", "test_hd95_small"),
        ("large", "test_hd95_large"),
    ]
    key_assd_specs = [
        ("all", "test_assd_all"),
        ("positive", "test_assd_positive"),
        ("gt50", "test_assd_gt50"),
        ("small", "test_assd_small"),
        ("large", "test_assd_large"),
    ]
    local_rank_specs = [
        ("Dice", "ranking_proxy_dice_rank"),
        ("HD95", "ranking_proxy_hd95_rank"),
        ("ASSD", "ranking_proxy_assd_rank"),
        ("sum", "ranking_proxy_sum"),
    ]
    selected_rank_specs = [
        ("Dice", "selected_dice_rank"),
        ("HD95", "selected_hd95_rank"),
        ("ASSD", "selected_assd_rank"),
        ("sum", "selected_rank_sum"),
        ("cmp", "selected_cmp_rank"),
    ]

    _plot_top_metric_panels(
        top_rows,
        output_path,
        f"Top {len(top_rows)} Settings: Compact Decision Overview ({'uncapped' if top_per_run <= 0 else f'max {top_per_run} per run'})",
        [
            ("Dice key", key_dice_specs, "YlGnBu", 0.0, 1.0, ".3f", 0.9, True),
            ("HD95 key", key_hd95_specs, "YlOrRd", 0.0, None, ".2f", 0.9, True),
            ("ASSD key", key_assd_specs, "YlOrRd", 0.0, None, ".2f", 0.9, True),
            ("Errors", ERROR_METRICS, "YlOrRd", 0.0, None, ".0f", 0.9, False),
            ("Own sweep ranks", local_rank_specs, "YlOrRd", 0.0, None, ".0f", 0.85, False),
            ("Ranks within shown rows", selected_rank_specs, "YlOrRd", 0.0, None, ".0f", 0.85, False),
        ],
        dpi,
    )
    return top_rows


def plot_surface_tradeoff_scatter(rows: list[SummaryRow], output_path: Path, dpi: int) -> bool:
    scatter_specs = [
        ("test_dice_all", "test_hd95_all", "test_assd_all", "All cases"),
        ("test_dice_positive", "test_hd95_positive", "test_assd_positive", "Positive cases"),
        ("test_dice_tiny", "test_hd95_tiny", "test_assd_tiny", "Tiny lesions"),
        ("test_dice_large", "test_hd95_large", "test_assd_large", "Large lesions"),
    ]
    scatter_specs = [
        spec
        for spec in scatter_specs
        if _has_metric(rows, spec[0]) and _has_metric(rows, spec[1]) and _has_metric(rows, spec[2])
    ]
    if not scatter_specs:
        return False

    run_names = sorted({row.run for row in rows})
    colors = _color_map(run_names)
    ncols = 2
    nrows = int(math.ceil(len(scatter_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6.5 * nrows), squeeze=False)
    axes_flat = list(axes.ravel())

    for ax, (x_metric, y_metric, size_metric, title) in zip(axes_flat, scatter_specs):
        all_sizes = _scaled_marker_sizes([_get(row, size_metric) for row in rows])
        size_by_id = {id(row): size for row, size in zip(rows, all_sizes)}
        for run in run_names:
            run_rows = [row for row in rows if row.run == run]
            x = [_get(row, x_metric) for row in run_rows]
            y = [_get(row, y_metric) for row in run_rows]
            sizes = [size_by_id[id(row)] for row in run_rows]
            ax.scatter(
                x,
                y,
                s=sizes,
                color=colors[run],
                alpha=0.55,
                edgecolors="black",
                linewidths=0.25,
                label=_short_run_name(run).replace("\n", " "),
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{x_metric} (higher is better)")
        ax.set_ylabel(f"{y_metric} (lower is better)")
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(scatter_specs) :]:
        ax.axis("off")
    axes_flat[min(1, len(scatter_specs) - 1)].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=8)
    fig.suptitle("Surface Risk Tradeoffs: Marker Size = ASSD", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_surface_run_heatmaps(rows: list[SummaryRow], output_path: Path, dpi: int) -> bool:
    run_names = sorted({row.run for row in rows})
    key_surface_specs = [
        ("HD95 all", "test_hd95_all", "YlOrRd", ".2f"),
        ("HD95 +", "test_hd95_positive", "YlOrRd", ".2f"),
        ("HD95 tiny", "test_hd95_tiny", "YlOrRd", ".2f"),
        ("HD95 small", "test_hd95_small", "YlOrRd", ".2f"),
        ("HD95 large", "test_hd95_large", "YlOrRd", ".2f"),
        ("ASSD all", "test_assd_all", "YlOrRd", ".2f"),
        ("ASSD +", "test_assd_positive", "YlOrRd", ".2f"),
        ("ASSD tiny", "test_assd_tiny", "YlOrRd", ".2f"),
        ("ASSD small", "test_assd_small", "YlOrRd", ".2f"),
        ("ASSD large", "test_assd_large", "YlOrRd", ".2f"),
        ("rank", "ranking_proxy_sum", "YlOrRd", ".0f"),
        ("cmp rank", "comparison_ranking_proxy_sum", "YlOrRd", ".0f"),
    ]
    specs = [spec for spec in key_surface_specs if _has_metric(rows, spec[1])]
    if not specs:
        return False

    fig, axes = plt.subplots(len(run_names), len(specs), figsize=(3.8 * len(specs), max(4, 2.7 * len(run_names))))
    if len(run_names) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(specs) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_index, run in enumerate(run_names):
        run_rows = [row for row in rows if row.run == run]
        for col_index, (title, metric, cmap, fmt) in enumerate(specs):
            matrix, mincc_labels, threshold_labels = _metric_grid(run_rows, metric)
            axes[row_index, col_index].set_ylabel(f"{_short_run_name(run)}\nmincc", fontsize=8)
            _heatmap(
                axes[row_index, col_index],
                matrix,
                mincc_labels,
                threshold_labels,
                title if row_index == 0 else "",
                vmin=0.0,
                vmax=_capped_vmax(matrix),
                cmap=cmap,
                fmt=fmt,
            )
            axes[row_index, col_index].set_xlabel("threshold", fontsize=8)

    fig.suptitle("Per-Run Surface-Metric Threshold x Min-Component Tradeoffs", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AIMS-TBI test_evaluations summary.csv sweeps.")
    parser.add_argument(
        "--root",
        default="test_evaluations",
        help="Directory containing run subfolders with summary.csv.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run folder names to include. Defaults to the six single_ep_* folders if present, otherwise all subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write plots. Defaults to <root>/comparison_plots.",
    )
    parser.add_argument(
        "--select-metric",
        default="comparison_ranking_proxy_sum",
        help="Metric used to choose the best row per run for the overview dashboard.",
    )
    parser.add_argument(
        "--select-mode",
        choices=["auto", "max", "min"],
        default="auto",
        help="Whether to maximize or minimize --select-metric. Auto minimizes loss/HD95/ASSD/rank metrics.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of globally top settings to show in the surface-metric leaderboard plot.",
    )
    parser.add_argument(
        "--top-per-run",
        type=int,
        default=5,
        help="Maximum rows each run/model can contribute to the top settings leaderboard. Use 0 for no cap.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_rows(root, args.runs)
    add_combined_ranking_proxy(rows)
    best_rows = choose_best_rows(rows, args.select_metric, args.select_mode)

    write_rows(output_dir / "combined_summary.csv", rows)
    write_rows(output_dir / "best_by_run.csv", best_rows)
    plot_best_dashboard(best_rows, output_dir / "plot_1_best_setting_dashboard.png", args.select_metric, args.dpi)
    plot_threshold_sweeps(rows, output_dir / "plot_2_threshold_mincc_sweeps.png", args.dpi)
    plot_run_heatmaps(rows, output_dir / "plot_3_threshold_mincc_heatmaps.png", args.dpi)
    plot_tradeoff_scatter(rows, output_dir / "plot_4_error_tradeoff_scatter.png", args.dpi)
    wrote_surface_plot = plot_surface_sweeps(rows, output_dir / "plot_5_surface_metric_sweeps.png", args.dpi)
    wrote_surface_dashboard = plot_best_surface_dashboard(
        best_rows,
        output_dir / "plot_6_best_surface_by_group.png",
        args.dpi,
    )
    top_rows = plot_top_surface_settings(
        rows,
        output_dir / "plot_7_top_ranked_settings_surface_groups.png",
        args.top_n,
        args.top_per_run,
        args.dpi,
    )
    top_detail_plots = plot_top_setting_detail_splits(top_rows, output_dir, args.dpi)
    write_rows(output_dir / "top_surface_settings.csv", top_rows)
    wrote_surface_scatter = plot_surface_tradeoff_scatter(rows, output_dir / "plot_8_surface_tradeoff_scatter.png", args.dpi)
    wrote_surface_heatmaps = plot_surface_run_heatmaps(
        rows,
        output_dir / "plot_9_threshold_mincc_surface_heatmaps.png",
        args.dpi,
    )

    print(f"[DONE] Loaded {len(rows)} rows from {root}")
    print(f"[DONE] Wrote combined CSV: {output_dir / 'combined_summary.csv'}")
    print(f"[DONE] Wrote best CSV    : {output_dir / 'best_by_run.csv'}")
    print(f"[DONE] Wrote top CSV     : {output_dir / 'top_surface_settings.csv'}")
    print(f"[DONE] Top row cap       : top_n={args.top_n}, top_per_run={args.top_per_run or 'none'}")
    if wrote_surface_plot:
        print(f"[DONE] Wrote surface plot: {output_dir / 'plot_5_surface_metric_sweeps.png'}")
    if wrote_surface_dashboard:
        print(f"[DONE] Wrote surface group dashboard: {output_dir / 'plot_6_best_surface_by_group.png'}")
    print(f"[DONE] Wrote top settings overview  : {output_dir / 'plot_7_top_ranked_settings_surface_groups.png'}")
    for plot_path in top_detail_plots:
        print(f"[DONE] Wrote top settings detail    : {plot_path}")
    if wrote_surface_scatter:
        print(f"[DONE] Wrote surface scatter        : {output_dir / 'plot_8_surface_tradeoff_scatter.png'}")
    if wrote_surface_heatmaps:
        print(f"[DONE] Wrote surface heatmaps       : {output_dir / 'plot_9_threshold_mincc_surface_heatmaps.png'}")
    print(f"[DONE] Wrote plots to   : {output_dir}")


if __name__ == "__main__":
    main()


'''
python visualize_test_evaluations.py \
  --root /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/checkpoints/Validation2025_100 \
  --output-dir comparison_plots \
  --select-metric comparison_ranking_proxy_sum \
  --top-n 20 \
  --top-per-run 5 \
  --dpi 180
'''
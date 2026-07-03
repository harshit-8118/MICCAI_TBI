from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    "single_ep_205_no_tta",
    "single_ep_205_tta",
    "single_ep_224_no_tta",
    "single_ep_224_tta",
    "single_ep_225_no_tta",
    "single_ep_225_tta",
]

DICE_METRICS = [
    ("balanced", "test_dice_balanced"),
    ("all", "test_dice_all"),
    ("positive", "test_dice_positive"),
    ("gt50", "test_dice_gt50"),
    ("micro", "test_dice_micro"),
    ("empty", "test_dice_empty"),
    ("very tiny", "test_dice_very_tiny"),
    ("tiny", "test_dice_tiny"),
    ("small", "test_dice_small"),
    ("large", "test_dice_large"),
]

SWEEP_METRICS = [
    ("balanced", "test_dice_balanced"),
    ("positive", "test_dice_positive"),
    ("gt50", "test_dice_gt50"),
    ("micro", "test_dice_micro"),
    ("empty", "test_dice_empty"),
    ("very tiny", "test_dice_very_tiny"),
    ("tiny", "test_dice_tiny"),
    ("small", "test_dice_small"),
    ("large", "test_dice_large"),
]

ERROR_METRICS = [
    ("missed +", "n_missed_positive"),
    ("empty FP", "n_empty_false_positive"),
]

SURFACE_METRICS = [
    ("HD95 all", "test_hd95_all"),
    ("HD95 positive", "test_hd95_positive"),
    ("ASSD all", "test_assd_all"),
    ("ASSD positive", "test_assd_positive"),
    ("rank proxy", "ranking_proxy_sum"),
    ("comparison rank", "comparison_ranking_proxy_sum"),
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


def _sort_key(row: SummaryRow) -> tuple[str, float, float]:
    return (row.run, _get(row, "threshold"), _get(row, "min_component_voxels"))


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
        "threshold",
        "min_component_voxels",
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
) -> None:
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                text = format(float(value), fmt)
                color = "white" if (vmax is not None and value > (vmin or 0) + 0.65 * (vmax - (vmin or 0))) else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
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
                selected = sorted(
                    [row for row in run_rows if _get(row, "min_component_voxels") == mincc],
                    key=lambda row: _get(row, "threshold"),
                )
                if not selected:
                    continue
                thresholds = [_get(row, "threshold") for row in selected]
                values = [_get(row, column) for row in selected]
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
                matrix[i, j] = _get(selected[0], metric)
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
                selected = sorted(
                    [row for row in run_rows if _get(row, "min_component_voxels") == mincc],
                    key=lambda row: _get(row, "threshold"),
                )
                if not selected:
                    continue
                thresholds = [_get(row, "threshold") for row in selected]
                values = [_get(row, column) for row in selected]
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
        default="test_dice_balanced",
        help="Metric used to choose the best row per run for the overview dashboard.",
    )
    parser.add_argument(
        "--select-mode",
        choices=["auto", "max", "min"],
        default="auto",
        help="Whether to maximize or minimize --select-metric. Auto minimizes loss/HD95/ASSD/rank metrics.",
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

    print(f"[DONE] Loaded {len(rows)} rows from {root}")
    print(f"[DONE] Wrote combined CSV: {output_dir / 'combined_summary.csv'}")
    print(f"[DONE] Wrote best CSV    : {output_dir / 'best_by_run.csv'}")
    if wrote_surface_plot:
        print(f"[DONE] Wrote surface plot: {output_dir / 'plot_5_surface_metric_sweeps.png'}")
    print(f"[DONE] Wrote plots to   : {output_dir}")


if __name__ == "__main__":
    main()



# python visualize_test_evaluations.py \
#   --root checkpoints/test_evaluations \
#   --runs \
#     single_ep_205_no_tta single_ep_205_tta \
#     single_ep_225_no_tta single_ep_225_tta \
#     single_ep_280_no_tta single_ep_280_tta \
#     ensemble_ep_205_225_tta ensemble_ep_205_225_no_tta 
"""Tables and plots of the idle-baseline runs and the contention runs (`run_experiment.py` CSVs)."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from slm_bench import csv_format, experiment_settings

# Each run: its label in tables and plots, the results folder, its run_id in the
# CSVs, and its plot color (categorical palette slots 1-3, fixed per run).
RUN_SOURCES = [
    ("balanced", "step2_balanced", "20261107_001457", "#2a78d6"),
    ("bestperf_run1", "step2_bestperf", "bestperf_run1", "#eb6834"),
    ("bestperf_run2", "step2_bestperf", "bestperf_run2", "#1baf7a"),
]
RESPONSE_TIME_LABEL = "Response time (ms)"
# Request-table column -> label used in the summary table.
SUMMARY_METRICS = [
    (csv_format.TTFT_MS, "TTFT (ms)"),
    (csv_format.RESPONSE_TIME_MS, RESPONSE_TIME_LABEL),
    (csv_format.TOK_PER_S_E2E, "Tokens/s (end-to-end)"),
]

PLOT_SCOPE = "q4_k_m, idle"  # what the figure titles say the data is

# Contention runs: (results folder, run_id). The heavy configs are a separate, shorter run.
CONTENTION_RUNS = [("contention_v2", "contention_v2"), ("contention_v2", "contention_heavy")]
QUANT_ORDER = list(experiment_settings.MODEL_FILES)
LOAD_ORDER = list(experiment_settings.LOAD_PROCESS_COUNTS)
BASELINE_LOAD = "idle"  # the deadline is defined from this load
DEADLINE_FACTOR = 1.5  # deadline D = DEADLINE_FACTOR x idle p50 of the same quantization
FIXED_DEADLINE_MS = 1500
# Two passes "disagree" for a config when their p50 or p99 differ by more than this fraction.
PASS_DISAGREEMENT_FRACTION = 0.25
PASS = "pass"  # table column: "1", "2", ... or ALL_PASSES (both passes pooled)
ALL_PASSES = "both"
CONFIG_KEYS = [csv_format.RUN_ID, csv_format.REPEAT, csv_format.ORDER_POSITION]  # one config run
CONFIG_PASS_KEYS = [csv_format.QUANT, csv_format.LOAD, PASS]  # grouping of the summary tables
# Plot encoding, fixed everywhere: color = quantization, line style = load.
QUANT_COLORS = {"q8_0": "#2a78d6", "q4_k_m": "#eb6834"}
LOAD_LINE_STYLES = {"idle": "-", "medium": "--", "heavy": ":"}

SURFACE_COLOR = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
# Applied once in main(), so the plot functions only have to draw.
PLOT_THEME = {
    "figure.facecolor": SURFACE_COLOR,
    "axes.facecolor": SURFACE_COLOR,
    "savefig.facecolor": SURFACE_COLOR,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.labelcolor": SECONDARY_INK,
    "axes.titlecolor": PRIMARY_INK,
    "axes.titlelocation": "left",
    "axes.titlesize": 11,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.labelcolor": SECONDARY_INK,
}


@dataclass
class Run:
    """One run's measured requests and its CPU readings."""

    label: str
    color: str
    requests: pd.DataFrame  # measured requests only (warm-ups dropped)
    cpu_readings: pd.DataFrame

    @property
    def line_style(self) -> dict:
        """Keyword arguments that draw this run's line in its own color and label."""
        return {"color": self.color, "label": self.label}


# --- Main flow ---------------------------------------------------------------------


def main() -> None:
    plt.rcParams.update(PLOT_THEME)
    experiment_settings.FIGURES_DIR.mkdir(exist_ok=True)
    analyze_idle_baseline()
    analyze_contention()


def analyze_idle_baseline() -> None:
    """Balanced vs Best performance idle runs: summary table and two figures."""
    runs = [load_run(*source) for source in RUN_SOURCES]
    request_summary = summarize_requests(runs)
    request_summary.round(3).to_csv(
        experiment_settings.RESULTS_DIR / "summary_idle.csv", index=False
    )
    print("## Summary (measured requests only; percentiles by linear interpolation)\n")
    print(to_markdown(request_summary))
    print("\n## CPU frequency / temperature during each run\n")
    print(to_markdown(summarize_cpu_readings(runs)))

    plot_response_time_and_frequency(
        runs, experiment_settings.FIGURES_DIR / "idle_response_time_and_freq.png"
    )
    plot_response_time_cdf(runs, experiment_settings.FIGURES_DIR / "idle_response_time_cdf.png")
    print("\nWrote results/summary_idle.csv and figures/*.png")


# --- Loading and summarizing -------------------------------------------------------


def load_run(label: str, results_folder: str, run_id: str, color: str) -> Run:
    folder = experiment_settings.RESULTS_DIR / results_folder
    all_requests = pd.read_csv(folder / csv_format.RAW_CSV_NAME)
    all_cpu_readings = pd.read_csv(folder / csv_format.FREQUENCY_CSV_NAME)

    requests = all_requests[all_requests[csv_format.RUN_ID] == run_id]
    return Run(
        label=label,
        color=color,
        requests=requests[requests[csv_format.PHASE] == csv_format.PHASE_MEASURED].reset_index(
            drop=True
        ),
        cpu_readings=all_cpu_readings[all_cpu_readings[csv_format.RUN_ID] == run_id].reset_index(
            drop=True
        ),
    )


def summarize_requests(runs: list[Run]) -> pd.DataFrame:
    """Mean, std and percentiles of TTFT, response time and tokens/s for every run."""
    rows = []
    for run in runs:
        for column, metric_label in SUMMARY_METRICS:
            values = run.requests[column]
            rows.append(
                {
                    "run": run.label,
                    "metric": metric_label,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "p50": values.quantile(0.5),
                    "p95": values.quantile(0.95),
                    "p99": values.quantile(0.99),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def summarize_cpu_readings(runs: list[Run]) -> pd.DataFrame:
    rows = []
    for run in runs:
        readings = run.cpu_readings
        frequency = readings[csv_format.FREQ_MHZ]
        rows.append(
            {
                "run": run.label,
                "samples": len(readings),
                "freq_min_mhz": frequency.min(),
                "freq_median_mhz": frequency.median(),
                "freq_max_mhz": frequency.max(),
                "samples_below_1500mhz": int(
                    (frequency < experiment_settings.LOW_FREQUENCY_THRESHOLD_MHZ).sum()
                ),
                "temp_min_c": readings[csv_format.TEMP_C].min(),
                "temp_max_c": readings[csv_format.TEMP_C].max(),
            }
        )
    return pd.DataFrame(rows)


# --- Plots ---------------------------------------------------------------------------


def plot_response_time_and_frequency(runs: list[Run], output_path: Path) -> None:
    """Response time per request above CPU frequency, sharing one request-index axis."""
    figure, (response_time_axes, frequency_axes) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12}
    )
    for run in runs:
        response_time_axes.plot(
            run.requests.index, run.requests[csv_format.RESPONSE_TIME_MS], **run.line_style
        )
        request_position, frequency_mhz = frequency_by_request_position(run)
        frequency_axes.plot(request_position, frequency_mhz, **run.line_style)

    response_time_axes.set_title(
        f"{PLOT_SCOPE}: response time and CPU frequency, 50 measured requests per run"
    )
    response_time_axes.set_ylabel("Response time per request (ms)")
    frequency_axes.set_ylabel("CPU frequency (MHz)")
    frequency_axes.set_xlabel("Request index (frequency samples placed at the request in flight)")
    for axes in (response_time_axes, frequency_axes):
        axes.set_ylim(bottom=0)
    response_time_axes.legend(loc="upper right")
    figure.savefig(output_path)
    plt.close(figure)


def plot_response_time_cdf(runs: list[Run], output_path: Path) -> None:
    """Empirical cumulative distribution of response time, one step line per run."""
    figure, axes = plt.subplots(figsize=(7.5, 4.6))
    for run in runs:
        sorted_times = np.sort(run.requests[csv_format.RESPONSE_TIME_MS].to_numpy())
        cumulative_fraction = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        axes.step(sorted_times, cumulative_fraction, where="post", **run.line_style)

    axes.set_title(f"{PLOT_SCOPE}: response time CDF")
    axes.set_xlabel("Response time (ms)")
    axes.set_ylabel("Fraction of requests ≤ response time")
    axes.set_ylim(0, 1.02)
    axes.legend(loc="lower right")
    figure.savefig(output_path)
    plt.close(figure)


# --- Contention: main flow and loading ------------------------------------------------


@dataclass
class ContentionData:
    """Everything the contention analysis needs, with quant and load added to every row."""

    requests: pd.DataFrame  # measured requests only
    cpu_readings: pd.DataFrame  # only readings taken while a config's measured requests ran
    cooldowns: pd.DataFrame


def analyze_contention() -> None:
    data = load_contention()
    requests_by_pass = with_pooled_pass(data.requests)
    summary = summarize_configs(requests_by_pass)
    deadlines = summarize_deadline_misses(data.requests)
    thermal = summarize_thermal(
        summary, with_pooled_pass(data.cpu_readings), with_pooled_pass(data.cooldowns)
    )
    pass_comparison = compare_passes(summary)

    tables = {
        "contention_summary": summary,
        "contention_deadline_misses": deadlines,
        "contention_thermal": thermal,
        "contention_pass_comparison": pass_comparison,
    }
    print("\n# Contention results (measured requests only)")
    for name, table in tables.items():
        table.round(3).to_csv(experiment_settings.RESULTS_DIR / f"{name}.csv", index=False)
        print(f"\n## {name}\n")
        print(to_markdown(table))

    figures_dir = experiment_settings.FIGURES_DIR
    plot_response_time_cdf_by_config(
        data.requests, figures_dir / "contention_response_time_cdf.png"
    )
    plot_response_time_boxplot(data.requests, figures_dir / "contention_response_time_boxplot.png")
    plot_deadline_miss_rates(deadlines, figures_dir / "contention_deadline_miss_rate.png")
    plot_heavy_response_time_and_frequency(data, figures_dir / "contention_heavy_over_time.png")
    print("\nWrote results/contention_*.csv and figures/contention_*.png")


def load_contention() -> ContentionData:
    request_frames, reading_frames, cooldown_frames = [], [], []
    for folder_name, run_id in CONTENTION_RUNS:
        folder = experiment_settings.RESULTS_DIR / folder_name
        request_frames.append(read_run_rows(folder / csv_format.RAW_CSV_NAME, run_id))
        reading_frames.append(read_run_rows(folder / csv_format.FREQUENCY_CSV_NAME, run_id))
        cooldown_frames.append(read_run_rows(folder / csv_format.COOLDOWN_CSV_NAME, run_id))

    all_requests = pd.concat(request_frames, ignore_index=True)
    requests = all_requests[all_requests[csv_format.PHASE] == csv_format.PHASE_MEASURED]
    requests = requests.reset_index(drop=True)
    # freq.csv and cooldowns.csv only name the config; add its quant and load.
    configs = requests[[*CONFIG_KEYS, csv_format.QUANT, csv_format.LOAD]].drop_duplicates()
    readings = pd.concat(reading_frames, ignore_index=True).merge(configs, on=CONFIG_KEYS)
    cooldowns = pd.concat(cooldown_frames, ignore_index=True).merge(configs, on=CONFIG_KEYS)
    return ContentionData(
        requests=requests,
        cpu_readings=keep_readings_during_requests(requests, readings),
        cooldowns=cooldowns,
    )


# --- Contention: summaries ------------------------------------------------------------


def summarize_configs(requests_by_pass: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, percentiles and jitter of every metric per config and pass."""
    rows = []
    for (quant, load, pass_label), group in requests_by_pass.groupby(CONFIG_PASS_KEYS):
        for column, metric_label in SUMMARY_METRICS:
            values = group[column]
            p50, p99 = values.quantile(0.5), values.quantile(0.99)
            rows.append(
                {
                    csv_format.QUANT: quant,
                    csv_format.LOAD: load,
                    PASS: pass_label,
                    "metric": metric_label,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "p50": p50,
                    "p95": values.quantile(0.95),
                    "p99": p99,
                    "max": values.max(),
                    "jitter (p99-p50)": p99 - p50,
                }
            )
    return sort_configs(pd.DataFrame(rows))


def summarize_deadline_misses(requests: pd.DataFrame) -> pd.DataFrame:
    """Share of requests with a response time above the deadline, per config and pass.

    D = DEADLINE_FACTOR x the idle p50 of the same quantization, pooled over all passes.
    The fixed deadline (FIXED_DEADLINE_MS) is reported next to it.
    """
    response_time = requests[csv_format.RESPONSE_TIME_MS]
    idle_p50 = response_time[requests[csv_format.LOAD] == BASELINE_LOAD].groupby(
        requests[csv_format.QUANT]
    )
    deadline_ms = requests[csv_format.QUANT].map(idle_p50.median() * DEADLINE_FACTOR)
    judged = requests.assign(
        deadline_ms=deadline_ms,
        missed_relative=response_time > deadline_ms,
        missed_fixed=response_time > FIXED_DEADLINE_MS,
    )
    rows = []
    for (quant, load, pass_label), group in with_pooled_pass(judged).groupby(CONFIG_PASS_KEYS):
        rows.append(
            {
                csv_format.QUANT: quant,
                csv_format.LOAD: load,
                PASS: pass_label,
                "n": len(group),
                "deadline_D_ms": group["deadline_ms"].iloc[0],
                "misses_vs_D": int(group["missed_relative"].sum()),
                "miss_pct_vs_D": 100 * group["missed_relative"].mean(),
                "misses_vs_fixed": int(group["missed_fixed"].sum()),
                "miss_pct_vs_fixed": 100 * group["missed_fixed"].mean(),
            }
        )
    return sort_configs(pd.DataFrame(rows))


def summarize_thermal(
    summary: pd.DataFrame, readings_by_pass: pd.DataFrame, cooldowns_by_pass: pd.DataFrame
) -> pd.DataFrame:
    """Response time next to CPU frequency, temperature and cool-down of the same config.

    Lets a contention effect be told apart from throttling.
    """
    response = summary[summary["metric"] == RESPONSE_TIME_LABEL]
    response = response[[*CONFIG_PASS_KEYS, "p50", "p99"]].rename(
        columns={"p50": "response_p50_ms", "p99": "response_p99_ms"}
    )
    frequency = readings_by_pass.groupby(CONFIG_PASS_KEYS).agg(
        freq_min_mhz=(csv_format.FREQ_MHZ, "min"),
        freq_median_mhz=(csv_format.FREQ_MHZ, "median"),
        temp_max_c=(csv_format.TEMP_C, "max"),
    )
    cooldown = cooldowns_by_pass.assign(
        target_missed=cooldowns_by_pass[csv_format.COOLDOWN_TARGET_REACHED].astype(str) != "True"
    )
    cooldown = cooldown.groupby(CONFIG_PASS_KEYS).agg(
        cooldown_s_mean=(csv_format.COOLDOWN_S, "mean"),
        config_start_temp_max_c=(csv_format.COOLDOWN_END_TEMP_C, "max"),
        cooldowns_target_missed=("target_missed", "sum"),
    )
    table = response.merge(frequency, on=CONFIG_PASS_KEYS).merge(cooldown, on=CONFIG_PASS_KEYS)
    return sort_configs(table)


def compare_passes(summary: pd.DataFrame) -> pd.DataFrame:
    """For configs measured in several passes: how far apart are the passes' p50 and p99?"""
    response = summary[(summary["metric"] == RESPONSE_TIME_LABEL) & (summary[PASS] != ALL_PASSES)]
    rows = []
    for (quant, load), group in response.groupby([csv_format.QUANT, csv_format.LOAD]):
        if len(group) < 2:
            continue
        p50_ratio = group["p50"].max() / group["p50"].min()
        p99_ratio = group["p99"].max() / group["p99"].min()
        rows.append(
            {
                csv_format.QUANT: quant,
                csv_format.LOAD: load,
                "p50_lowest_ms": group["p50"].min(),
                "p50_highest_ms": group["p50"].max(),
                "p50_ratio": p50_ratio,
                "p99_lowest_ms": group["p99"].min(),
                "p99_highest_ms": group["p99"].max(),
                "p99_ratio": p99_ratio,
                "passes_disagree": max(p50_ratio, p99_ratio) > 1 + PASS_DISAGREEMENT_FRACTION,
            }
        )
    return sort_configs(pd.DataFrame(rows))


# --- Contention: plots ---------------------------------------------------------------


def plot_response_time_cdf_by_config(requests: pd.DataFrame, output_path: Path) -> None:
    """Empirical CDF of response time, one line per config (passes pooled)."""
    figure, axes = plt.subplots(figsize=(8, 5))
    for quant, load, group in iterate_configs(requests):
        sorted_times = np.sort(group[csv_format.RESPONSE_TIME_MS].to_numpy())
        cumulative_fraction = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        axes.step(
            sorted_times,
            cumulative_fraction,
            where="post",
            color=QUANT_COLORS[quant],
            linestyle=LOAD_LINE_STYLES[load],
            label=f"{quant}, {load}",
        )
    axes.set_xscale("log")
    axes.set_title("Response time CDF per config (log scale)")
    axes.set_xlabel("Response time (ms)")
    axes.set_ylabel("Fraction of requests ≤ response time")
    axes.set_ylim(0, 1.02)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.savefig(output_path)
    plt.close(figure)


def plot_response_time_boxplot(requests: pd.DataFrame, output_path: Path) -> None:
    """Response time by load, one box per quantization (passes pooled)."""
    loads = [load for load in LOAD_ORDER if (requests[csv_format.LOAD] == load).any()]
    box_width = 0.32
    figure, axes = plt.subplots(figsize=(7.5, 4.8))
    for load_index, load in enumerate(loads):
        for quant_index, quant in enumerate(QUANT_ORDER):
            in_config = (requests[csv_format.LOAD] == load) & (requests[csv_format.QUANT] == quant)
            values = requests.loc[in_config, csv_format.RESPONSE_TIME_MS]
            if values.empty:
                continue
            offset = (quant_index - (len(QUANT_ORDER) - 1) / 2) * (box_width + 0.04)
            axes.boxplot(
                values,
                positions=[load_index + offset],
                widths=box_width,
                patch_artist=True,
                boxprops={"facecolor": QUANT_COLORS[quant], "edgecolor": SECONDARY_INK},
                medianprops={"color": PRIMARY_INK},
                flierprops={
                    "markersize": 3,
                    "markerfacecolor": MUTED_INK,
                    "markeredgecolor": "none",
                },
            )
    axes.set_yscale("log")
    axes.set_xticks(range(len(loads)), loads)
    axes.grid(axis="x", visible=False)
    axes.set_title("Response time by CPU load (log scale)")
    axes.set_xlabel("CPU load")
    axes.set_ylabel("Response time (ms)")
    axes.legend(
        handles=[Patch(facecolor=QUANT_COLORS[quant], label=quant) for quant in QUANT_ORDER],
        loc="upper left",
    )
    figure.savefig(output_path)
    plt.close(figure)


def plot_deadline_miss_rates(deadlines: pd.DataFrame, output_path: Path) -> None:
    """Bars of the deadline miss rate per load and quantization (passes pooled)."""
    pooled = deadlines[deadlines[PASS] == ALL_PASSES]
    loads = [load for load in LOAD_ORDER if (pooled[csv_format.LOAD] == load).any()]
    bar_width = 0.36
    panels = [
        ("miss_pct_vs_D", "misses_vs_D", f"Deadline D = {DEADLINE_FACTOR} × idle p50 (per quant)"),
        ("miss_pct_vs_fixed", "misses_vs_fixed", f"Fixed deadline {FIXED_DEADLINE_MS} ms"),
    ]
    figure, all_axes = plt.subplots(1, len(panels), figsize=(10, 4.4), sharey=True)
    for axes, (rate_column, count_column, title) in zip(all_axes, panels, strict=True):
        for quant_index, quant in enumerate(QUANT_ORDER):
            rows = pooled[pooled[csv_format.QUANT] == quant].set_index(csv_format.LOAD)
            for load_index, load in enumerate(loads):
                if load not in rows.index:
                    continue
                row = rows.loc[load]
                x = load_index + (quant_index - (len(QUANT_ORDER) - 1) / 2) * bar_width
                axes.bar(
                    x, row[rate_column], bar_width, color=QUANT_COLORS[quant],
                    label=quant if load_index == 0 else None,
                )  # fmt: skip
                axes.annotate(
                    f"{row[count_column]}/{row['n']}",
                    (x, row[rate_column]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=SECONDARY_INK,
                )
        axes.set_xticks(range(len(loads)), loads)
        axes.grid(axis="x", visible=False)
        axes.set_title(title, fontsize=10)
        axes.set_xlabel("CPU load")
    all_axes[0].set_ylabel("Requests over the deadline (%)  [labels: misses/requests]")
    all_axes[0].set_ylim(0, 108)
    all_axes[0].legend(loc="upper left")
    figure.savefig(output_path)
    plt.close(figure)


def plot_heavy_response_time_and_frequency(data: ContentionData, output_path: Path) -> None:
    """Heavy configs: response time of each request above the CPU frequency, over real time."""
    heavy_requests = data.requests[data.requests[csv_format.LOAD] == "heavy"]
    quants = [quant for quant in QUANT_ORDER if (heavy_requests[csv_format.QUANT] == quant).any()]
    if not quants:
        return
    figure, all_axes = plt.subplots(
        2,
        len(quants),
        figsize=(5.5 * len(quants), 6),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    for column, quant in enumerate(quants):
        response_time_axes, frequency_axes = all_axes[0, column], all_axes[1, column]
        color = QUANT_COLORS[quant]
        quant_requests = heavy_requests[heavy_requests[csv_format.QUANT] == quant]
        for keys, config_requests in quant_requests.groupby(CONFIG_KEYS):
            start_time = config_requests[csv_format.SENT_AT_S].min()
            finish_s = (
                config_requests[csv_format.SENT_AT_S]
                - start_time
                + config_requests[csv_format.RESPONSE_TIME_MS] / 1e3
            )
            response_time_axes.plot(
                finish_s, config_requests[csv_format.RESPONSE_TIME_MS], marker="o", color=color
            )
            readings = readings_of_config(data.cpu_readings, keys)
            frequency_axes.plot(
                readings[csv_format.READING_TIME_S] - start_time,
                readings[csv_format.FREQ_MHZ],
                color=color,
            )
        process_count = experiment_settings.LOAD_PROCESS_COUNTS["heavy"]
        response_time_axes.set_title(f"{quant}, heavy load ({process_count} busy processes)")
        response_time_axes.set_ylim(bottom=0)
        frequency_axes.set_ylim(bottom=0)
        frequency_axes.set_xlabel("Time since the first measured request was sent (s)")
    all_axes[0, 0].set_ylabel("Response time (ms)")
    all_axes[1, 0].set_ylabel("CPU frequency (MHz)")
    figure.savefig(output_path)
    plt.close(figure)


# --- Helpers---------------------------------


def to_markdown(table: pd.DataFrame, digits: int = 1) -> str:
    """Markdown table text (pandas' own `to_markdown` would need the `tabulate` package)."""
    header = "| " + " | ".join(table.columns) + " |"
    separator = "|" + "|".join("---" for _ in table.columns) + "|"
    lines = [header, separator]
    for _, row in table.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_run_rows(path: Path, run_id: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    return rows[rows[csv_format.RUN_ID] == run_id]


def with_pooled_pass(rows: pd.DataFrame) -> pd.DataFrame:
    """Each row once per pass ("1", "2", ...) and once more as ALL_PASSES, for uniform grouping."""
    per_pass = rows.assign(**{PASS: (rows[csv_format.REPEAT] + 1).astype(str)})
    pooled = rows.assign(**{PASS: ALL_PASSES})
    return pd.concat([per_pass, pooled], ignore_index=True)


def sort_configs(table: pd.DataFrame) -> pd.DataFrame:
    """Order rows by quantization, load (plan order), then pass; otherwise keeps row order."""
    sort_keys = {
        "quant_rank": table[csv_format.QUANT].map(QUANT_ORDER.index),
        "load_rank": table[csv_format.LOAD].map(LOAD_ORDER.index),
    }
    by = [*sort_keys, PASS] if PASS in table else list(sort_keys)
    sorted_table = table.assign(**sort_keys).sort_values(by, kind="stable")
    return sorted_table.drop(columns=list(sort_keys)).reset_index(drop=True)


def iterate_configs(requests: pd.DataFrame):
    """Yield (quant, load, that config's requests) in plan order, skipping absent configs."""
    for quant in QUANT_ORDER:
        for load in LOAD_ORDER:
            in_config = (requests[csv_format.QUANT] == quant) & (requests[csv_format.LOAD] == load)
            if in_config.any():
                yield quant, load, requests[in_config]


def keep_readings_during_requests(requests: pd.DataFrame, readings: pd.DataFrame) -> pd.DataFrame:
    """Per config, only the CPU readings between its first measured request and its last."""
    kept = []
    for keys, config_requests in requests.groupby(CONFIG_KEYS):
        start_time = config_requests[csv_format.SENT_AT_S].min()
        end_time = (
            config_requests[csv_format.SENT_AT_S]
            + config_requests[csv_format.RESPONSE_TIME_MS] / 1e3
        ).max()
        config_readings = readings_of_config(readings, keys)
        during_requests = config_readings[csv_format.READING_TIME_S].between(start_time, end_time)
        kept.append(config_readings[during_requests])
    return pd.concat(kept, ignore_index=True)


def readings_of_config(readings: pd.DataFrame, config_keys: tuple) -> pd.DataFrame:
    """The rows of one config run; `config_keys` is a value for each of CONFIG_KEYS."""
    return readings[(readings[CONFIG_KEYS] == list(config_keys)).all(axis=1)]


def frequency_by_request_position(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """CPU frequency readings as (position on the request-index axis, MHz).

    Readings taken before the first or after the last measured request are dropped.
    """
    readings = run.cpu_readings.dropna(subset=[csv_format.FREQ_MHZ])
    position = request_position_at(run.requests, readings[csv_format.READING_TIME_S].to_numpy())
    is_during_requests = ~np.isnan(position)
    return position[is_during_requests], readings[csv_format.FREQ_MHZ].to_numpy()[
        is_during_requests
    ]


def request_position_at(requests: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Fractional request index at each time: request i spans [i, i + 1)."""
    start_times = requests[csv_format.SENT_AT_S].to_numpy()
    last_end_time = start_times[-1] + requests[csv_format.RESPONSE_TIME_MS].iloc[-1] / 1e3
    boundary_times = np.append(start_times, last_end_time)
    boundary_indices = np.arange(len(boundary_times), dtype=float)
    return np.interp(times, boundary_times, boundary_indices, left=np.nan, right=np.nan)


if __name__ == "__main__":
    main()

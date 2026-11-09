"""Run the experiment: every quantization x CPU load config, in random order, with a cool-down.

Writes raw.csv, freq.csv, cooldowns.csv and meta_<run_id>.json to --out (names: `csv_format.py`).
A config's rows are written only when it has finished, so an interrupted run can be continued
by repeating the same command (same --run-id): finished configs are skipped.
"""

import argparse
import contextlib
import csv
import ctypes
import hashlib
import json
import platform
import random
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil

from slm_bench import cpu_recorder, csv_format, experiment_settings
from slm_bench.llama_server import LlamaServer, StreamedCompletion

# Windows 11 "power mode" slider settings are overlays on top of the power scheme.
POWER_MODE_OVERLAY_NAMES = {
    "00000000-0000-0000-0000-000000000000": "none (Balanced / Recommended)",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


@dataclass
class ConfigSlot:
    """One entry of the run plan: a quantization at a load level."""

    repeat: int  # which full pass over all configs this belongs to
    position: int  # index in the overall run order
    quant: str
    load: str

    @property
    def name(self) -> str:
        return f"{self.quant}_{self.load}"

    def identity_columns(self, run_id: str) -> dict:
        """The CSV columns (in every output CSV) saying where a row belongs."""
        return {
            csv_format.RUN_ID: run_id,
            csv_format.REPEAT: self.repeat,
            csv_format.ORDER_POSITION: self.position,
            csv_format.CONFIG: self.name,
        }


# --- Main flow ------------------------------------------------------------------


def main() -> None:
    args = parse_arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = plan_config_order(args.quants, args.loads, args.repeats, args.seed)
    print("Config order:", [f"r{slot.repeat}:{slot.name}" for slot in plan])

    finished_positions = read_finished_positions(args.out, args.run_id)
    remaining = [slot for slot in plan if slot.position not in finished_positions]
    if not remaining:
        print(f"All {len(plan)} configs of run '{args.run_id}' are already finished.")
        return
    if finished_positions:
        print(f"Resuming run '{args.run_id}': {len(finished_positions)} of {len(plan)} done.")

    metadata_path = args.out / f"meta_{args.run_id}.json"
    if metadata_path.exists():
        metadata = load_metadata_for_resume(metadata_path, args, plan)
    else:
        metadata = collect_metadata(args, plan)
    metadata_path.write_text(json.dumps(metadata, indent=2))  # kept even if the run is stopped

    for slot in remaining:
        print(f"[{slot.position + 1}/{len(plan)}] repeat {slot.repeat}: {slot.quant} / {slot.load}")
        cooldown_row = cool_down(slot, args)
        recorder = run_config(slot, args, cooldown_row)
        metadata["freq_sampling"] = {
            "has_freq": recorder.has_frequency,
            "has_temp": recorder.has_temperature,
        }

    metadata["finished"] = datetime.now().isoformat(timespec="seconds")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print("Done. Wrote", args.out)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quants", nargs="+", choices=experiment_settings.MODEL_FILES, default=["q8_0", "q4_k_m"]
    )
    parser.add_argument(
        "--loads",
        nargs="+",
        choices=experiment_settings.LOAD_PROCESS_COUNTS,
        default=list(experiment_settings.LOAD_PROCESS_COUNTS),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=experiment_settings.WARMUP_REQUESTS,
        help="discarded requests per config",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=experiment_settings.MEASURED_REQUESTS,
        help="measured requests per config",
    )
    parser.add_argument(
        "--cooldown-min",
        type=float,
        default=experiment_settings.COOLDOWN_MIN_S,
        help="minimum seconds between configs",
    )
    parser.add_argument(
        "--cooldown-max",
        type=float,
        default=experiment_settings.COOLDOWN_MAX_S,
        help="maximum seconds between configs, even if the CPU is still warm",
    )
    parser.add_argument("--repeats", type=int, default=1, help="full passes over all configs")
    parser.add_argument(
        "--seed",
        type=int,
        default=experiment_settings.CONFIG_ORDER_SEED,
        help="seed for the config order",
    )
    parser.add_argument("--out", type=Path, default=experiment_settings.RESULTS_DIR)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def plan_config_order(
    quants: list[str], loads: list[str], repeats: int, seed: int
) -> list[ConfigSlot]:
    """All quant x load configs, shuffled differently (but reproducibly) per repeat."""
    plan = []
    for repeat in range(repeats):
        configs = [(quant, load) for quant in quants for load in loads]
        random.Random(seed + repeat).shuffle(configs)
        for quant, load in configs:
            plan.append(ConfigSlot(repeat, len(plan), quant, load))
    return plan


def read_finished_positions(out_dir: Path, run_id: str) -> set[int]:
    """Run-order positions of the configs of `run_id` that have been completely written.

    cooldowns.csv is the last file written for a config, so it marks the config as finished.
    """
    path = out_dir / csv_format.COOLDOWN_CSV_NAME
    if not path.exists():
        return set()
    with open(path, newline="") as file:
        return {
            int(row[csv_format.ORDER_POSITION])
            for row in csv.DictReader(file)
            if row[csv_format.RUN_ID] == run_id
        }


# --- Cooling down and running one config -------------------------------------------


def cool_down(slot: ConfigSlot, args: argparse.Namespace) -> dict:
    """Wait until the CPU is cool before a config; returns the config's cooldowns.csv row.

    The first config of a run does not wait. A missing temperature sensor means the minimum wait.
    """
    min_wait_s, max_wait_s = (
        (0, 0) if slot.position == 0 else (args.cooldown_min, args.cooldown_max)
    )
    sensor = cpu_recorder.open_temperature_counter()
    start_temperature = cpu_recorder.read_temperature_c(sensor)
    if max_wait_s > 0:
        print(
            f"  Cooling down {min_wait_s:.0f}-{max_wait_s:.0f} s from {format_c(start_temperature)}"
        )

    started_at = time.monotonic()
    while True:
        waited_s = time.monotonic() - started_at
        temperature = cpu_recorder.read_temperature_c(sensor)
        if cooldown_is_over(waited_s, temperature, min_wait_s, max_wait_s):
            break
        time.sleep(experiment_settings.COOLDOWN_CHECK_INTERVAL_S)

    target_reached = (
        "" if temperature is None else temperature < experiment_settings.COOLDOWN_TARGET_TEMP_C
    )
    print(f"  Waited {waited_s:.0f} s; CPU is now {format_c(temperature)}")
    return {
        **slot.identity_columns(args.run_id),
        csv_format.COOLDOWN_S: round(waited_s, 1),
        csv_format.COOLDOWN_START_TEMP_C: rounded_or_blank(start_temperature, 2),
        csv_format.COOLDOWN_END_TEMP_C: rounded_or_blank(temperature, 2),
        csv_format.COOLDOWN_TARGET_REACHED: target_reached,
    }


def run_config(
    slot: ConfigSlot, args: argparse.Namespace, cooldown_row: dict
) -> cpu_recorder.CpuReadingRecorder:
    """Run one config start to finish, then write all of its rows."""
    log_path = args.out / f"server_{args.run_id}_{slot.repeat}_{slot.position}_{slot.name}.log"
    recorder = cpu_recorder.CpuReadingRecorder(experiment_settings.CPU_READING_INTERVAL_S)
    with (
        apply_cpu_load(slot.load),
        LlamaServer(experiment_settings.MODEL_FILES[slot.quant], log_path) as server,
    ):
        recorder.start()
        try:
            request_rows = run_phase(
                slot, args.run_id, csv_format.PHASE_WARMUP, args.warmup, server
            )
            request_rows += run_phase(slot, args.run_id, csv_format.PHASE_MEASURED, args.n, server)
        finally:
            recorder.stop()

    # Written together and with cooldowns.csv last: that file marks the config as finished.
    append_rows(args.out / csv_format.RAW_CSV_NAME, csv_format.RAW_COLUMNS, request_rows)
    append_rows(
        args.out / csv_format.FREQUENCY_CSV_NAME,
        csv_format.FREQUENCY_COLUMNS,
        build_frequency_rows(slot, args.run_id, recorder.readings),
    )
    append_rows(
        args.out / csv_format.COOLDOWN_CSV_NAME, csv_format.COOLDOWN_COLUMNS, [cooldown_row]
    )
    return recorder


@contextlib.contextmanager
def apply_cpu_load(load: str):
    """Keep the busy-loop processes for `load` running while the block executes."""
    process_count = experiment_settings.LOAD_PROCESS_COUNTS[load]
    if process_count == 0:
        yield
        return
    from slm_bench import cpu_stress  # imported here because idle runs never need it

    with cpu_stress.Stressor(process_count):
        yield


def run_phase(
    slot: ConfigSlot, run_id: str, phase: str, request_count: int, server: LlamaServer
) -> list[dict]:
    """Send `request_count` sequential requests; returns one raw.csv row for each."""
    rows = []
    for request_index in range(request_count):
        completion = server.stream()
        row = build_request_row(slot, run_id, phase, request_index, completion)
        rows.append(row)
        if row[csv_format.N_TOKENS] != experiment_settings.MAX_NEW_TOKENS:
            print(
                f"  WARNING: {slot.name} {phase}#{request_index} generated "
                f"{row[csv_format.N_TOKENS]} tokens, expected {experiment_settings.MAX_NEW_TOKENS}"
            )
        is_progress_report = (request_index + 1) % experiment_settings.PROGRESS_REPORT_EVERY == 0
        if phase == csv_format.PHASE_MEASURED and is_progress_report:
            print(
                f"  {slot.name}: {request_index + 1}/{request_count} measured, "
                f"last response time {row[csv_format.RESPONSE_TIME_MS]:.0f} ms"
            )
    return rows


def build_request_row(
    slot: ConfigSlot,
    run_id: str,
    phase: str,
    request_index: int,
    completion: StreamedCompletion,
) -> dict:
    """Turn one streamed completion into a raw.csv row (times in ms, rates in tok/s)."""
    arrivals = completion.chunk_arrival_times
    server_timings = completion.server_timings or {}
    tokens = completion.tokens_predicted

    end_to_end_rate = tokens / completion.total_time if tokens else None
    has_decode_interval = len(arrivals) > 1
    decode_rate = (
        (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if has_decode_interval else None
    )
    return {
        **slot.identity_columns(run_id),
        csv_format.QUANT: slot.quant,
        csv_format.LOAD: slot.load,
        csv_format.PHASE: phase,
        csv_format.REQUEST_INDEX: request_index,
        csv_format.SENT_AT_S: round(completion.sent_at, 6),
        csv_format.TTFT_MS: seconds_to_ms(arrivals[0]) if arrivals else "",
        csv_format.RESPONSE_TIME_MS: seconds_to_ms(completion.total_time),
        csv_format.N_TOKENS: tokens,
        csv_format.N_CHUNKS: len(arrivals),
        csv_format.TOK_PER_S_E2E: rounded_or_blank(end_to_end_rate, 3),
        csv_format.TOK_PER_S_DECODE: rounded_or_blank(decode_rate, 3),
        csv_format.SERVER_PROMPT_MS: server_timings.get("prompt_ms", ""),
        csv_format.SERVER_PREDICTED_MS: server_timings.get("predicted_ms", ""),
        csv_format.SERVER_PREDICTED_PER_S: server_timings.get("predicted_per_second", ""),
        csv_format.TOKEN_TIMES_MS: json.dumps([seconds_to_ms(t, digits=2) for t in arrivals]),
    }


def build_frequency_rows(
    slot: ConfigSlot, run_id: str, readings: list[cpu_recorder.CpuReading]
) -> list[dict]:
    return [
        {
            **slot.identity_columns(run_id),
            csv_format.READING_TIME_S: round(reading.time, 3),
            csv_format.FREQ_MHZ: rounded_or_blank(reading.frequency_mhz, 1),
            csv_format.TEMP_C: rounded_or_blank(reading.temperature_c, 2),
        }
        for reading in readings
    ]


# --- Run metadata -------------------------------------------------------------------


def collect_metadata(args: argparse.Namespace, plan: list[ConfigSlot]) -> dict:
    """Everything needed to interpret (and repeat) this run, as plain JSON data."""
    quants_in_use = sorted({slot.quant for slot in plan})
    return {
        "run_id": args.run_id,
        "started": datetime.now().isoformat(timespec="seconds"),
        "cpu": read_cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "power_plan": run_command(["powercfg", "/getactivescheme"]),
        "power_mode_overlay": read_power_mode_overlay(),
        "on_ac_power": read_on_ac_power(),
        "python": platform.python_version(),
        "llama_server_version_output": run_command(
            [str(experiment_settings.SERVER_EXECUTABLE), "--version"]
        ),
        "models": {
            quant: {
                "file": experiment_settings.MODEL_FILES[quant].name,
                "sha256": sha256_of_file(experiment_settings.MODEL_FILES[quant]),
            }
            for quant in quants_in_use
        },
        "settings": read_experiment_settings(),
        "arguments": read_run_arguments(args),
        "freq_sampling": {"has_freq": None, "has_temp": None},  # filled in after the runs
        "config_order": describe_plan(plan),
    }


def load_metadata_for_resume(path: Path, args: argparse.Namespace, plan: list[ConfigSlot]) -> dict:
    """Metadata of the run being continued, plus a note on when and under which power mode."""
    metadata = json.loads(path.read_text())
    previous = metadata["arguments"]
    if (
        metadata["config_order"] != describe_plan(plan)
        or previous["n"] != args.n
        or previous["warmup"] != args.warmup
    ):
        raise SystemExit(f"{path.name} was made with a different plan; use a new --run-id.")
    metadata.pop("finished", None)
    metadata.setdefault("resumed", []).append(
        {
            "at": datetime.now().isoformat(timespec="seconds"),
            "power_mode_overlay": read_power_mode_overlay(),
            "on_ac_power": read_on_ac_power(),
        }
    )
    return metadata


def describe_plan(plan: list[ConfigSlot]) -> list[dict]:
    return [
        {"repeat": slot.repeat, "pos": slot.position, "quant": slot.quant, "load": slot.load}
        for slot in plan
    ]


def read_experiment_settings() -> dict:
    """Every ALL_CAPS setting of experiment_settings.py, so a new setting is recorded automatically.

    Paths are left out: they differ per machine and would reveal the user name.
    """
    return {
        name.lower(): value
        for name, value in vars(experiment_settings).items()
        if name.isupper() and not contains_path(value)
    }


def read_run_arguments(args: argparse.Namespace) -> dict:
    """The command-line values this run actually used (the run id is stored separately)."""
    return {
        name: value
        for name, value in vars(args).items()
        if name != "run_id" and not contains_path(value)
    }


def read_power_mode_overlay() -> dict:
    """The Windows power-mode overlay that is currently in effect.

    `powercfg /getactivescheme` cannot see it, so ask the power API directly.
    """
    guid_bytes = (ctypes.c_ubyte * 16)()
    status = ctypes.WinDLL("powrprof.dll").PowerGetEffectiveOverlayScheme(guid_bytes)
    if status != 0:
        return {"guid": None, "name": f"unavailable (rc={status})"}
    guid = str(uuid.UUID(bytes_le=bytes(guid_bytes)))
    return {"guid": guid, "name": POWER_MODE_OVERLAY_NAMES.get(guid, "unknown")}


def read_on_ac_power() -> bool | None:
    return getattr(psutil.sensors_battery(), "power_plugged", None)


def read_cpu_name() -> str:
    name = run_command(
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
    )
    return name or platform.processor()


# --- Helpers ---------------------------------


def append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Append rows to a CSV file; the header is written only if the file is still empty."""
    needs_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, columns)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def cooldown_is_over(
    waited_s: float, temperature: float | None, min_wait_s: float, max_wait_s: float
) -> bool:
    if waited_s >= max_wait_s:
        return True
    is_cool = temperature is None or temperature < experiment_settings.COOLDOWN_TARGET_TEMP_C
    return waited_s >= min_wait_s and is_cool


def format_c(temperature: float | None) -> str:
    return "n/a" if temperature is None else f"{temperature:.1f} °C"


def seconds_to_ms(seconds: float, digits: int = 3) -> float:
    return round(seconds * 1e3, digits)


def rounded_or_blank(value: float | None, digits: int) -> float | str:
    """Round `value`; an empty CSV cell when there is none."""
    return "" if value is None else round(value, digits)


def contains_path(value) -> bool:
    """True for a Path, or a dict with Paths among its values."""
    values = value.values() if isinstance(value, dict) else [value]
    return any(isinstance(item, Path) for item in values)


def run_command(command: list[str]) -> str:
    """Stdout and stderr of `command` (llama-server prints its version to stderr)."""
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    return (result.stdout + result.stderr).strip()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

"""All experiment settings; each ALL_CAPS constant (except paths) is saved in the run metadata."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # the repo folder
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# --- llama.cpp build and models (both are gitignored downloads) ---------------
LLAMA_CPP_BUILD = "b12177"  # pinned release so results stay comparable
SERVER_EXECUTABLE = PROJECT_ROOT / "bin" / LLAMA_CPP_BUILD / "llama-server.exe"
MODEL_FILES = {
    "q8_0": PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf",
    "q4_k_m": PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

# --- Server settings (identical for every run) --------------------------------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8089
SERVER_THREADS = 4
CONTEXT_SIZE = 2048
SERVER_PARALLEL_SLOTS = 1  # one request at a time: no slot sharing
SERVER_STARTUP_TIMEOUT_S = 120
SERVER_STOP_TIMEOUT_S = 10
HEALTH_POLL_INTERVAL_S = 0.5
HEALTH_REQUEST_TIMEOUT_S = 2

# --- Request settings (identical for every request) ---------------------------
PROMPT = "Explain in simple terms why predictable response time matters for real-time systems."
MAX_NEW_TOKENS = 64
TEMPERATURE = 0
SEED = 42
# Never stop early at an end-of-sequence token: every request must generate exactly
# MAX_NEW_TOKENS, so output length cannot change the response time.
IGNORE_EOS = True
# Don't reuse the previous request's KV cache: with an identical prompt that would
# skip prompt processing and make later requests cheaper than the first.
CACHE_PROMPT = False
REQUEST_TIMEOUT_S = 300

# --- Load levels: number of busy-loop processes (16 logical cores here) --------
LOAD_PROCESS_COUNTS = {"idle": 0, "medium": 8, "heavy": 16}

# --- Experiment defaults (each can be overridden on the command line) ----------
WARMUP_REQUESTS = 5
MEASURED_REQUESTS = 50
COOLDOWN_MIN_S = 60  # between configs: always wait at least this long ...
COOLDOWN_MAX_S = 300  # ... and never longer than this ...
COOLDOWN_TARGET_TEMP_C = 60  # ... stopping once the CPU is below this temperature
COOLDOWN_CHECK_INTERVAL_S = 5
CONFIG_ORDER_SEED = 1234
CPU_READING_INTERVAL_S = 1.0
PROGRESS_REPORT_EVERY = 10  # requests

# --- Analysis ------------------------------------------------------------------
LOW_FREQUENCY_THRESHOLD_MHZ = 1500  # samples below this count as a frequency dip

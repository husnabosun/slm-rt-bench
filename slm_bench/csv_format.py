"""File, column and phase names of the CSVs, shared by `run_experiment.py` and `analyze.py`."""

RAW_CSV_NAME = "raw.csv"  # one row per request
FREQUENCY_CSV_NAME = "freq.csv"  # one row per CPU reading
COOLDOWN_CSV_NAME = "cooldowns.csv"  # one row per config: the wait before it started

# Columns in all three files: which run and config a row belongs to.
RUN_ID = "run_id"
REPEAT = "repeat"
ORDER_POSITION = "order_pos"
CONFIG = "config"

# Columns of raw.csv.
QUANT = "quant"
LOAD = "load"
PHASE = "phase"
REQUEST_INDEX = "req_idx"
SENT_AT_S = "t_start_s"  # time.perf_counter() when the request was sent
TTFT_MS = "ttft_ms"
RESPONSE_TIME_MS = "response_time_ms"
N_TOKENS = "n_tokens"
N_CHUNKS = "n_chunks"
TOK_PER_S_E2E = "tok_per_s_e2e"
TOK_PER_S_DECODE = "tok_per_s_decode"
SERVER_PROMPT_MS = "srv_prompt_ms"  # llama-server's own timings
SERVER_PREDICTED_MS = "srv_predicted_ms"
SERVER_PREDICTED_PER_S = "srv_predicted_per_s"
TOKEN_TIMES_MS = "token_times_ms"  # JSON list: arrival time of each token

# Columns of freq.csv.
READING_TIME_S = "t_s"  # time.perf_counter() of the reading, same clock as SENT_AT_S
FREQ_MHZ = "freq_mhz"
TEMP_C = "temp_c"

# Columns of cooldowns.csv. The first config of a run does not wait (cooldown_s = 0).
COOLDOWN_S = "cooldown_s"  # actual wait
COOLDOWN_START_TEMP_C = "cooldown_start_temp_c"  # temperature when the wait began
COOLDOWN_END_TEMP_C = "config_start_temp_c"  # temperature when the config started
COOLDOWN_TARGET_REACHED = "target_reached"  # True if below the target temperature at the end

# Values of the PHASE column.
PHASE_WARMUP = "warmup"  # discarded requests
PHASE_MEASURED = "measured"

RAW_COLUMNS = [
    RUN_ID, REPEAT, ORDER_POSITION, CONFIG, QUANT, LOAD, PHASE, REQUEST_INDEX,
    SENT_AT_S, TTFT_MS, RESPONSE_TIME_MS, N_TOKENS, N_CHUNKS,
    TOK_PER_S_E2E, TOK_PER_S_DECODE,
    SERVER_PROMPT_MS, SERVER_PREDICTED_MS, SERVER_PREDICTED_PER_S,
    TOKEN_TIMES_MS,
]  # fmt: skip
FREQUENCY_COLUMNS = [
    RUN_ID, REPEAT, ORDER_POSITION, CONFIG, READING_TIME_S, FREQ_MHZ, TEMP_C,
]  # fmt: skip
COOLDOWN_COLUMNS = [
    RUN_ID, REPEAT, ORDER_POSITION, CONFIG,
    COOLDOWN_S, COOLDOWN_START_TEMP_C, COOLDOWN_END_TEMP_C, COOLDOWN_TARGET_REACHED,
]  # fmt: skip

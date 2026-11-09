"""Start and stop llama-server and send it timed requests."""

import argparse
import http.client
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass

from slm_bench import experiment_settings

# llama-server's HTTP interface.
COMPLETION_ENDPOINT = "/completion"
HEALTH_ENDPOINT = "/health"
STREAM_DATA_PREFIX = b"data: "  # every streamed event line starts with this


@dataclass
class StreamedCompletion:
    """One streamed request, timed on the client with a monotonic clock."""

    sent_at: float  # time.perf_counter() just before the request was sent
    chunk_arrival_times: list[float]  # seconds after sent_at, one per token chunk
    total_time: float  # seconds from sending until the final chunk arrived
    server_timings: dict | None  # llama-server's own timings, for comparison
    tokens_predicted: int | None


class LlamaServer:
    """A `llama-server` process serving one model, usable as a context manager."""

    def __init__(self, model_path, log_path=None):
        self.base_url = (
            f"http://{experiment_settings.SERVER_HOST}:{experiment_settings.SERVER_PORT}"
        )
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self._log_file = None
        self._command = [
            str(experiment_settings.SERVER_EXECUTABLE),
            "-m", str(model_path),
            "-t", str(experiment_settings.SERVER_THREADS),
            "-c", str(experiment_settings.CONTEXT_SIZE),
            "--host", experiment_settings.SERVER_HOST,
            "--port", str(experiment_settings.SERVER_PORT),
            "--parallel", str(experiment_settings.SERVER_PARALLEL_SLOTS),
            "--no-webui",
            "--no-warmup",  # the experiment sends its own, discarded warm-up requests
        ]  # fmt: skip

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exception_info) -> None:
        self.stop()

    def start(self) -> None:
        """Launch the server and block until its health endpoint answers."""
        # The log must stay open while the server runs; stop() closes it.
        self._log_file = open(self.log_path, "wb") if self.log_path else None  # noqa: SIM115
        output = self._log_file or subprocess.DEVNULL
        self.process = subprocess.Popen(self._command, stdout=output, stderr=subprocess.STDOUT)
        self._wait_until_healthy()

    def stop(self) -> None:
        """Terminate the server (killing it if it does not exit in time)."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=experiment_settings.SERVER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def stream(self, prompt: str = experiment_settings.PROMPT) -> StreamedCompletion:
        """Send a streaming request and timestamp every token chunk on arrival.

        Streaming is what makes time-to-first-token measurable from the client.
        """
        connection = http.client.HTTPConnection(
            experiment_settings.SERVER_HOST,
            experiment_settings.SERVER_PORT,
            timeout=experiment_settings.REQUEST_TIMEOUT_S,
        )
        connection.connect()  # connect first so connection setup is not timed
        body = request_body(prompt, stream=True)
        sent_at = time.perf_counter()
        connection.request("POST", COMPLETION_ENDPOINT, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        chunk_arrival_times, final_message = read_stream(response, sent_at)
        total_time = time.perf_counter() - sent_at
        connection.close()
        return StreamedCompletion(
            sent_at=sent_at,
            chunk_arrival_times=chunk_arrival_times,
            total_time=total_time,
            server_timings=final_message.get("timings"),
            tokens_predicted=final_message.get("tokens_predicted"),
        )

    def complete(self, prompt: str = experiment_settings.PROMPT) -> tuple[dict, float]:
        """Send a non-streaming request. Returns (response JSON, wall seconds)."""
        request = urllib.request.Request(
            f"{self.base_url}{COMPLETION_ENDPOINT}",
            data=request_body(prompt, stream=False),
            headers={"Content-Type": "application/json"},
        )
        started_at = time.perf_counter()
        with urllib.request.urlopen(
            request, timeout=experiment_settings.REQUEST_TIMEOUT_S
        ) as reply:
            response_json = json.load(reply)
        return response_json, time.perf_counter() - started_at

    def _wait_until_healthy(self) -> None:
        deadline = time.time() + experiment_settings.SERVER_STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.process.returncode})")
            if self._health_check_passes():
                return
            time.sleep(experiment_settings.HEALTH_POLL_INTERVAL_S)
        raise TimeoutError("llama-server did not become healthy in time")

    def _health_check_passes(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}{HEALTH_ENDPOINT}",
                timeout=experiment_settings.HEALTH_REQUEST_TIMEOUT_S,
            ) as reply:
                return reply.status == 200
        except Exception:  # server still starting: connection refused / timeout
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=experiment_settings.MODEL_FILES, default="q4_k_m")
    arguments = parser.parse_args()
    with LlamaServer(experiment_settings.MODEL_FILES[arguments.quant]) as server:
        response_json, wall_seconds = server.complete()
    print(
        json.dumps(
            {
                "quant": arguments.quant,
                "wall_seconds": round(wall_seconds, 4),
                "tokens_predicted": response_json.get("tokens_predicted"),
                "timings": response_json.get("timings"),
                "content": response_json.get("content"),
            },
            indent=2,
        )
    )


# --- Helpers: small functions used by the code above ---------------------------------


def request_body(prompt: str, stream: bool) -> bytes:
    """JSON body of a completion request; its settings keep every request identical."""
    return json.dumps(
        {
            "prompt": prompt,
            "n_predict": experiment_settings.MAX_NEW_TOKENS,
            "temperature": experiment_settings.TEMPERATURE,
            "seed": experiment_settings.SEED,
            "ignore_eos": experiment_settings.IGNORE_EOS,
            "cache_prompt": experiment_settings.CACHE_PROMPT,
            "stream": stream,
        }
    ).encode()


def read_stream(response: http.client.HTTPResponse, sent_at: float) -> tuple[list[float], dict]:
    """Read server-sent events; return chunk arrival times and the final message."""
    chunk_arrival_times = []
    final_message: dict = {}
    while True:
        line = response.readline()
        if not line:
            break
        arrived_at = time.perf_counter()
        if not line.startswith(STREAM_DATA_PREFIX):
            continue
        message = json.loads(line[len(STREAM_DATA_PREFIX) :])
        if message.get("stop"):
            final_message = message
            break
        if message.get("content"):
            chunk_arrival_times.append(arrived_at - sent_at)
    return chunk_arrival_times, final_message


if __name__ == "__main__":
    main()

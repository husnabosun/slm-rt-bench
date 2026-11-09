"""Keep the CPU busy with N busy-loop processes; creates the medium and heavy loads."""

import argparse
import multiprocessing
import time

from slm_bench import experiment_settings


class Stressor:
    """N busy-loop processes that run for as long as the `with` block lasts."""

    def __init__(self, process_count: int):
        self.process_count = process_count
        self._processes: list[multiprocessing.Process] = []

    def __enter__(self) -> "Stressor":
        self.start()
        return self

    def __exit__(self, *exception_info) -> None:
        self.stop()

    def start(self) -> None:
        # Separate processes (not threads): Python threads in one process share a
        # single interpreter lock, so they could not keep several cores busy.
        # daemon=True makes the busy processes die with us if this script crashes.
        self._processes = [
            multiprocessing.Process(target=burn_cpu, daemon=True) for _ in range(self.process_count)
        ]
        for process in self._processes:
            process.start()

    def stop(self) -> None:
        for process in self._processes:
            process.terminate()
        for process in self._processes:
            process.join()
        self._processes = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=experiment_settings.LOAD_PROCESS_COUNTS, default="heavy")
    parser.add_argument("--seconds", type=float, default=5, help="how long to keep the load on")
    arguments = parser.parse_args()

    process_count = experiment_settings.LOAD_PROCESS_COUNTS[arguments.load]
    print(f"Starting {process_count} busy processes for {arguments.seconds:g} s ...")
    with Stressor(process_count):
        time.sleep(arguments.seconds)
    print("Stopped.")


# --- Helpers: small functions used by the code above ---------------------------------


def burn_cpu() -> None:
    """Loop forever doing nothing useful, so the core is never idle."""
    while True:
        pass


if __name__ == "__main__":
    main()

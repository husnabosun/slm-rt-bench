"""Record CPU frequency and temperature in a background thread (Windows, PDH API).

Lets a response-time spike be compared with a frequency drop at the same moment.
"""

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

PDH_FORMAT_DOUBLE = 0x00000200
KELVIN_TO_CELSIUS_OFFSET = 273.15
PDH_SUCCESS = 0

# Windows writes the names of its performance counters in the system language
# Asking for a counter by its English name therefore fails in
# PowerShell's `Get-Counter` and in `typeperf` on this machine. The PDH function
# `PdhAddEnglishCounterW` accepts English names on every Windows language, so all
# counters below are given by their English names.
PERFORMANCE_PERCENT_COUNTER = r"\Processor Information(_Total)\% Processor Performance"
NOMINAL_FREQUENCY_COUNTER = r"\Processor Information(_Total)\Processor Frequency"
TEMPERATURE_COUNTER = r"\Thermal Zone Information(*)\Temperature"  # Kelvin

_pdh = ctypes.WinDLL("pdh.dll")


class _CounterValueUnion(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double)]


class _PdhFormattedCounterValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("value", _CounterValueUnion)]


class PerformanceCounter:
    """One Windows performance counter, readable as a float."""

    def __init__(self, english_path: str):
        self._query = ctypes.c_void_p()
        self._handle = ctypes.c_void_p()
        if _pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != PDH_SUCCESS:
            raise OSError("PdhOpenQuery failed")
        status = _pdh.PdhAddEnglishCounterW(
            self._query, english_path, 0, ctypes.byref(self._handle)
        )
        if status != PDH_SUCCESS:
            raise OSError(f"counter unavailable: {english_path} (0x{status & 0xFFFFFFFF:08x})")
        # Rate counters need one earlier sample as a reference, so take it now.
        _pdh.PdhCollectQueryData(self._query)

    def read(self) -> float | None:
        """Current value, or None if Windows could not produce one."""
        _pdh.PdhCollectQueryData(self._query)
        formatted = _PdhFormattedCounterValue()
        status = _pdh.PdhGetFormattedCounterValue(
            self._handle, PDH_FORMAT_DOUBLE, None, ctypes.byref(formatted)
        )
        return formatted.value.doubleValue if status == PDH_SUCCESS else None


@dataclass
class CpuReading:
    """One reading; a field is None when that sensor is unavailable."""

    time: float  # time.perf_counter(), the same clock as the request timestamps
    frequency_mhz: float | None
    temperature_c: float | None


class CpuReadingRecorder(threading.Thread):
    """Background thread that appends a `CpuReading` to `readings` every interval."""

    def __init__(self, interval_s: float):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.readings: list[CpuReading] = []
        self._stop_requested = threading.Event()
        self._performance_percent = open_counter_if_available(PERFORMANCE_PERCENT_COUNTER)
        self._nominal_frequency = open_counter_if_available(NOMINAL_FREQUENCY_COUNTER)
        self._temperature = open_temperature_counter()
        self.has_frequency = (
            self._performance_percent is not None and self._nominal_frequency is not None
        )
        self.has_temperature = self._temperature is not None

    def run(self) -> None:
        while not self._stop_requested.wait(self.interval_s):
            self.readings.append(
                CpuReading(
                    time=time.perf_counter(),
                    frequency_mhz=self._read_frequency_mhz(),
                    temperature_c=self._read_temperature_c(),
                )
            )

    def stop(self) -> None:
        self._stop_requested.set()
        self.join()

    def _read_frequency_mhz(self) -> float | None:
        """Effective frequency = nominal MHz x '% Processor Performance' / 100.

        The percentage can exceed 100 when the CPU boosts above its nominal clock.
        """
        if not self.has_frequency:
            return None
        percent = self._performance_percent.read()
        nominal_mhz = self._nominal_frequency.read()
        if percent is None or nominal_mhz is None:
            return None
        return nominal_mhz * percent / 100.0

    def _read_temperature_c(self) -> float | None:
        return read_temperature_c(self._temperature)


def open_temperature_counter() -> PerformanceCounter | None:
    """The temperature counter, or None if the firmware does not expose a thermal zone."""
    return open_counter_if_available(TEMPERATURE_COUNTER)


def read_temperature_c(counter: PerformanceCounter | None) -> float | None:
    """Temperature in °C from a counter made by `open_temperature_counter` (None: no value)."""
    kelvin = counter.read() if counter else None
    return None if kelvin is None else kelvin - KELVIN_TO_CELSIUS_OFFSET


# --- Helpers: small functions used by the code above ---------------------------------


def open_counter_if_available(english_path: str) -> PerformanceCounter | None:
    try:
        return PerformanceCounter(english_path)
    except OSError:
        return None  # e.g. the firmware does not expose a thermal zone

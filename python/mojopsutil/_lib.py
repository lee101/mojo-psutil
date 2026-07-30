"""ctypes bindings for the Mojo parsing and rate kernels."""

from __future__ import annotations

import ctypes
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPSUTIL_LIB", os.path.join(ROOT, "dist", "libmojo-psutil.so"))
I = ctypes.c_int64
P = ctypes.c_void_p
F = ctypes.c_double

_SIGNATURES = {
    "mps_parse_table_i64": ([P, I, P, I, I], I),
    "mps_cpu_percent": ([P, P, P, I, I], I),
    "mps_cpu_times_percent": ([P, P, P, I, I], I),
    "mps_counter_rates": ([P, P, P, I, F], I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library is not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> ctypes.c_void_p:
    pointer = int(array.ctypes.data)
    if array.size and pointer == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty array")
    return ctypes.c_void_p(pointer)


def parse_table(payload: bytes, rows: int, cols: int) -> np.ndarray:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise TypeError("rows and cols must be integers")
    if rows < 0 or cols <= 0:
        raise ValueError("rows must be non-negative and cols must be positive")
    if not payload or rows == 0:
        return np.empty((0, cols), dtype=np.int64)
    source = np.frombuffer(payload, dtype=np.uint8)
    result = np.empty((rows, cols), dtype=np.int64)
    count = lib().mps_parse_table_i64(
        addr(source), source.size, addr(result), rows, cols
    )
    if count == -2:
        raise OverflowError("integer in payload does not fit in int64")
    if count < 0 or count > rows:
        raise RuntimeError(f"Mojo parser failed with status {count}")
    return result[:count]


def _cpu_inputs(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before = np.asarray(before)
    after = np.asarray(after)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("before and after must be two-dimensional")
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    if before.shape[1] != 10:
        raise ValueError("CPU arrays must have exactly 10 fields")
    return _as_float64(before, "before"), _as_float64(after, "after")


def _as_float64(array: np.ndarray, name: str) -> np.ndarray:
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if array.dtype.kind in "iu" and array.size:
        if np.any(np.abs(array.astype(object)) > 2**53):
            raise OverflowError(
                f"{name} contains integers not exactly representable as float64"
            )
    return np.ascontiguousarray(array, dtype=np.float64)


def cpu_percent(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before, after = _cpu_inputs(before, after)
    result = np.empty(before.shape[0], dtype=np.float64)
    if not result.size:
        return result
    status = lib().mps_cpu_percent(
        addr(before), addr(after), addr(result), before.shape[0], before.shape[1]
    )
    if status:
        raise RuntimeError(f"Mojo CPU-percent kernel failed with status {status}")
    return result


def cpu_times_percent(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before, after = _cpu_inputs(before, after)
    result = np.empty_like(before)
    if not result.size:
        return result
    status = lib().mps_cpu_times_percent(
        addr(before), addr(after), addr(result), before.shape[0], before.shape[1]
    )
    if status:
        raise RuntimeError(f"Mojo CPU-times-percent kernel failed with status {status}")
    return result


def counter_rates(before, after, elapsed: float) -> np.ndarray:
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    if before_array.shape != after_array.shape:
        raise ValueError("before and after must have the same shape")
    before = _as_float64(before_array, "before")
    after = _as_float64(after_array, "after")
    try:
        elapsed = float(elapsed)
    except (TypeError, ValueError) as error:
        raise TypeError("elapsed must be a real number") from error
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed must be finite and non-negative")
    result = np.empty_like(before)
    if not result.size:
        return result
    status = lib().mps_counter_rates(
        addr(before), addr(after), addr(result), before.size, elapsed
    )
    if status:
        raise RuntimeError(f"Mojo counter-rate kernel failed with status {status}")
    return result

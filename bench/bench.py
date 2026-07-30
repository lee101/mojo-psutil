"""Benchmark Mojo-backed sampling and bulk metric transformations."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojopsutil  # noqa: E402
from mojopsutil import _lib  # noqa: E402


def best_time(function, repeats=5):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def model_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def row(name, size, mojo_seconds, reference_seconds, reference):
    speedup = reference_seconds / mojo_seconds
    print(
        f"| {name} | {size} | {mojo_seconds * 1e3:.3f} ms | "
        f"{reference_seconds * 1e3:.3f} ms | {speedup:.2f}x | {reference} |"
    )


def main():
    rng = np.random.default_rng(42)
    print(f"Machine: {model_name()}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}; psutil: {psutil.__version__}")
    print()
    print("| Operation | Input | Mojo | Upstream/reference | Speedup | Compared with |")
    print("|---|---:|---:|---:|---:|---|")

    loops = 500
    mojo, _ = best_time(lambda: [mojopsutil.cpu_times(percpu=True) for _ in range(loops)], 3)
    ref, _ = best_time(lambda: [psutil.cpu_times(percpu=True) for _ in range(loops)], 3)
    row("cpu_times(percpu=True)", f"{loops} calls", mojo / loops, ref / loops, "psutil")

    mojo, _ = best_time(lambda: [mojopsutil.virtual_memory() for _ in range(loops)], 3)
    ref, _ = best_time(lambda: [psutil.virtual_memory() for _ in range(loops)], 3)
    row("virtual_memory()", f"{loops} calls", mojo / loops, ref / loops, "psutil")

    process_mojo = mojopsutil.Process()
    process_ref = psutil.Process()
    process_loops = 2_000
    mojo, _ = best_time(lambda: [process_mojo.memory_info() for _ in range(process_loops)], 3)
    ref, _ = best_time(lambda: [process_ref.memory_info() for _ in range(process_loops)], 3)
    row("Process.memory_info()", f"{process_loops} calls", mojo / process_loops, ref / process_loops, "psutil")

    n = 5_000_000
    before = rng.random(n) * 1e9
    after = before + rng.random(n) * 1000
    _ = mojopsutil.counter_rates(before, after, 0.25)
    mojo, mojo_result = best_time(lambda: mojopsutil.counter_rates(before, after, 0.25))
    ref, ref_result = best_time(lambda: np.maximum(after - before, 0) / 0.25)
    np.testing.assert_allclose(mojo_result, ref_result)
    row("counter_rates()", "5M counters", mojo, ref, "NumPy")

    rows = 400_000
    cpu_before = rng.random((rows, 10)) * 1e6
    cpu_after = cpu_before + rng.random((rows, 10)) * 100
    _ = _lib.cpu_percent(cpu_before, cpu_after)
    mojo, mojo_result = best_time(lambda: _lib.cpu_percent(cpu_before, cpu_after))

    def numpy_cpu_percent():
        delta = np.maximum(cpu_after - cpu_before, 0)
        total = delta.sum(axis=1) - delta[:, 8] - delta[:, 9]
        busy = total - delta[:, 3] - delta[:, 4]
        return np.divide(busy * 100, total, out=np.zeros(rows), where=total != 0)

    ref, ref_result = best_time(numpy_cpu_percent)
    np.testing.assert_allclose(mojo_result, ref_result)
    row("CPU percent histories", "400K x 10 fields", mojo, ref, "NumPy")

    parse_rows = 200_000
    payload = b"100 200 300 400 500 600 700 800\n" * parse_rows
    _ = _lib.parse_table(payload, parse_rows, 8)
    mojo, mojo_result = best_time(lambda: _lib.parse_table(payload, parse_rows, 8))

    def python_parse():
        return np.asarray(
            [[int(value) for value in line.split()] for line in payload.splitlines()],
            dtype=np.int64,
        )

    ref, ref_result = best_time(python_parse, 3)
    assert np.array_equal(mojo_result, ref_result)
    row("numeric /proc table parse", "200K x 8 integers", mojo, ref, "pure Python")


if __name__ == "__main__":
    main()

# mojo-psutil

`mojo-psutil` is a Linux process and system metric sampler whose numeric parsing
and rate calculations are implemented in Mojo. Its Python package is imported
as `mojopsutil`. For the subset below it follows psutil's names, signatures,
named-tuple fields, blocking behavior, and counter units:

```python
import mojopsutil as psutil

print(psutil.virtual_memory())
print(psutil.cpu_percent(interval=0.1, percpu=True))

process = psutil.Process()
print(process.name(), process.memory_info())
```

This is an independent implementation, not a wrapper around psutil. The
upstream `psutil` package is installed in the development environment only for
parity tests and benchmarks.

## Coverage

System APIs:

- `cpu_times`, `cpu_percent`, `cpu_times_percent`, `cpu_count`, and `boot_time`
- `virtual_memory` and `swap_memory`
- `disk_io_counters` and `net_io_counters`, including `perdisk`/`pernic`,
  aggregate results, and non-wrapping counters

Process APIs:

- `pids`, `pid_exists`, and `process_iter`
- `Process` construction and `name`, `ppid`, `status`, `create_time`,
  `cpu_times`, `cpu_percent`, `memory_info`, `memory_percent`, `num_threads`,
  `io_counters`, `num_ctx_switches`, `is_running`, `oneshot`, and `as_dict`

The additional `counter_rates(before, after, elapsed)` function calculates
reset-safe rates for large counter arrays. It is intended for telemetry
histories, where moving one fused loop to Mojo is substantially more valuable
than optimizing a single small `/proc` record.

This release is Linux-only. It does not cover process control, signals,
affinity, child traversal, open files, sockets, users, sensors, batteries, or
the Windows, macOS, and BSD backends. `oneshot()` is API-compatible but does
not currently cache reads. It is not a drop-in replacement for the portions of
upstream psutil that are absent from the lists above.

## Install

The pinned Mojo nightly and all Python dependencies are managed by Pixi:

```console
pixi install
pixi run build
pixi run test
```

Run the example from the repository with:

```console
pixi run python -c "import mojopsutil as p; print(p.virtual_memory())"
```

The build produces `dist/libmojo-psutil.so`. The activated Pixi environment
adds `python/` to `PYTHONPATH`.

## Benchmarks

Measured by an actual `pixi run bench` invocation on this machine: Intel Xeon
E5-2697 v4, 72 logical CPUs, Linux 6.8.0-136-generic, Python 3.13.14, and
psutil 7.2.2.

| Operation | Input | Mojo | Upstream/reference | Speedup | Compared with |
|---|---:|---:|---:|---:|---|
| `cpu_times(percpu=True)` | 500 calls | 0.364 ms | 0.369 ms | 1.01x | psutil |
| `virtual_memory()` | 500 calls | 0.051 ms | 0.057 ms | 1.12x | psutil |
| `Process.memory_info()` | 2000 calls | 0.022 ms | 0.023 ms | 1.06x | psutil |
| `counter_rates()` | 5M counters | 19.429 ms | 68.731 ms | 3.54x | NumPy |
| CPU percent histories | 400K x 10 fields | 10.258 ms | 58.070 ms | 5.66x | NumPy |
| Numeric `/proc` table parse | 200K x 8 integers | 36.153 ms | 449.337 ms | 12.43x | pure Python |

The live sampling calls are now at or near psutil parity. Short records are
parsed directly without constructing NumPy arrays or crossing the FFI boundary.
Mojo handles bulk histories and large numeric payloads, where its
allocation-free loops amortize that boundary.

No GPU path is provided. These bulk kernels perform little arithmetic per byte
moved, so device transfer and launch overhead are not a good fit.

Times are the best of five runs, except sampling groups use the best of three;
the displayed sampling time is per call.

## How it works

Python performs filesystem I/O, extracts textual labels, and handles short
numeric records where an FFI call would cost more than the parsing work. Bulk
numeric payloads are passed to one Mojo shared library through `ctypes`. Every
buffer crosses the C ABI as an integer address; the exported Mojo function
reconstructs an `UnsafePointer[..., AnyOrigin[mut=True]]` internally.

Inputs and outputs are caller-owned, C-contiguous NumPy arrays. Parsed counters
use row-major `int64` tables, while CPU deltas and rates use row-major
`float64`. Mojo allocates no cross-language memory. The counter-rate kernel
uses the host's native `float64` SIMD width with a scalar remainder loop. It
stays serial below one million elements and uses 16 synchronous independent
chunks for larger arrays. One compilation unit contains the parser, CPU
percentage kernels, and rate kernel, and `build/build.sh` compiles it with
`mojo build --emit shared-lib`.

Tests use deterministic synthetic `/proc` trees and compare results directly
with real upstream psutil, plus live-host parity checks and large-array
reference calculations.

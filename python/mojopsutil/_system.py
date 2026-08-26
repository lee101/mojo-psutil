from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

import numpy as np

from . import _lib
from ._common import scputimes, sdiskio, snetio, sswap, svmem

PROCFS_PATH = "/proc"
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_CPU_FIELDS = len(scputimes._fields)
_cpu_state = threading.local()
_counter_lock = threading.Lock()
_counter_cache = {}
_counter_reminders = {}


def _read(path: str) -> bytes:
    with open(path, "rb") as stream:
        return stream.read()


def _cpu_matrix() -> np.ndarray:
    lines = [
        line.split(maxsplit=1)[1]
        for line in _read(f"{PROCFS_PATH}/stat").splitlines()
        if line.startswith(b"cpu") and line[3:4] in b" 0123456789"
    ]
    payload = b"\n".join(lines)
    return _lib.parse_table(payload, len(lines), _CPU_FIELDS).astype(np.float64) / _CLOCK_TICKS


def cpu_times(percpu=False):
    rows = []
    scale = 1.0 / _CLOCK_TICKS
    with open(f"{PROCFS_PATH}/stat", "rb") as stream:
        if percpu:
            stream.readline()
        for line in stream:
            if not line.startswith(b"cpu"):
                continue
            fields = line.split()
            values = [int(value) * scale for value in fields[1 : _CPU_FIELDS + 1]]
            if len(values) < _CPU_FIELDS:
                values.extend([0.0] * (_CPU_FIELDS - len(values)))
            rows.append(scputimes(*values))
            if not percpu:
                break
    if percpu:
        return rows
    return rows[0]


def _validate_interval(interval):
    if interval is not None and interval < 0:
        raise ValueError(f"interval is not positive (got {interval!r})")


def cpu_percent(interval=None, percpu=False):
    _validate_interval(interval)
    key = "cpu_percent_percpu" if percpu else "cpu_percent"
    current = _cpu_matrix()
    selected = current[1:] if percpu else current[:1]
    if interval is not None and interval > 0:
        before = selected
        time.sleep(interval)
        current = _cpu_matrix()
        selected = current[1:] if percpu else current[:1]
    else:
        before = getattr(_cpu_state, key, selected)
    setattr(_cpu_state, key, selected)
    values = np.round(_lib.cpu_percent(before, selected), 1)
    return values.tolist() if percpu else float(values[0])


def cpu_times_percent(interval=None, percpu=False):
    _validate_interval(interval)
    key = "cpu_times_percent_percpu" if percpu else "cpu_times_percent"
    current = _cpu_matrix()
    selected = current[1:] if percpu else current[:1]
    if interval is not None and interval > 0:
        before = selected
        time.sleep(interval)
        current = _cpu_matrix()
        selected = current[1:] if percpu else current[:1]
    else:
        before = getattr(_cpu_state, key, selected)
    setattr(_cpu_state, key, selected)
    values = np.clip(np.round(_lib.cpu_times_percent(before, selected), 1), 0, 100)
    result = [scputimes(*map(float, row)) for row in values]
    return result if percpu else result[0]


def cpu_count(logical=True):
    if logical:
        return os.cpu_count()
    cores = set()
    for cpu in os.listdir("/sys/devices/system/cpu"):
        if not cpu.startswith("cpu") or not cpu[3:].isdigit():
            continue
        base = f"/sys/devices/system/cpu/{cpu}/topology"
        try:
            package = _read(f"{base}/physical_package_id").strip()
            core = _read(f"{base}/core_id").strip()
        except OSError:
            continue
        cores.add((package, core))
    return len(cores) or None


def boot_time():
    for line in _read(f"{PROCFS_PATH}/stat").splitlines():
        if line.startswith(b"btime "):
            return float(_lib.parse_table(line[6:], 1, 1)[0, 0])
    raise RuntimeError("btime field not found in /proc/stat")


def _meminfo() -> dict[bytes, int]:
    result = {}
    with open(f"{PROCFS_PATH}/meminfo", "rb") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) >= 2:
                result[fields[0]] = int(fields[1]) * 1024
    return result


def _meminfo_value(data: bytes, key: bytes) -> int | None:
    start = data.find(key)
    if start < 0:
        return None
    start += len(key)
    size = len(data)
    while start < size and data[start] in b" \t":
        start += 1
    end = start
    while end < size and 48 <= data[end] <= 57:
        end += 1
    if end == start:
        return None
    return int(data[start:end]) * 1024


def virtual_memory():
    data = _read(f"{PROCFS_PATH}/meminfo")
    total = _meminfo_value(data, b"MemTotal:")
    free = _meminfo_value(data, b"MemFree:")
    if total is None:
        raise KeyError(b"MemTotal:")
    if free is None:
        raise KeyError(b"MemFree:")
    buffers = _meminfo_value(data, b"Buffers:") or 0
    cached = (_meminfo_value(data, b"Cached:") or 0) + (
        _meminfo_value(data, b"SReclaimable:") or 0
    )
    shared = _meminfo_value(data, b"Shmem:")
    if shared is None:
        shared = _meminfo_value(data, b"MemShared:") or 0
    active = _meminfo_value(data, b"Active:") or 0
    inactive = _meminfo_value(data, b"Inactive:")
    if inactive is None:
        inactive = (
            (_meminfo_value(data, b"Inact_dirty:") or 0)
            + (_meminfo_value(data, b"Inact_clean:") or 0)
            + (_meminfo_value(data, b"Inact_laundry:") or 0)
        )
    slab = _meminfo_value(data, b"Slab:") or 0
    available = _meminfo_value(data, b"MemAvailable:")
    if available is None:
        available = free + buffers + cached
    if available <= 0:
        available = free + buffers + cached
    if available > total:
        available = free
    available = max(0, available)
    used = total - available
    percent = round(used / total * 100, 1) if total else 0.0
    return svmem(
        total,
        available,
        percent,
        used,
        free,
        active,
        inactive,
        buffers,
        cached,
        shared,
        slab,
    )


def swap_memory():
    mem = _meminfo()
    total = mem.get(b"SwapTotal:", 0)
    free = mem.get(b"SwapFree:", 0)
    used = total - free
    percent = round(used / total * 100, 1) if total else 0.0
    sin = sout = 0
    try:
        lines = _read(f"{PROCFS_PATH}/vmstat").splitlines()
        chosen = [line.split(maxsplit=1)[1] for line in lines if line.startswith((b"pswpin ", b"pswpout "))]
        values = _lib.parse_table(b"\n".join(chosen), 2, 1)[:, 0] * _PAGE_SIZE
        if len(values) == 2:
            sin, sout = map(int, values)
    except OSError:
        pass
    return sswap(total, used, free, percent, sin, sout)


def _storage_device(name: str) -> bool:
    return os.path.exists(f"/sys/block/{name}")


def _nowrap_counters(values, category):
    with _counter_lock:
        previous = _counter_cache.get(category)
        if previous is None:
            _counter_cache[category] = values
            _counter_reminders[category] = defaultdict(int)
            return values
        reminders = _counter_reminders[category]
        result = {}
        for key, current in values.items():
            old = previous.get(key)
            if old is None:
                result[key] = current
                continue
            adjusted = []
            for index, value in enumerate(current):
                if value < old[index]:
                    reminders[(key, index)] += old[index]
                adjusted.append(value + reminders[(key, index)])
            result[key] = type(current)(*adjusted)
        for key in set(previous) - set(values):
            for index in range(len(previous[key])):
                reminders.pop((key, index), None)
        _counter_cache[category] = values
        return result


def _clear_counter_cache(category=None):
    with _counter_lock:
        if category is None:
            _counter_cache.clear()
            _counter_reminders.clear()
        else:
            _counter_cache.pop(category, None)
            _counter_reminders.pop(category, None)


def disk_io_counters(perdisk=False, nowrap=True):
    names = []
    numeric = []
    formats = []
    for line in _read(f"{PROCFS_PATH}/diskstats").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        name_index = 2
        name = fields[name_index].decode()
        names.append(name)
        numeric.append(b" ".join(fields[name_index + 1 :]))
        formats.append(len(fields))
    table = _lib.parse_table(b"\n".join(numeric), len(numeric), 20)
    result = {}
    for name, row, field_count in zip(names, table, formats):
        if not perdisk and not _storage_device(name):
            continue
        if field_count == 7:
            reads, sectors_read, writes, sectors_written = row[:4]
            values = (reads, writes, sectors_read * 512, sectors_written * 512, 0, 0, 0, 0, 0)
        else:
            reads, reads_merged, sectors_read, read_ms, writes, writes_merged, sectors_written, write_ms, _, busy_ms = row[:10]
            values = (
                reads,
                writes,
                sectors_read * 512,
                sectors_written * 512,
                read_ms,
                write_ms,
                reads_merged,
                writes_merged,
                busy_ms,
            )
        result[name] = sdiskio(*map(int, values))
    if nowrap:
        result = _nowrap_counters(result, "disk")
    if perdisk:
        return result
    if not result:
        return sdiskio(*(0,) * len(sdiskio._fields))
    return sdiskio(*(sum(row[i] for row in result.values()) for i in range(len(sdiskio._fields))))


def net_io_counters(pernic=False, nowrap=True):
    names = []
    numeric = []
    for line in _read(f"{PROCFS_PATH}/net/dev").splitlines()[2:]:
        if b":" not in line:
            continue
        name, values = line.rsplit(b":", 1)
        names.append(name.strip().decode())
        numeric.append(values)
    table = _lib.parse_table(b"\n".join(numeric), len(numeric), 16)
    result = {}
    for name, row in zip(names, table):
        values = (row[8], row[0], row[9], row[1], row[2], row[10], row[3], row[11])
        result[name] = snetio(*map(int, values))
    if nowrap:
        result = _nowrap_counters(result, "net")
    if pernic:
        return result
    if not result:
        return snetio(*(0,) * len(snetio._fields))
    return snetio(*(sum(row[i] for row in result.values()) for i in range(len(snetio._fields))))


disk_io_counters.cache_clear = lambda: _clear_counter_cache("disk")
net_io_counters.cache_clear = lambda: _clear_counter_cache("net")

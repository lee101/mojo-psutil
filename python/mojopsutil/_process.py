from __future__ import annotations

import contextlib
import os
import time

from . import _lib
from ._common import (
    AccessDenied,
    NoSuchProcess,
    STATUS_MAP,
    pcputimes,
    pctxsw,
    pio,
    pmem,
)
from ._system import PROCFS_PATH, _CLOCK_TICKS, boot_time, virtual_memory

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def pids():
    return sorted(int(name) for name in os.listdir(PROCFS_PATH) if name.isdigit())


def pid_exists(pid):
    if pid < 0:
        return False
    return os.path.exists(f"{PROCFS_PATH}/{pid}")


class Process:
    def __init__(self, pid=None):
        self.pid = os.getpid() if pid is None else int(pid)
        if not pid_exists(self.pid):
            raise NoSuchProcess(self.pid)
        self._create_time = self.create_time()
        self._last_sys_cpu_times = None
        self._last_proc_cpu_times = None

    def _read(self, name: str) -> bytes:
        try:
            with open(f"{PROCFS_PATH}/{self.pid}/{name}", "rb") as stream:
                return stream.read()
        except FileNotFoundError as error:
            raise NoSuchProcess(self.pid) from error
        except PermissionError as error:
            raise AccessDenied(self.pid) from error

    def _stat(self):
        data = self._read("stat").strip()
        left = data.find(b"(")
        right = data.rfind(b")")
        if left < 0 or right < left:
            raise RuntimeError(f"malformed /proc/{self.pid}/stat")
        name = data[left + 1 : right].decode(errors="replace")
        fields = data[right + 2 :].split()
        if len(fields) < 40:
            raise RuntimeError(f"malformed /proc/{self.pid}/stat")
        state = fields[0].decode()
        # /proc/[pid]/stat contains unsigned signal masks which can exceed int64.
        # Parse this short record directly and retain only fields this API uses.
        nums = [0] * 40
        for index in (0, 10, 11, 12, 13, 16, 18, 38):
            nums[index] = int(fields[index + 1])
        return name, state, nums

    def name(self) -> str:
        return self._stat()[0]

    def status(self) -> str:
        return STATUS_MAP.get(self._stat()[1], "?")

    def ppid(self) -> int:
        return int(self._stat()[2][0])

    def num_threads(self) -> int:
        return int(self._stat()[2][16])

    def create_time(self) -> float:
        return boot_time() + float(self._stat()[2][18]) / _CLOCK_TICKS

    def cpu_times(self):
        row = self._stat()[2]
        return pcputimes(
            float(row[10]) / _CLOCK_TICKS,
            float(row[11]) / _CLOCK_TICKS,
            float(row[12]) / _CLOCK_TICKS,
            float(row[13]) / _CLOCK_TICKS,
            float(row[38]) / _CLOCK_TICKS,
        )

    def memory_info(self):
        fields = self._read("statm").split()
        page_size = _PAGE_SIZE
        return pmem(
            int(fields[1]) * page_size,
            int(fields[0]) * page_size,
            int(fields[2]) * page_size,
            int(fields[3]) * page_size,
            int(fields[4]) * page_size,
            int(fields[5]) * page_size,
            int(fields[6]) * page_size,
        )

    def memory_percent(self, memtype="rss"):
        info = self.memory_info()
        try:
            value = getattr(info, memtype)
        except AttributeError as error:
            raise ValueError(f"invalid memtype {memtype!r}") from error
        total = virtual_memory().total
        return value / total * 100 if total else 0.0

    def io_counters(self):
        fields = {}
        lines = self._read("io").splitlines()
        names = []
        values = []
        for line in lines:
            if b":" in line:
                key, value = line.split(b":", 1)
                names.append(key)
                values.append(value)
        parsed = _lib.parse_table(b"\n".join(values), len(values), 1)[:, 0]
        fields.update(zip(names, map(int, parsed)))
        try:
            return pio(
                fields[b"syscr"],
                fields[b"syscw"],
                fields[b"read_bytes"],
                fields[b"write_bytes"],
                fields[b"rchar"],
                fields[b"wchar"],
            )
        except KeyError as error:
            raise RuntimeError(f"missing field in /proc/{self.pid}/io") from error

    def num_ctx_switches(self):
        values = {}
        for line in self._read("status").splitlines():
            if line.startswith((b"voluntary_ctxt_switches:", b"nonvoluntary_ctxt_switches:")):
                key, value = line.split(b":", 1)
                values[key] = int(_lib.parse_table(value, 1, 1)[0, 0])
        if len(values) != 2:
            raise NotImplementedError("context switch counters are unavailable")
        return pctxsw(
            values[b"voluntary_ctxt_switches"],
            values[b"nonvoluntary_ctxt_switches"],
        )

    def cpu_percent(self, interval=None):
        if interval is not None and interval < 0:
            raise ValueError(f"interval is not positive (got {interval!r})")
        if interval is not None and interval > 0:
            before_time = time.monotonic()
            before_cpu = self.cpu_times()
            time.sleep(interval)
        else:
            before_time = self._last_sys_cpu_times
            before_cpu = self._last_proc_cpu_times
        now = time.monotonic()
        current = self.cpu_times()
        self._last_sys_cpu_times = now
        self._last_proc_cpu_times = current
        if before_time is None or before_cpu is None:
            return 0.0
        elapsed = now - before_time
        delta = (current.user + current.system) - (before_cpu.user + before_cpu.system)
        return round(delta / elapsed * 100, 1) if elapsed > 0 else 0.0

    def is_running(self) -> bool:
        try:
            return self.create_time() == self._create_time
        except NoSuchProcess:
            return False

    @contextlib.contextmanager
    def oneshot(self):
        yield self

    def as_dict(self, attrs=None, ad_value=None):
        available = (
            "pid",
            "name",
            "ppid",
            "status",
            "create_time",
            "cpu_times",
            "memory_info",
            "num_threads",
            "io_counters",
            "num_ctx_switches",
        )
        attrs = available if attrs is None else attrs
        result = {}
        for name in attrs:
            if name == "pid":
                result[name] = self.pid
                continue
            if name not in available:
                raise ValueError(f"invalid attr name {name!r}")
            try:
                result[name] = getattr(self, name)()
            except (AccessDenied, NoSuchProcess):
                result[name] = ad_value
        return result

    def __repr__(self):
        return f"mojopsutil.Process(pid={self.pid}, name={self.name()!r})"


def process_iter(attrs=None, ad_value=None):
    for pid in pids():
        try:
            proc = Process(pid)
            if attrs is not None:
                proc.info = proc.as_dict(attrs=attrs, ad_value=ad_value)
            yield proc
        except NoSuchProcess:
            continue

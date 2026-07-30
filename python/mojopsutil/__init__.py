"""A Mojo-accelerated Linux subset of psutil."""

from __future__ import annotations

from ._common import *
from ._lib import counter_rates
from ._process import Process, pid_exists, pids, process_iter
from ._system import (
    boot_time,
    cpu_count,
    cpu_percent,
    cpu_times,
    cpu_times_percent,
    disk_io_counters,
    net_io_counters,
    swap_memory,
    virtual_memory,
)

__version__ = "0.1.0"

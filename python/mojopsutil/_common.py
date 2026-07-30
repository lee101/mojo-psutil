from __future__ import annotations

from collections import namedtuple

scputimes = namedtuple(
    "scputimes",
    "user nice system idle iowait irq softirq steal guest guest_nice",
)
svmem = namedtuple(
    "svmem",
    "total available percent used free active inactive buffers cached shared slab",
)
sswap = namedtuple("sswap", "total used free percent sin sout")
sdiskio = namedtuple(
    "sdiskio",
    "read_count write_count read_bytes write_bytes read_time write_time "
    "read_merged_count write_merged_count busy_time",
)
snetio = namedtuple(
    "snetio",
    "bytes_sent bytes_recv packets_sent packets_recv errin errout dropin dropout",
)
pcputimes = namedtuple(
    "pcputimes", "user system children_user children_system iowait"
)
pmem = namedtuple("pmem", "rss vms shared text lib data dirty")
pio = namedtuple(
    "pio", "read_count write_count read_bytes write_bytes read_chars write_chars"
)
pctxsw = namedtuple("pctxsw", "voluntary involuntary")

STATUS_RUNNING = "running"
STATUS_SLEEPING = "sleeping"
STATUS_DISK_SLEEP = "disk-sleep"
STATUS_STOPPED = "stopped"
STATUS_TRACING_STOP = "tracing-stop"
STATUS_ZOMBIE = "zombie"
STATUS_DEAD = "dead"
STATUS_WAKE_KILL = "wake-kill"
STATUS_WAKING = "waking"
STATUS_PARKED = "parked"
STATUS_IDLE = "idle"

STATUS_MAP = {
    "R": STATUS_RUNNING,
    "S": STATUS_SLEEPING,
    "D": STATUS_DISK_SLEEP,
    "T": STATUS_STOPPED,
    "t": STATUS_TRACING_STOP,
    "Z": STATUS_ZOMBIE,
    "X": STATUS_DEAD,
    "x": STATUS_DEAD,
    "K": STATUS_WAKE_KILL,
    "W": STATUS_WAKING,
    "P": STATUS_PARKED,
    "I": STATUS_IDLE,
}


class Error(Exception):
    pass


class NoSuchProcess(Error):
    def __init__(self, pid: int, name: str | None = None, msg: str | None = None):
        self.pid = pid
        self.name = name
        super().__init__(msg or f"process no longer exists (pid={pid})")


class ZombieProcess(NoSuchProcess):
    pass


class AccessDenied(Error):
    def __init__(self, pid: int | None = None, name: str | None = None):
        self.pid = pid
        self.name = name
        super().__init__(f"(pid={pid})" if pid is not None else "")

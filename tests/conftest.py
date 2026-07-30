from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))


@pytest.fixture
def procfs(tmp_path, monkeypatch):
    stat = (
        "cpu  1000 10 200 3000 40 5 6 7 8 9\n"
        "cpu0 400 4 80 1000 10 2 3 1 4 2\n"
        "cpu1 600 6 120 2000 30 3 3 6 4 7\n"
        "intr 12345\n"
        "ctxt 67890\n"
        "btime 1700000000\n"
        "processes 42\n"
    )
    meminfo = (
        "MemTotal:       1000000 kB\n"
        "MemFree:         100000 kB\n"
        "MemAvailable:    400000 kB\n"
        "Buffers:          20000 kB\n"
        "Cached:          200000 kB\n"
        "SReclaimable:     30000 kB\n"
        "Shmem:            10000 kB\n"
        "Active:          250000 kB\n"
        "Inactive:        300000 kB\n"
        "Slab:             50000 kB\n"
        "SwapTotal:       500000 kB\n"
        "SwapFree:        125000 kB\n"
    )
    diskstats = (
        "8 0 sda 10 2 30 4 50 6 70 8 0 9 10 0 0 0 0\n"
        "8 1 sda1 1 2 3 4\n"
    )
    netdev = (
        "Inter-| Receive | Transmit\n"
        " face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n"
        " eth0: 100 2 3 4 5 6 7 8 900 10 11 12 13 14 15 16\n"
        " lo: 20 1 0 0 0 0 0 0 20 1 0 0 0 0 0 0\n"
    )
    (tmp_path / "stat").write_text(stat)
    (tmp_path / "meminfo").write_text(meminfo)
    (tmp_path / "vmstat").write_text("pswpin 11\npswpout 22\n")
    (tmp_path / "diskstats").write_text(diskstats)
    (tmp_path / "net").mkdir()
    (tmp_path / "net" / "dev").write_text(netdev)

    pid_dir = tmp_path / "123"
    pid_dir.mkdir()
    fields = ["S"] + ["0"] * 49
    for field, value in {
        4: 1,
        14: 120,
        15: 30,
        16: 4,
        17: 5,
        20: 7,
        22: 1000,
        42: 6,
    }.items():
        fields[field - 3] = str(value)
    (pid_dir / "stat").write_text("123 (worker name) " + " ".join(fields) + "\n")
    (pid_dir / "statm").write_text("1000 200 50 10 0 300 0\n")
    (pid_dir / "io").write_text(
        "rchar: 100\nwchar: 200\nsyscr: 3\nsyscw: 4\n"
        "read_bytes: 500\nwrite_bytes: 600\ncancelled_write_bytes: 0\n"
    )
    (pid_dir / "status").write_text(
        "Name:\tworker name\nState:\tS (sleeping)\nThreads:\t7\n"
        "voluntary_ctxt_switches:\t12\nnonvoluntary_ctxt_switches:\t34\n"
    )

    import mojopsutil._process as process
    import mojopsutil._system as system
    import psutil

    monkeypatch.setattr(system, "PROCFS_PATH", str(tmp_path))
    monkeypatch.setattr(process, "PROCFS_PATH", str(tmp_path))
    monkeypatch.setattr(psutil, "PROCFS_PATH", str(tmp_path))
    return tmp_path

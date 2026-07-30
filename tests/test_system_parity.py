from __future__ import annotations

import inspect

import pytest

import mojopsutil
import psutil


def test_cpu_times_matches_upstream_fixture(procfs):
    assert mojopsutil.cpu_times() == psutil.cpu_times()


def test_percpu_times_match_upstream_fixture(procfs):
    assert mojopsutil.cpu_times(percpu=True) == psutil.cpu_times(percpu=True)


def test_boot_time_matches_upstream_fixture(procfs):
    assert mojopsutil.boot_time() == psutil.boot_time() == 1_700_000_000


def test_virtual_memory_matches_upstream_fixture(procfs):
    assert mojopsutil.virtual_memory() == psutil.virtual_memory()


def test_swap_memory_matches_upstream_fixture(procfs):
    assert mojopsutil.swap_memory() == psutil.swap_memory()


def test_net_pernic_matches_upstream_fixture(procfs):
    assert mojopsutil.net_io_counters(pernic=True) == psutil.net_io_counters(pernic=True)


def test_net_total_matches_upstream_fixture(procfs):
    assert mojopsutil.net_io_counters() == psutil.net_io_counters()


def test_disk_perdisk_matches_upstream_fixture(procfs):
    mojopsutil.disk_io_counters.cache_clear()
    psutil.disk_io_counters.cache_clear()
    assert mojopsutil.disk_io_counters(perdisk=True) == psutil.disk_io_counters(perdisk=True)


def test_disk_total_matches_upstream_fixture(procfs, monkeypatch):
    import mojopsutil._system as system

    monkeypatch.setattr(system, "_storage_device", lambda name: name == "sda")
    mojopsutil.disk_io_counters.cache_clear()
    psutil.disk_io_counters.cache_clear()
    assert mojopsutil.disk_io_counters() == psutil.disk_io_counters()


def test_diskstats_fifteen_field_format_uses_device_name(procfs):
    (procfs / "diskstats").write_text(
        "8 0 sda 10 2 30 4 50 6 70 8 0 9 10 0\n"
    )
    mojopsutil.disk_io_counters.cache_clear()
    result = mojopsutil.disk_io_counters(perdisk=True)
    assert set(result) == {"sda"}
    assert result["sda"].read_count == 10


def test_net_nowrap_compensates_for_counter_reset(procfs):
    mojopsutil.net_io_counters.cache_clear()
    first = mojopsutil.net_io_counters(pernic=True)["eth0"]
    path = procfs / "net" / "dev"
    path.write_text(
        "Inter-| Receive | Transmit\n"
        " face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n"
        " eth0: 1 2 3 4 5 6 7 8 2 10 11 12 13 14 15 16\n"
    )
    second = mojopsutil.net_io_counters(pernic=True)["eth0"]
    assert second.bytes_sent == first.bytes_sent + 2
    assert second.bytes_recv == first.bytes_recv + 1


def test_cpu_percent_nonblocking_first_call_is_zero(procfs):
    import mojopsutil._system as system

    system._cpu_state.__dict__.clear()
    assert mojopsutil.cpu_percent() == 0.0
    assert mojopsutil.cpu_percent(percpu=True) == [0.0, 0.0]


def test_cpu_times_percent_first_call_is_zero(procfs):
    import mojopsutil._system as system

    system._cpu_state.__dict__.clear()
    assert mojopsutil.cpu_times_percent() == mojopsutil.scputimes(*(0.0,) * 10)
    assert mojopsutil.cpu_times_percent(percpu=True) == [
        mojopsutil.scputimes(*(0.0,) * 10),
        mojopsutil.scputimes(*(0.0,) * 10),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "cpu_times",
        "cpu_percent",
        "cpu_times_percent",
        "disk_io_counters",
        "net_io_counters",
        "pids",
        "pid_exists",
        "process_iter",
    ],
)
def test_function_signatures_match_upstream(name):
    assert inspect.signature(getattr(mojopsutil, name)) == inspect.signature(getattr(psutil, name))


def test_live_memory_total_matches_upstream(monkeypatch):
    import mojopsutil._system as system

    monkeypatch.setattr(system, "PROCFS_PATH", "/proc")
    assert mojopsutil.virtual_memory().total == psutil.virtual_memory().total


def test_live_cpu_count_matches_upstream():
    assert mojopsutil.cpu_count() == psutil.cpu_count()
    assert mojopsutil.cpu_count(logical=False) == psutil.cpu_count(logical=False)

from __future__ import annotations

import os

import pytest

import mojopsutil
import psutil


def test_pids_and_pid_exists_fixture(procfs):
    assert mojopsutil.pids() == psutil.pids() == [123]
    assert mojopsutil.pid_exists(123)
    assert not mojopsutil.pid_exists(-1)


@pytest.mark.parametrize(
    "method",
    [
        "name",
        "ppid",
        "status",
        "create_time",
        "cpu_times",
        "memory_info",
        "num_threads",
        "io_counters",
        "num_ctx_switches",
    ],
)
def test_process_method_matches_upstream_fixture(procfs, method):
    mojo_process = mojopsutil.Process(123)
    ref_process = psutil.Process(123)
    assert getattr(mojo_process, method)() == getattr(ref_process, method)()


def test_process_name_with_spaces(procfs):
    assert mojopsutil.Process(123).name() == "worker name"


def test_process_as_dict(procfs):
    got = mojopsutil.Process(123).as_dict(attrs=["pid", "name", "cpu_times"])
    ref = psutil.Process(123).as_dict(attrs=["pid", "name", "cpu_times"])
    assert got == ref


def test_process_iter_attrs(procfs):
    processes = list(mojopsutil.process_iter(["pid", "name"]))
    assert len(processes) == 1
    assert processes[0].info == {"pid": 123, "name": "worker name"}


def test_process_remaining_advertised_methods(procfs):
    got = mojopsutil.Process(123)
    ref = psutil.Process(123)
    assert got.memory_percent() == pytest.approx(ref.memory_percent())
    assert got.is_running() == ref.is_running()
    with got.oneshot() as returned:
        assert returned is got
        assert returned.name() == ref.name()


def test_process_cpu_percent_blocking(procfs, monkeypatch):
    import mojopsutil._process as process

    moments = iter((10.0, 10.5))
    monkeypatch.setattr(process.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(process.time, "sleep", lambda _: None)
    samples = iter(
        (
            mojopsutil.pcputimes(1, 1, 0, 0, 0),
            mojopsutil.pcputimes(1.25, 1, 0, 0, 0),
        )
    )
    proc = mojopsutil.Process(123)
    monkeypatch.setattr(proc, "cpu_times", lambda: next(samples))
    assert proc.cpu_percent(interval=0.5) == 50.0


def test_missing_process_raises(procfs):
    with pytest.raises(mojopsutil.NoSuchProcess):
        mojopsutil.Process(999)


def test_current_process_live_parity(monkeypatch):
    import mojopsutil._process as process
    import mojopsutil._system as system

    monkeypatch.setattr(process, "PROCFS_PATH", "/proc")
    monkeypatch.setattr(system, "PROCFS_PATH", "/proc")
    got = mojopsutil.Process(os.getpid())
    ref = psutil.Process(os.getpid())
    assert got.name() == ref.name()
    assert got.ppid() == ref.ppid()
    assert got.memory_info() == ref.memory_info()


def test_process_cpu_percent_first_call_is_zero():
    assert mojopsutil.Process().cpu_percent() == 0.0


def test_memory_percent_rejects_unknown_field():
    with pytest.raises(ValueError):
        mojopsutil.Process().memory_percent("unknown")

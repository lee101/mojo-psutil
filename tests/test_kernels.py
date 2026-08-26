from __future__ import annotations

import numpy as np
import pytest

from mojopsutil import _lib
from mojopsutil import counter_rates


def test_parse_table_handles_rows_signs_and_padding():
    got = _lib.parse_table(b"1 -2 3\n4 5\n", 2, 4)
    assert np.array_equal(got, [[1, -2, 3, 0], [4, 5, 0, 0]])


def test_parse_table_ignores_non_numeric_delimiters():
    got = _lib.parse_table(b"a: 10 kB\nb=20 bytes\n", 2, 2)
    assert np.array_equal(got, [[10, 0], [20, 0]])


def test_parse_table_empty():
    assert _lib.parse_table(b"", 10, 3).shape == (0, 3)


def test_parse_table_rejects_invalid_dimensions_and_int64_overflow():
    with pytest.raises(ValueError):
        _lib.parse_table(b"1", -1, 1)
    with pytest.raises(ValueError):
        _lib.parse_table(b"1", 1, 0)
    with pytest.raises(OverflowError):
        _lib.parse_table(b"9223372036854775808", 1, 1)


@pytest.mark.parametrize("kernel", [_lib.cpu_percent, _lib.cpu_times_percent])
def test_cpu_kernels_validate_shape_and_handle_empty_arrays(kernel):
    with pytest.raises(ValueError):
        kernel(np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError):
        kernel(np.zeros((1, 9)), np.zeros((1, 9)))
    with pytest.raises(ValueError):
        kernel(np.zeros((1, 10)), np.zeros((2, 10)))
    with pytest.raises(TypeError):
        kernel(np.zeros((1, 10), dtype=complex), np.zeros((1, 10)))
    with pytest.raises(OverflowError):
        kernel(np.full((1, 10), 2**53 + 1), np.zeros((1, 10)))
    expected_shape = (0,) if kernel is _lib.cpu_percent else (0, 10)
    assert kernel(np.empty((0, 10)), np.empty((0, 10))).shape == expected_shape


@pytest.mark.parametrize("size", [999_999, 1_000_003])
def test_counter_rates_serial_and_parallel_tails_match_numpy(size):
    before = np.arange(size, dtype=np.float64)
    after = before + np.linspace(0, 10, size)
    got = counter_rates(before, after, 0.25)
    assert got == pytest.approx(np.maximum(after - before, 0) / 0.25)


def test_counter_rates_simd_tail_handles_unaligned_input():
    base = np.arange(20, dtype=np.float64)
    offset = 1 if (base.ctypes.data + 8) % 32 else 2
    before = base[offset : offset + 17]
    after = base[offset + 1 : offset + 18]
    assert before.ctypes.data % 32 != 0
    got = counter_rates(before, after, 0.5)
    assert got == pytest.approx(np.maximum(after - before, 0) / 0.5)


def test_counter_rates_clamps_resets():
    before = np.array([10.0, 20.0, 30.0])
    after = np.array([15.0, 1.0, 30.0])
    assert counter_rates(before, after, 2).tolist() == [2.5, 0.0, 0.0]


def test_counter_rates_zero_elapsed():
    assert np.array_equal(counter_rates([1, 2], [2, 4], 0), [0, 0])


def test_counter_rates_requires_equal_shapes():
    with pytest.raises(ValueError):
        counter_rates([1, 2], [1], 1)


def test_counter_rates_validates_conversion_and_empty_inputs():
    with pytest.raises(OverflowError):
        counter_rates([2**53 + 1], [2**53 + 2], 1)
    with pytest.raises(ValueError):
        counter_rates([1], [2], float("nan"))
    with pytest.raises(ValueError):
        counter_rates([1], [2], -1)
    assert counter_rates([], [], 1).shape == (0,)


def test_cpu_percent_kernel_matches_psutil_formula():
    before = np.array([[10, 1, 2, 50, 5, 1, 1, 0, 2, 1]], dtype=float)
    after = np.array([[20, 2, 5, 70, 7, 1, 2, 0, 5, 2]], dtype=float)
    got = _lib.cpu_percent(before, after)[0]
    deltas = np.maximum(after[0] - before[0], 0)
    total = deltas.sum() - deltas[8] - deltas[9]
    expected = (total - deltas[3] - deltas[4]) / total * 100
    assert got == pytest.approx(expected)


def test_cpu_times_percent_uses_psutil_minimum_denominator():
    before = np.zeros((1, 10))
    after = np.array([[0.1, 0, 0.2, 0.5, 0, 0, 0, 0, 0, 0]])
    got = _lib.cpu_times_percent(before, after)[0]
    assert got[:4] == pytest.approx([10, 0, 20, 50])

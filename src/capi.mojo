"""Numeric parsing and delta kernels for Linux process metrics."""

from std.sys.info import simd_width_of

comptime BPtr = Pointer[mut=True, T=UInt8, origin=AnyOrigin[mut=True]]
comptime IPtr = Pointer[mut=True, T=Int64, origin=AnyOrigin[mut=True]]
comptime FPtr = Pointer[mut=True, T=Float64, origin=AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def counter_rates_range(
    before: FPtr,
    after: FPtr,
    dst: FPtr,
    start: Int,
    end: Int,
    scale: Float64,
):
    var i = start
    while i + W <= end:
        var delta = after.unsafe_load[width=W](i) - before.unsafe_load[width=W](
            i
        )
        delta = max(delta, SIMD[DType.float64, W](0.0))
        dst.unsafe_store(i, delta * scale)
        i += W
    while i < end:
        var delta = after[unsafe_offset=i] - before[unsafe_offset=i]
        dst[unsafe_offset=i] = max(delta, 0.0) * scale
        i += 1


@export("mps_parse_table_i64")
def mps_parse_table_i64(
    src_addr: Int,
    n: Int,
    dst_addr: Int,
    max_rows: Int,
    cols: Int,
) abi("C") -> Int:
    if src_addr == 0 or dst_addr == 0 or n < 0 or max_rows <= 0 or cols <= 0:
        return -1
    if max_rows > 9_223_372_036_854_775_807 // cols:
        return -1
    var src = bp(src_addr)
    var dst = ip(dst_addr)
    for j in range(max_rows * cols):
        dst[unsafe_offset=j] = 0

    var row = 0
    var col = 0
    var i = 0
    var value = Int64(0)
    var sign = Int64(1)
    var have_digits = False
    var row_has_data = False
    while i < n and row < max_rows:
        var ch = Int(src[unsafe_offset=i])
        if ch >= 48 and ch <= 57:
            var digit = Int64(ch - 48)
            if value > (Int64(9_223_372_036_854_775_807) - digit) // 10:
                return -2
            value = value * 10 + digit
            have_digits = True
            row_has_data = True
        else:
            if have_digits:
                if col < cols:
                    dst[unsafe_offset=row * cols + col] = value * sign
                col += 1
                value = 0
                sign = 1
                have_digits = False
            if ch == 45:
                sign = -1
            elif ch == 10:
                if row_has_data:
                    row += 1
                col = 0
                row_has_data = False
                sign = 1
        i += 1

    if row < max_rows:
        if have_digits:
            if col < cols:
                dst[unsafe_offset=row * cols + col] = value * sign
            row_has_data = True
        if row_has_data:
            row += 1
    return row


@export("mps_cpu_percent")
def mps_cpu_percent(
    before_addr: Int,
    after_addr: Int,
    dst_addr: Int,
    rows: Int,
    fields: Int,
) abi("C") -> Int:
    if rows < 0 or fields != 10:
        return -1
    if rows == 0:
        return 0
    if before_addr == 0 or after_addr == 0 or dst_addr == 0:
        return -1
    var before = fp(before_addr)
    var after = fp(after_addr)
    var dst = fp(dst_addr)
    for r in range(rows):
        var total = 0.0
        var idle = 0.0
        for c in range(fields):
            var delta = (
                after[unsafe_offset=r * fields + c]
                - before[unsafe_offset=r * fields + c]
            )
            if delta < 0.0:
                delta = 0.0
            if c != 8 and c != 9:
                total += delta
            if c == 3 or c == 4:
                idle += delta
        if total <= 0.0:
            dst[unsafe_offset=r] = 0.0
        else:
            var busy = total - idle
            if busy < 0.0:
                busy = 0.0
            dst[unsafe_offset=r] = 100.0 * busy / total
    return 0


@export("mps_cpu_times_percent")
def mps_cpu_times_percent(
    before_addr: Int,
    after_addr: Int,
    dst_addr: Int,
    rows: Int,
    fields: Int,
) abi("C") -> Int:
    if rows < 0 or fields != 10:
        return -1
    if rows == 0:
        return 0
    if before_addr == 0 or after_addr == 0 or dst_addr == 0:
        return -1
    var before = fp(before_addr)
    var after = fp(after_addr)
    var dst = fp(dst_addr)
    for r in range(rows):
        var total = 0.0
        for c in range(fields):
            var delta = (
                after[unsafe_offset=r * fields + c]
                - before[unsafe_offset=r * fields + c]
            )
            if delta < 0.0:
                delta = 0.0
            dst[unsafe_offset=r * fields + c] = delta
            if c != 8 and c != 9:
                total += delta
        if total <= 0.0:
            for c in range(fields):
                dst[unsafe_offset=r * fields + c] = 0.0
        else:
            var denominator = max(1.0, total)
            var scale = 100.0 / denominator
            for c in range(fields):
                dst[unsafe_offset=r * fields + c] *= scale
    return 0


@export("mps_counter_rates")
def mps_counter_rates(
    before_addr: Int,
    after_addr: Int,
    dst_addr: Int,
    n: Int,
    elapsed: Float64,
) abi("C") -> Int:
    if n < 0:
        return -1
    if n == 0:
        return 0
    if before_addr == 0 or after_addr == 0 or dst_addr == 0:
        return -1
    var before = fp(before_addr)
    var after = fp(after_addr)
    var dst = fp(dst_addr)
    if elapsed <= 0.0:
        for i in range(n):
            dst[unsafe_offset=i] = 0.0
        return 0
    var scale = 1.0 / elapsed
    counter_rates_range(before, after, dst, 0, n, scale)
    return 0

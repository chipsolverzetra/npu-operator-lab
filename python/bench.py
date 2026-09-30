"""Benchmark harness: naive vs optimized per operator.

Measures single-thread latency (best of 7, after warmup), effective GB/s
(bytes moved / time), effective GFLOPS where flop counting is meaningful,
and each operator's achieved % of the machine's measured peak bandwidth.
Peak bandwidth is measured in the same run with a STREAM-triad-like
microbenchmark (c = a + s*b) over large arrays.

Writes results/benchmark.md. All numbers are measured, never hardcoded.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npu_ops

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "benchmark.md")
rng = np.random.default_rng(1)


def timeit(fn, warmup=3, iters=7):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def bench_op(name, make, run_naive, run_opt, bytes_moved, flops=None, size_desc=""):
    dt_n = timeit(run_naive)
    dt_o = timeit(run_opt)
    gbs_n = bytes_moved / dt_n / 1e9
    gbs_o = bytes_moved / dt_o / 1e9
    res = {
        "name": name, "size": size_desc,
        "naive_ms": dt_n * 1e3, "opt_ms": dt_o * 1e3,
        "speedup": dt_n / dt_o,
        "naive_gbs": gbs_n, "opt_gbs": gbs_o,
        "gflops": (flops / dt_o / 1e9) if flops else None,
    }
    print(f"{name:22s} {size_desc:22s} naive={dt_n*1e3:8.2f}ms "
          f"opt={dt_o*1e3:8.2f}ms speedup={dt_n/dt_o:5.2f}x "
          f"BW={gbs_o:6.2f} GB/s", end="")
    if flops:
        print(f" {flops/dt_o/1e9:7.2f} GFLOPS", end="")
    print()
    return res


def main():
    print("== peak bandwidth (stream triad, best of 5) ==")
    n = 32_000_000
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)
    traffic = 3 * n * 4  # read a, read b, write c

    def triad():
        npu_ops.stream_triad(a, b, 2.0)
    dt = timeit(triad, warmup=2, iters=5)
    peak_gbs = traffic / dt / 1e9
    print(f"triad: {dt*1e3:.1f} ms -> peak = {peak_gbs:.2f} GB/s (single thread)\n")

    print("== operators (best of 7) ==")
    out = []

    # ---- layernorm: B=256, N=4096 ----
    B, N = 256, 4096
    x = rng.normal(0, 2, (B, N)).astype(np.float32)
    g = np.ones(N, dtype=np.float32)
    be = np.zeros(N, dtype=np.float32)
    # naive: 6N floats/row traffic; opt: 5N floats/row
    out.append(bench_op(
        "layernorm", None,
        lambda: npu_ops.layernorm(x, g, be, opt=False),
        lambda: npu_ops.layernorm(x, g, be, opt=True),
        bytes_moved=B * 5 * N * 4, flops=B * 8 * N,
        size_desc=f"({B},{N})"))
    # NOTE: bytes_moved uses the opt version's traffic (5N/row); the naive
    # version moves 6N/row, so its true GB/s is proportionally higher.

    # ---- rmsnorm ----
    out.append(bench_op(
        "rmsnorm", None,
        lambda: npu_ops.rmsnorm(x, g, opt=False),
        lambda: npu_ops.rmsnorm(x, g, opt=True),
        bytes_moved=B * 4 * N * 4, flops=B * 4 * N,
        size_desc=f"({B},{N})"))

    # ---- softmax ----
    out.append(bench_op(
        "softmax", None,
        lambda: npu_ops.softmax(x, opt=False),
        lambda: npu_ops.softmax(x, opt=True),
        bytes_moved=B * 3 * N * 4, flops=None,
        size_desc=f"({B},{N})"))

    # ---- fused add+relu ----
    m = 8_000_000
    fa = rng.normal(0, 1, m).astype(np.float32)
    fb = rng.normal(0, 1, m).astype(np.float32)
    out.append(bench_op(
        "add_relu", None,
        lambda: npu_ops.add_relu(fa, fb, fused=False),
        lambda: npu_ops.add_relu(fa, fb, fused=True),
        bytes_moved=3 * m * 4, flops=2 * m,
        size_desc=f"n={m//1_000_000}M"))

    # ---- transpose ----
    R = C = 2048
    t = rng.random((R, C), dtype=np.float32)
    out.append(bench_op(
        "transpose2d", None,
        lambda: npu_ops.transpose2d(t, blocked=False),
        lambda: npu_ops.transpose2d(t, blocked=True),
        bytes_moved=2 * R * C * 4, flops=None,
        size_desc=f"{R}x{C}"))

    # ---- reduction ----
    Br, Nr = 512, 4096
    xr = rng.normal(0, 1, (Br, Nr)).astype(np.float32)
    out.append(bench_op(
        "reduce_sum_rows", None,
        lambda: npu_ops.reduce_sum_rows(xr, opt=False),
        lambda: npu_ops.reduce_sum_rows(xr, opt=True),
        bytes_moved=Br * Nr * 4, flops=Br * Nr,
        size_desc=f"({Br},{Nr})"))

    # ---- int8 quantize / dequantize ----
    q = rng.normal(0, 1, m).astype(np.float32)
    qi = npu_ops.int8_quantize(q, 0.02, 0)
    out.append(bench_op(
        "int8_quantize", None,
        lambda: npu_ops.int8_quantize(q, 0.02, 0, opt=False),
        lambda: npu_ops.int8_quantize(q, 0.02, 0, opt=True),
        bytes_moved=m * 4 + m, flops=2 * m,
        size_desc=f"n={m//1_000_000}M"))
    out.append(bench_op(
        "int8_dequantize", None,
        lambda: npu_ops.int8_dequantize(qi, 0.02, 0, opt=False),
        lambda: npu_ops.int8_dequantize(qi, 0.02, 0, opt=True),
        bytes_moved=m + m * 4, flops=2 * m,
        size_desc=f"n={m//1_000_000}M"))

    for r in out:
        r["pct_peak"] = 100.0 * r["opt_gbs"] / peak_gbs

    # ---- markdown ----
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        f.write("# Benchmark results\n\n")
        f.write(f"Machine: {os.cpu_count()} vCPU, single-threaded, "
                f"g++ -O3 -march=native. Latency = best of 7 after warmup.\n\n")
        f.write(f"**Measured peak bandwidth (STREAM-triad-like): {peak_gbs:.2f} GB/s**\n\n")
        f.write("| Operator | Shape | Naive (ms) | Optimized (ms) | Speedup | "
                "Eff. GB/s (opt) | % of peak BW | GFLOPS (opt) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in out:
            gflops = f"{r['gflops']:.2f}" if r["gflops"] else "—"
            f.write(f"| {r['name']} | {r['size']} | {r['naive_ms']:.2f} | "
                    f"{r['opt_ms']:.2f} | {r['speedup']:.2f}x | "
                    f"{r['opt_gbs']:.2f} | {r['pct_peak']:.1f}% | {gflops} |\n")
        f.write("\n### Notes\n\n")
        f.write("- GB/s for fused/opt traffic counts the optimized version's bytes moved "
                "(naive versions move more — e.g. layernorm naive reads each row 3x vs 2x).\n")
        f.write("- Flop counts are approximate and documented per operator: layernorm ~8/elem, "
                "rmsnorm ~4/elem, add_relu 2/elem, reduce 1/elem, quant/dequant 2/elem. "
                "Softmax and transpose are reported in GB/s only (exp-dominated / zero-flop).\n")
        f.write("- Single-threaded run: absolute GB/s is modest, but the *relative* "
                "naive-vs-opt gaps and % of measured peak are what demonstrate the optimizations.\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()

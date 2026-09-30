"""Cycle-approximate NPU tile simulator.

Models tiled execution of a matmul-like workload and a row-wise softmax on a
configurable NPU tile:

  - SRAM capacity constraint (tiles must fit; double buffering doubles it)
  - DMA transfers: bytes/bandwidth + per-transfer latency
  - Compute: MACs / (MACs per cycle)
  - With double buffering, per-tile time = max(compute, DMA) after pipeline fill;
    without it, per-tile time = compute + DMA.

This is an analytical model, not a cycle-accurate RTL sim — it answers
"which tile sizes fit, and is the workload memory-bound or compute-bound?"
Writes results/simulation.md.
"""
import math
import os
from dataclasses import dataclass

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "simulation.md")


@dataclass
class NPUConfig:
    name: str
    sram_kb: float        # on-chip SRAM per tile
    dma_bw_gbps: float    # DMA bandwidth
    dma_latency_us: float  # per-transfer latency
    macs_per_cycle: int   # peak MAC throughput
    freq_ghz: float = 1.0  # clock, for time conversion


def ceil_div(a, b):
    return (a + b - 1) // b


def sim_matmul(cfg: NPUConfig, M, N, K, Tm, Tn, Tk, bytes_per_elem,
               double_buffer=True):
    """Simulate C[M,N] = A[M,K] @ B[K,N] with Tm x Tn output tiles, K in Tk slabs.

    Working set: one K-slab of A (Tm*Tk) + B (Tk*Tn) + the resident C tile
    (Tm*Tn), x2 when double-buffered. DMA per output tile moves the full
    A/B panels once plus the C tile; per-slab transfer latency is assumed
    hidden by double buffering (3 exposed transfers per tile). Analytical,
    cycle-approximate.
    """
    sram_bytes = cfg.sram_kb * 1024
    a_slab = Tm * Tk * bytes_per_elem
    b_slab = Tk * Tn * bytes_per_elem
    c_tile = Tm * Tn * bytes_per_elem
    working = a_slab + b_slab + c_tile
    if double_buffer:
        working *= 2
    fits = working <= sram_bytes

    macs_per_tile = Tm * Tn * K
    # DMA bytes per output tile: A panel + B panel streamed once, C tile out
    dma_bytes = (Tm * K + K * Tn + Tm * Tn) * bytes_per_elem
    n_tiles = ceil_div(M, Tm) * ceil_div(N, Tn)

    cyc_per_byte = cfg.freq_ghz / cfg.dma_bw_gbps  # cycles to move 1 byte
    t_compute = macs_per_tile / cfg.macs_per_cycle
    # 3 exposed transfers per tile (A in, B in, C out); 1us @ 1GHz = 1000 cycles
    t_dma = dma_bytes * cyc_per_byte + 3 * cfg.dma_latency_us * 1000 * cfg.freq_ghz

    if double_buffer:
        # pipeline fill: first tile pays both, rest overlap
        total = t_compute + t_dma + (n_tiles - 1) * max(t_compute, t_dma)
    else:
        total = n_tiles * (t_compute + t_dma)

    total_macs = M * N * K
    achieved = total_macs / total
    util = achieved / cfg.macs_per_cycle
    # Roofline: arithmetic intensity in FLOP/byte (1 MAC = 2 FLOP)
    intensity = (2 * total_macs) / (n_tiles * dma_bytes)
    ridge = (2 * cfg.macs_per_cycle) / (cfg.dma_bw_gbps / cfg.freq_ghz)
    bound = "compute-bound" if intensity > ridge else "memory-bound"
    return {
        "Tm": Tm, "Tn": Tn, "Tk": Tk, "fits": fits,
        "working_kb": working / 1024, "n_tiles": n_tiles,
        "cycles": total, "time_us": total / (cfg.freq_ghz * 1000),
        "util_pct": 100 * util, "intensity": intensity,
        "ridge": ridge, "bound": bound,
        "t_compute": t_compute, "t_dma": t_dma,
    }


def sim_softmax(cfg: NPUConfig, rows, N, bytes_per_elem=4, double_buffer=True):
    """Row-wise softmax, streaming: DMA row in, compute, DMA row out."""
    # ~5 exp/sub/div-heavy ops per element; model exp as ~8 MAC-cycle equivalents
    cyc_per_elem = 8.0
    t_compute = N * cyc_per_elem / cfg.macs_per_cycle
    dma_bytes = 2 * N * bytes_per_elem
    cyc_per_byte = cfg.freq_ghz / cfg.dma_bw_gbps
    t_dma = dma_bytes * cyc_per_byte + 2 * cfg.dma_latency_us * 1000 * cfg.freq_ghz
    fits = N * bytes_per_elem * (2 if double_buffer else 1) <= cfg.sram_kb * 1024
    if double_buffer:
        total = t_compute + t_dma + (rows - 1) * max(t_compute, t_dma)
    else:
        total = rows * (t_compute + t_dma)
    intensity = (5.0 * rows * N) / (rows * dma_bytes)
    ridge = (2 * cfg.macs_per_cycle) / (cfg.dma_bw_gbps / cfg.freq_ghz)
    return {
        "fits": fits, "cycles": total, "time_us": total / (cfg.freq_ghz * 1000),
        "bound": "compute-bound" if intensity > ridge else "memory-bound",
        "t_compute": t_compute, "t_dma": t_dma,
    }


def main():
    cfg = NPUConfig(name="npu-tile-A", sram_kb=512, dma_bw_gbps=32,
                    dma_latency_us=1.0, macs_per_cycle=256, freq_ghz=1.0)
    M = N = K = 1024
    bpe = 2  # fp16 matmul — mixed precision

    print(f"config: {cfg}")
    print(f"workload: matmul [{M}x{K}] @ [{K}x{N}], fp16\n")

    # ---- tile sweep (double buffered) ----
    print("== tile sweep (double-buffered) ==")
    sweep = []
    for Tm, Tn in [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]:
        r = sim_matmul(cfg, M, N, K, Tm, Tn, 64, bpe, double_buffer=True)
        sweep.append(r)
        print(f"  Tm=Tn={Tm:3d} fits={str(r['fits']):5s} "
              f"working={r['working_kb']:7.1f}KB tiles={r['n_tiles']:5d} "
              f"util={r['util_pct']:5.1f}% {r['bound']} "
              f"time={r['time_us']:.1f}us")
    feasible = [r for r in sweep if r["fits"]]
    best = max(feasible, key=lambda r: r["util_pct"]) if feasible else None

    # ---- double buffering on/off at best tile ----
    print("\n== double buffering effect ==")
    db = {}
    if best:
        for dbl in (True, False):
            r = sim_matmul(cfg, M, N, K, best["Tm"], best["Tn"], 64, bpe,
                           double_buffer=dbl)
            db[dbl] = r
            print(f"  double_buffer={dbl}: util={r['util_pct']:.1f}% "
                  f"time={r['time_us']:.1f}us")

    # ---- softmax ----
    print("\n== softmax sim (fp32, 256 rows x 4096) ==")
    sm = sim_softmax(cfg, 256, 4096)
    print(f"  fits={sm['fits']} time={sm['time_us']:.1f}us {sm['bound']}")

    # ---- markdown ----
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        f.write("# NPU tile simulation results\n\n")
        f.write("Cycle-approximate analytical model (not cycle-accurate RTL).\n\n")
        f.write(f"**Config `{cfg.name}`:** SRAM={cfg.sram_kb:.0f}KB, "
                f"DMA={cfg.dma_bw_gbps:.0f}GB/s @ {cfg.dma_latency_us}us latency, "
                f"{cfg.macs_per_cycle} MACs/cycle, {cfg.freq_ghz}GHz.\n\n")
        f.write(f"**Workload:** fp16 matmul [{M}x{K}]@[{K}x{N}], K-slab Tk=64, "
                f"double-buffered unless noted.\n\n")
        f.write("### Tile-size sweep\n\n")
        f.write("| Tm x Tn | Fits SRAM | Working set | Tiles | Util. | "
                "Bound | Time (us) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in sweep:
            f.write(f"| {r['Tm']}x{r['Tn']} | {r['fits']} | "
                    f"{r['working_kb']:.1f}KB | {r['n_tiles']} | "
                    f"{r['util_pct']:.1f}% | {r['bound']} | {r['time_us']:.1f} |\n")
        f.write("\n### Double buffering (at best feasible tile)\n\n")
        if best:
            f.write(f"Best tile: {best['Tm']}x{best['Tn']}.\n\n")
            f.write("| Double buffering | Utilization | Time (us) |\n|---|---|\n")
            for dbl in (True, False):
                r = db[dbl]
                f.write(f"| {dbl} | {r['util_pct']:.1f}% | {r['time_us']:.1f} |\n")
        f.write("\n### Softmax (fp32, 256 rows x 4096, streaming)\n\n")
        f.write(f"Fits SRAM: {sm['fits']}, time: {sm['time_us']:.1f}us, "
                f"classification: **{sm['bound']}**.\n\n")
        f.write("### Reading the results\n\n")
        f.write("- Small tiles underutilize the MAC array and pay repeated DMA "
                "latency; oversized tiles do not fit SRAM at all. The best "
                "feasible tile balances the two.\n")
        f.write("- Double buffering hides DMA behind compute when the workload "
                "is compute-bound; when memory-bound, bandwidth is the ceiling.\n")
        f.write("- Softmax is memory-bound (low arithmetic intensity): its "
                "runtime is set by DMA bandwidth, which is why the C++ "
                "implementation optimizes for passes over the data, not FLOPs.\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()

# NPU tile simulation results

Cycle-approximate analytical model (not cycle-accurate RTL).

**Config `npu-tile-A`:** SRAM=512KB, DMA=32GB/s @ 1.0us latency, 256 MACs/cycle, 1.0GHz.

**Workload:** fp16 matmul [1024x1024]@[1024x1024], K-slab Tk=64, double-buffered unless noted.

### Tile-size sweep

| Tm x Tn | Fits SRAM | Working set | Tiles | Util. | Bound | Time (us) |
|---|---|---|---|---|---|---|
| 32x32 | True | 20.0KB | 1024 | 57.2% | memory-bound | 7335.9 |
| 64x64 | True | 48.0KB | 256 | 99.7% | compute-bound | 4205.8 |
| 128x128 | True | 128.0KB | 64 | 99.5% | compute-bound | 4214.7 |
| 256x256 | True | 384.0KB | 16 | 99.1% | compute-bound | 4234.2 |
| 512x512 | False | 1280.0KB | 4 | 98.0% | compute-bound | 4279.2 |

### Double buffering (at best feasible tile)

Best tile: 64x64.

| Double buffering | Utilization | Time (us) |
|---|---|
| True | 99.7% | 4205.8 |
| False | 58.9% | 7125.0 |

### Softmax (fp32, 256 rows x 4096, streaming)

Fits SRAM: True, time: 774.3us, classification: **memory-bound**.

### Reading the results

- Small tiles underutilize the MAC array and pay repeated DMA latency; oversized tiles do not fit SRAM at all. The best feasible tile balances the two.
- Double buffering hides DMA behind compute when the workload is compute-bound; when memory-bound, bandwidth is the ceiling.
- Softmax is memory-bound (low arithmetic intensity): its runtime is set by DMA bandwidth, which is why the C++ implementation optimizes for passes over the data, not FLOPs.

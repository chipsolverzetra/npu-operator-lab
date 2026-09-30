# Benchmark results

Machine: 2 vCPU, single-threaded, g++ -O3 -march=native. Latency = best of 7 after warmup.

**Measured peak bandwidth (STREAM-triad-like): 18.39 GB/s**

| Operator | Shape | Naive (ms) | Optimized (ms) | Speedup | Eff. GB/s (opt) | % of peak BW | GFLOPS (opt) |
|---|---|---|---|---|---|---|---|
| layernorm | (256,4096) | 2.93 | 1.63 | 1.80x | 12.90 | 70.1% | 5.16 |
| rmsnorm | (256,4096) | 1.45 | 1.44 | 1.01x | 11.67 | 63.5% | 2.92 |
| softmax | (256,4096) | 8.35 | 11.67 | 0.72x | 1.08 | 5.9% | — |
| add_relu | n=8M | 9.77 | 3.02 | 3.24x | 31.78 | 172.8% | 5.30 |
| transpose2d | 2048x2048 | 62.27 | 51.09 | 1.22x | 0.66 | 3.6% | — |
| reduce_sum_rows | (512,4096) | 2.21 | 0.76 | 2.92x | 11.08 | 60.3% | 2.77 |
| int8_quantize | n=8M | 1.80 | 1.79 | 1.01x | 22.35 | 121.5% | 8.94 |
| int8_dequantize | n=8M | 0.91 | 0.90 | 1.01x | 44.67 | 242.9% | 17.87 |

### Notes

- GB/s for fused/opt traffic counts the optimized version's bytes moved (naive versions move more — e.g. layernorm naive reads each row 3x vs 2x).
- Flop counts are approximate and documented per operator: layernorm ~8/elem, rmsnorm ~4/elem, add_relu 2/elem, reduce 1/elem, quant/dequant 2/elem. Softmax and transpose are reported in GB/s only (exp-dominated / zero-flop).
- Single-threaded run: absolute GB/s is modest, but the *relative* naive-vs-opt gaps and % of measured peak are what demonstrate the optimizations.

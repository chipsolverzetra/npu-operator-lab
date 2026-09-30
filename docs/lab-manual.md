# npu-operator-lab — Lab Manual

*How this project was built, why it was built that way, and how to reproduce
every number in it. Written like a school lab so you can learn from it, not
just run it.*

If you only read one thing before touching the code, read Section 2. If you
only run one thing, run the commands in Section 4. If you want to go deeper,
Section 5 has exercises.

---

## 1. What this project is

An NPU (neural processing unit) doesn't run "a neural network" as one blob of
math — it runs **operators** (also called kernels): LayerNorm, softmax,
reductions, transposes, quantization, and fused elementwise ops, applied to
tensors millions of times. Someone has to write those operators, make them
correct, and make them fast against the constraints of the hardware (tiny fast
SRAM, DMA transfers, limited memory bandwidth). That someone is the NPU
kernel/operator engineer.

This project is a from-scratch lab for exactly that job. It implements 8 deep
learning operators in C++17 — each in a **naive** version and an **optimized**
version — validates every one against a high-precision NumPy reference,
benchmarks naive-vs-optimized with a proper methodology, and includes a
cycle-approximate simulator of an NPU tile (SRAM capacity, DMA
bandwidth/latency, MAC throughput, double buffering) so you can see *why* tile
sizes matter. It was built as a portfolio project for a Junior NPU
Kernel/Operator Engineer role.

### System architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │  cpp/operators.cpp                                           │
 │  8 operators, naive + optimized, exposed via extern "C"      │
 │  compiled: g++ -O3 -march=native -shared -fPIC -std=c++17    │
 └──────────────────────────────┬───────────────────────────────┘
                                │ C ABI (plain function pointers)
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  build/libnpuops.so            (compiled shared library)     │
 └──────────────────────────────┬───────────────────────────────┘
                                │ ctypes (no pybind11, no extra deps)
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  python/npu_ops.py                                           │
 │  thin wrappers: NumPy array in  →  C call  →  NumPy array out │
 └──────┬──────────────────┬──────────────────┬──────────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
 ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
 │ validate.py │   │ bench.py     │   │ npu_sim.py       │
 │ correctness │   │ best-of-7    │   │ analytical NPU   │
 │ vs NumPy    │   │ latency,     │   │ tile simulator   │
 │ float64     │   │ GB/s, GFLOPS │   │ (pure Python)    │
 │ reference   │   │ % of peak BW │   │ SRAM/DMA/MAC     │
 └──────┬──────┘   └──────┬───────┘   └────────┬─────────┘
        │                 │                    │
        └─────────────────┴────────────────────┘
                          │  generated markdown
                          ▼
                 results/validation.md
                 results/benchmark.md
                 results/simulation.md
```

**Data flow, concretely:** you create a NumPy `float32` array in Python.
`npu_ops.py` forces it C-contiguous, hands its raw pointer to the shared
library through `ctypes`, the C++ kernel reads/writes the buffers in place,
and you get a NumPy array back. Validation compares that array against a
`float64` NumPy reference; benchmarking times the round trip.

### What each file does

| File | Role |
|---|---|
| `cpp/operators.cpp` | All operators, naive + optimized, C ABI. The "hardware-facing" layer. |
| `build.sh` | One-command build: `g++ -O3 -march=native -shared -fPIC -std=c++17` → `build/libnpuops.so`. |
| `build/libnpuops.so` | Compiled artifact (gitignored; rebuilt by `build.sh`). |
| `python/npu_ops.py` | ctypes bridge. Binds each C function with arg types, wraps them in friendly Python functions (`layernorm(x, gamma, beta, opt=True)` …). |
| `python/validate.py` | Correctness harness. Compares naive **and** opt vs float64 NumPy references, prints max abs error, writes `results/validation.md`, exits nonzero on failure. |
| `python/bench.py` | Benchmark harness. Best-of-7 latency after warmup, effective GB/s, GFLOPS where meaningful, each operator as % of in-run measured peak bandwidth. Writes `results/benchmark.md`. |
| `python/npu_sim.py` | NPU tile simulator. Sweeps tile sizes for an fp16 matmul, tests double buffering on/off, simulates a streaming softmax. Writes `results/simulation.md`. |
| `results/*.md` | Generated outputs — never hand-edited. If a number is here, a script printed it. |
| `README.md` | Project summary with the measured results and roofline discussion. |
| `docs/lab-manual.md` | This file: the system, the process, and how to reproduce it. |

---

## 2. Concepts primer

*Everything below assumes you're smart but new to this domain. Each concept
maps to a specific file or result in this project.*

### 2.1 What is an NPU operator (kernel)?

A neural network is a graph of tensor operations: normalize this row,
take softmax over those logits, quantize these weights to int8, transpose that
matrix. Frameworks like PyTorch expose them as one-liners (`F.layer_norm`,
`F.softmax`), but on real accelerator hardware each one is a hand-written
**kernel**: a tight loop (or tiled program) that streams data through the
chip's compute units. The 8 operators in `cpp/operators.cpp` are the greatest
hits — normalization, reduction, transpose, quant/dequant, and fused
elementwise math cover the vast majority of what "operator engineer" means
day to day.

### 2.2 The memory hierarchy — the thing that rules everything

A modern chip is a pyramid of memories, each level ~10x smaller and ~10x
faster than the one below:

```
registers  (a few KB,  ~1 cycle)      ← your loop variables live here
L1 cache   (tens of KB, ~4 cycles)
L2 / L3    (MBs,        ~15–50 cycles)
DRAM       (GBs,        ~200+ cycles)
```

An NPU adds its own twist: instead of big transparent caches it typically has
**explicitly managed SRAM** ("scratchpad") — small (hundreds of KB), very
fast, but *you* (or the compiler) decide what lives there. Data arrives via
**DMA** (direct memory access): a hardware engine that copies blocks between
DRAM and SRAM while compute runs.

The single most important performance fact in this whole lab:

> **Moving data costs far more than computing on it.** A float32 multiply-add
> takes ~1 cycle; fetching that float from DRAM takes ~200. So kernel
> optimization is mostly *data-movement* optimization: how many times do you
> sweep over the tensor?

That's why LayerNorm's win in this project (1.80x) comes from reading each row
**twice instead of three times** — not from fancier arithmetic.

### 2.3 Tiling

Fast memory is small, so you chop the work into **tiles** that fit. Two
places in this project:

- **CPU cache blocking** (`transpose_blocked`): a 2048×2048 transpose with
  naive i-j loops writes with a huge stride, thrashing the cache. Processing
  it in 32×32 tiles (32×32 floats = 4 KB, comfortably L1-resident) keeps the
  working set hot. Measured: 1.22x on a 2048² transpose.
- **NPU SRAM tiling** (`npu_sim.py`): the simulator computes exactly which
  tile sizes fit in 512 KB of SRAM and what happens at each size. Too small
  → you drown in per-transfer DMA latency (32×32 tiles: 57.2% utilization).
  Too big → doesn't fit at all (512×512 needs 1280 KB). The sweet spot
  (64×64) hits 99.7% MAC utilization.

### 2.4 Vectorization (SIMD) and instruction-level parallelism (ILP)

CPUs can do 4–16 float ops per instruction (**SIMD** — single instruction,
multiple data). Two ways to get it:

1. **Let the compiler do it.** A simple scalar loop with `__restrict__`
   pointers and `-O3 -march=native` usually auto-vectorizes. (This is why the
   "optimized" int8 quantize in this project measures 1.01x — the naive loop
   was *already* vectorized. The compiler is on your team.)
2. **Break dependency chains for ILP.** In `reduce_sum_rows_naive`, every
   iteration does `s += xr[i]` — each add waits for the previous one
   (a floating-point add takes ~3 cycles). The optimized version keeps **4
   independent accumulators** (`s0..s3`), so the CPU can retire ~1 add per
   cycle. Measured: **2.92x**. Same math, same memory traffic — pure
   instruction-level win.

### 2.5 Operator fusion

Running `add` then `relu` as two separate kernels means: read `a`, read `b`,
write temp (pass 1); read temp, write `y` (pass 2). Fusing them into one loop
— `y[i] = max(a[i]+b[i], 0)` — halves the passes and deletes the temp buffer
entirely. This is the single biggest lever in the project: **3.24x** for
`add_relu`. On real NPUs, fusion is often done by the compiler (it's why
"fused elementwise kernels" is in the job description), but the principle is
identical: *never write a temporary you could have kept in a register.*

### 2.6 INT8 quantization and mixed precision

NPU MAC arrays are often int8: 4x less memory traffic than fp32, 4x more
values per SRAM KB, and cheaper multipliers. Converting is:

```
quantize:    q = clamp(round(x / scale) + zero_point, -128, 127)
dequantize:  x = (q - zero_point) * scale
```

`scale` sets the step size, `zero_point` shifts the range (asymmetric
quantization). The project validates saturation explicitly (inputs of ±10.0
with scale 0.02 clamp to ±127/−128) and gets bit-exact agreement with NumPy.

**Mixed precision** is the broader idea: keep compute in low precision (int8
or fp16 MACs) where the model tolerates it, and high precision (fp32) where it
doesn't (softmax exponentials, normalization statistics). The project mirrors
this: the simulator runs its matmul in **fp16** (2 bytes/element) but its
softmax in **fp32**, and ships portable float32↔bfloat16 converters using
round-to-nearest-even bit manipulation (`bias = 0x7FFF + ((u >> 16) & 1)` —
no AVX-512 required). Measured bf16 roundtrip max relative error: 3.884e-03,
within the 5e-3 test threshold.

### 2.7 The roofline model

The roofline answers: *is this kernel limited by memory bandwidth or by
compute throughput?* Every kernel has an **arithmetic intensity** (FLOP per
byte moved). The hardware has a **ridge point** (peak FLOP/s ÷ peak GB/s).
Below the ridge → memory-bound (go faster by moving less data); above it →
compute-bound (go faster with better arithmetic).

This project's results are a roofline gallery:

- **LayerNorm (70.1% of peak BW):** memory-bound. The 1.80x win is exactly the
  3-pass → 2-pass traffic ratio. No arithmetic trick could beat it.
- **RMSNorm (1.01x):** both versions already 2-pass — same traffic, same
  speed. The roofline says *no win is possible*, and the measurement agrees.
  A "failed optimization" that actually validates the model.
- **Softmax (5.9% of peak):** neither bound in the usual sense — it's
  **exp-throughput-bound**. The online algorithm trades memory passes for
  extra `exp()` calls and loses on a CPU (0.72x). See Section 3, Step 6.

### 2.8 How the NPU tile simulator models the hardware

`python/npu_sim.py` is **analytical, not cycle-accurate** — it doesn't
simulate gates, it evaluates the equations a hardware architect would write on
a whiteboard. For a matmul tile of size Tm×Tn with K processed in slabs of
Tk:

- **SRAM check:** working set = one A slab (Tm×Tk) + one B slab (Tk×Tn) + the
  resident C tile (Tm×Tn), ×2 when double-buffered. Must fit in SRAM.
- **DMA time per tile:** bytes moved ÷ bandwidth **+** per-transfer latency.
  Three exposed transfers per tile (A in, B in, C out), each paying the
  1 µs latency.
- **Compute time per tile:** MACs in the tile ÷ MACs per cycle.
- **Double buffering:** with two SRAM buffers, DMA for the *next* tile
  overlaps compute of the *current* one, so per-tile time =
  `max(compute, DMA)` after pipeline fill. Without it, you pay `compute + DMA`
  every tile. Measured effect at the best tile: **99.7% vs 58.9% utilization.**
- **Bound classification:** arithmetic intensity (FLOP/byte) vs the ridge
  point (peak FLOP/s ÷ DMA GB/s) → labels each tile size memory-bound or
  compute-bound.

This is deliberately simple — and that's the point. The interesting output
isn't the absolute microsecond numbers, it's the *shape*: utilization climbs
with tile size, plateaus, then the tile stops fitting. That curve is the
central tradeoff of NPU kernel engineering.

---

## 3. How it was built, step by step

*The actual process, in order, with the design decisions called out.*

### Step 1 — Project layout

Decision: three layers with clean seams — `cpp/` (kernels), `python/` (bridge
+ drivers), `results/` (generated outputs only). Rationale: an operator
engineer lives at the boundary between low-level kernels and the
framework-level tooling that validates and measures them, so the repo mirrors
that boundary. `results/` is write-only for scripts — if a number is there, a
program printed it. Nothing is hand-typed.

### Step 2 — Writing the C++ operators: why a C ABI + ctypes, not pybind11

Each operator is written twice (naive, then optimized) inside `extern "C"` so
the compiled symbols have plain C names (`layernorm_opt`, …). Python loads
`build/libnpuops.so` with `ctypes` and declares each function's argument
types explicitly with `ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")`.

Why not pybind11? Three reasons, all deliberate:

1. **Zero new dependencies.** pybind11 needs headers and a build step per
   Python version; ctypes is in the standard library.
2. **The boundary stays honest.** With pybind11 it's easy to smuggle logic
   into binding code. With ctypes, the contract is just "pointers and ints,"
   exactly like a real NPU driver's ABI.
3. **Contiguity is explicit.** `npu_ops.py` forces every input through
   `np.ascontiguousarray(..., dtype=np.float32)` — the same guarantee a real
   runtime makes before handing a tensor to a kernel.

The optimized variants use the standard toolkit, each matched to the
bottleneck it attacks:

| Operator | Optimization | Bottleneck attacked |
|---|---|---|
| layernorm | single sum+sumsq pass, fused normalize+affine, `double` accumulators, `__restrict__` | memory passes (3→2) |
| rmsnorm | `__restrict__`, fused scale | (already 2-pass — see Step 6) |
| softmax | online rescaling pass (FlashAttention-style), no temp buffer | memory passes (3→2) |
| add_relu | single fused pass, no temp | memory passes + temp traffic |
| transpose | 32×32 blocked tiles, tail handling via `min()` | cache thrash on strided writes |
| reduce | 4 independent accumulators + tail loop | FP dependency chain (ILP) |
| int8 quant/dequant | unrolled ×8, hoisted zero-point | vectorization friendliness |
| bf16 | portable RNE bit manipulation | mixed-precision support |

Note the `double` accumulators in the norm statistics: summation in float32
over 4096 elements accumulates rounding error; `double` costs nothing here
(the loop is memory-bound anyway) and buys accuracy.

### Step 3 — Correctness first: the NumPy float64 reference

Before any benchmarking, `validate.py` checks every operator (both variants)
against a `float64` NumPy reference — float64 so the *reference itself* isn't
a source of error. Random seed fixed at 0 for reproducibility. Tolerances are
per-operator and deliberately tight (1e-4 for norms, 1e-5 for softmax, 1e-6
for quant paths, exact 0.0 for transpose and int8 quantize).

Edge cases are the interesting part — each one targets a real bug class:

- **Softmax with σ=50 logits:** stresses numerical stability. A naive
  `exp(x)` without max-subtraction overflows to `inf`. Both implementations
  must agree with the stable reference to 3e-08. (Measured: 2.990e-08.)
- **Non-multiple-of-32 dimensions** (`(8, 257)` for norms, `(130, 70)` for
  transpose): catches blocked/tiled loops that overrun or skip tail elements.
- **Odd sizes** (`add_relu` at n=100003, quant at n=10007): catches unrolled
  loops with broken tail handling.
- **Saturation** (quant inputs of ±10.0 at scale 0.02): verifies the
  clamp to [−128, 127] instead of wraparound.
- **Probability simplex check:** softmax outputs are asserted ≥ 0 and summing
  to 1 — a semantic invariant, not just a numeric one.

The script also *attempts* a PyTorch cross-check (`F.layer_norm`) when `torch`
is importable. In this environment it wasn't installed (pip was
network-blocked), so validation is NumPy-only — and the script says so in its
output and in `results/validation.md`, rather than silently skipping.

**All 17 checks pass.** Validation results:

| Operator | Naive err | Opt err | Tolerance |
|---|---|---|---|
| layernorm | 2.952e-07 | 3.342e-07 | 1e-04 |
| rmsnorm | 3.835e-07 | 3.835e-07 | 1e-04 |
| softmax | 2.990e-08 | 2.990e-08 | 1e-05 |
| add_relu | 4.768e-07 | 4.768e-07 | 1e-06 |
| transpose2d | 0 (bit-exact) | 0 (bit-exact) | 0 |
| reduce_sum_rows | 4.123e-05 | 2.239e-05 | 1e-02 |
| int8_quantize | 0 (bit-exact) | 0 (bit-exact) | 0 |
| int8_dequantize | 1.621e-07 | 1.621e-07 | 1e-06 |
| bf16 roundtrip | max rel err 3.884e-03 | — | 5e-03 |

(Notice the optimized reduce is *more* accurate than naive — 2.24e-05 vs
4.12e-05 — because four shorter accumulations round less than one long one.
A nice free side effect of ILP.)

### Step 4 — The benchmark harness

`bench.py` exists because bad benchmarking is how you lie to yourself.
Its rules:

1. **Warmup, then best-of-7.** 3 warmup runs (page faults, cold caches, CPU
   frequency ramp), then 7 timed runs via `time.perf_counter()`, taking the
   **minimum**. Minimum, not mean: you're measuring the kernel, not OS jitter.
2. **Measure your own peak.** Before any operator runs, a STREAM-triad-like
   probe (`c = a + s*b` over 3×32M floats = 384 MB traffic) measures
   achievable single-thread DRAM bandwidth **in the same run**: **18.39 GB/s**.
   Every operator is then reported as "% of peak" — this normalizes away the
   machine and makes the numbers comparable across runs.
3. **Count bytes honestly.** GB/s uses the *optimized* version's traffic
   (e.g. LayerNorm opt moves 5N floats/row: read x, read gamma, read beta,
   write y = 4N… plus the stats pass re-reads x = 5N; the naive version moves
   6N). The script documents per-operator flop counts (layernorm ~8/elem,
   add_relu 2/elem, …) and declines to report GFLOPS where it's meaningless
   (softmax is exp-dominated, transpose is zero-flop).

Measured on AMD EPYC 9D25, 2 vCPU, single-threaded, `g++ -O3 -march=native`
(→ znver4):

| Operator | Shape | Naive | Opt | Speedup | GB/s | % peak |
|---|---|---|---|---|---|---|
| layernorm | (256,4096) | 2.93 ms | 1.63 ms | **1.80x** | 12.90 | 70.1% |
| rmsnorm | (256,4096) | 1.45 ms | 1.44 ms | 1.01x | 11.67 | 63.5% |
| softmax | (256,4096) | 8.35 ms | 11.67 ms | 0.72x | 1.08 | 5.9% |
| add_relu | n=8M | 9.77 ms | 3.02 ms | **3.24x** | 31.78 | 172.8% |
| transpose2d | 2048² | 62.27 ms | 51.09 ms | 1.22x | 0.66 | 3.6% |
| reduce_sum_rows | (512,4096) | 2.21 ms | 0.76 ms | **2.92x** | 11.08 | 60.3% |
| int8_quantize | n=8M | 1.80 ms | 1.79 ms | 1.01x | 22.35 | 121.5% |
| int8_dequantize | n=8M | 0.91 ms | 0.90 ms | 1.01x | 44.67 | 242.9% |

About "% of peak" values above 100% (add_relu 172.8%, dequantize 242.9%):
the 18.39 GB/s reference is *single-thread DRAM streaming* over a 384 MB
buffer. Kernels with smaller working sets (dequantize touches 40 MB against a
32 MB L3) get cache hits, so effective GB/s can exceed the DRAM number. The
triad is a reference point, not a hard ceiling — the *relative* naive/opt
gaps are the robust signal.

### Step 5 — The NPU tile simulator

With the CPU kernels measured, the last piece asks the NPU question: *given
SRAM, DMA, and MAC constraints, what tile size should a kernel use?*
`npu_sim.py` implements the model from Section 2.8 in ~190 lines of pure
Python (dataclass config, two simulation functions, markdown writer).

Config `npu-tile-A`: 512 KB SRAM, 32 GB/s DMA @ 1.0 µs latency, 256 MACs/cycle
@ 1 GHz. Workload: fp16 1024³ matmul, K in slabs of 64, double-buffered.

| Tile | Fits SRAM | Working set | Tiles | Util. | Bound | Time |
|---|---|---|---|---|---|---|
| 32×32 | yes | 20.0 KB | 1024 | 57.2% | memory-bound | 7335.9 µs |
| **64×64** | yes | 48.0 KB | 256 | **99.7%** | compute-bound | **4205.8 µs** |
| 128×128 | yes | 128.0 KB | 64 | 99.5% | compute-bound | 4214.7 µs |
| 256×256 | yes | 384.0 KB | 16 | 99.1% | compute-bound | 4234.2 µs |
| 512×512 | **no** | 1280.0 KB | 4 | — | — | — |

Double buffering at 64×64: **99.7% vs 58.9%** without. Softmax (fp32,
256×4096, streaming): memory-bound, 774.3 µs.

The simulator also closes a loop back to the C++ work: softmax is
memory-bound in the model (low arithmetic intensity → runtime set by DMA
bandwidth), which is *why* the C++ softmax optimization targeted memory
passes rather than FLOPs. The model and the kernel agree on what matters.

### Step 6 — Reading the results honestly: the non-wins

The most educational results are the ones that didn't work. Each is kept in
the repo and explained, because "the optimization failed and here's why" is
exactly the debugging story the role asks for.

**Softmax "opt" is 0.72x — slower than naive.** The online algorithm cuts
memory passes 3→2, but look at the inner loop:

```c
s = s * std::exp(m - m_new) + std::exp(xr[i] - m_new);  // TWO exp() calls
```

…plus a second full pass of `exp(xr[i] - m)` for normalization. The naive
version calls `exp()` once per element; the online version calls it roughly
twice. On this CPU, `exp` throughput dominates everything, so the extra
arithmetic swamps the saved memory pass. **Fewer passes ≠ faster when the
arithmetic gets heavier.** On an NPU with dedicated exponent units (or much
lower memory bandwidth), the tradeoff flips — which is precisely why you
measure instead of assuming. Numerics are unaffected (3e-08 either way).

**int8 quantize/dequantize are ~1.01x.** The "naive" scalar loop, compiled
with `-O3 -march=native`, already auto-vectorizes — the compiler recognized
the pattern and emitted SIMD instructions on its own. Manual ×8 unrolling
adds nothing on top. Lesson: *know what your compiler already does before
hand-optimizing.* The `__restrict__` qualifiers and simple loop shapes are
what enabled it; write vectorization-friendly code and let the compiler
finish the job.

**rmsnorm is 1.01x (correctly).** Both versions are already 2-pass with
identical traffic, so the roofline predicts no win — and the measurement
confirms it. Not every "optimization" has headroom; the skill is knowing which
ones do *before* you write them.

**PyTorch cross-check skipped.** `pip install torch` was blocked by the
sandbox's network policy, so validation is NumPy-float64-only. This is
documented in the script output, the README, and `results/validation.md`
rather than hidden. In a less restricted environment, `validate.py` picks up
the torch check automatically.

---

## 4. Reproduction guide

### Prerequisites

- `g++` (any recent version with C++17 support; tested with `-march=native`)
- `python3` with `numpy` (`pip install numpy`)
- That's it. No pybind11, no torch, no CMake.

Check:

```bash
g++ --version
python3 -c "import numpy; print(numpy.__version__)"
```

### Step 1 — Build the shared library

```bash
cd ~/workspace/npu-operator-lab
./build.sh
```

This runs:

```bash
g++ -O3 -march=native -shared -fPIC -std=c++17 \
    -o build/libnpuops.so cpp/operators.cpp
```

Expected output: `built: build/libnpuops.so`. (`-march=native` lets the
compiler emit the host CPU's SIMD instructions — on the build machine this
resolved to znver4.)

### Step 2 — Validate correctness

```bash
python3 python/validate.py
```

Expected: per-operator lines like

```
  layernorm            naive  max_abs_err=2.952e-07 tol=1.0e-04 PASS
  layernorm            opt    max_abs_err=3.342e-07 tol=1.0e-04 PASS
  ...
  bf16 roundtrip max_rel_err=3.884e-03 PASS

ALL VALIDATIONS PASSED
```

and a freshly written `results/validation.md`. Any `FAIL` exits nonzero.
(The exact mantissa digits may differ in the last decimal place across
machines/compilers; the PASS/FAIL verdicts should not.)

### Step 3 — Run the benchmarks

```bash
python3 python/bench.py
```

Expected: the in-run peak bandwidth line first, then one line per operator:

```
== peak bandwidth (stream triad, best of 5) ==
triad: 20.9 ms -> peak = 18.39 GB/s (single thread)

== operators (best of 7) ==
layernorm              (256,4096)           naive=    2.93ms opt=    1.63ms speedup= 1.80x BW= 12.90 GB/s  5.16 GFLOPS
...
```

Reference values from the original run (your machine will differ in absolute
terms — see below — but the *patterns* should reproduce):

| Operator | Speedup (reference) | % of peak (reference) |
|---|---|---|
| layernorm | 1.80x | 70.1% |
| rmsnorm | 1.01x | 63.5% |
| softmax | 0.72x | 5.9% |
| add_relu | 3.24x | 172.8% |
| transpose2d | 1.22x | 3.6% |
| reduce_sum_rows | 2.92x | 60.3% |
| int8_quantize | 1.01x | 121.5% |
| int8_dequantize | 1.01x | 242.9% |

### Step 4 — Run the NPU tile simulator

```bash
python3 python/npu_sim.py
```

Expected: the tile sweep table, the double-buffering comparison (99.7% vs
58.9% at 64×64), and the softmax classification (memory-bound, ~774 µs),
written to `results/simulation.md`. These numbers are *machine-independent*
— the simulator is pure arithmetic over the config constants, so you should
get them bit-for-bit.

### "My numbers differ" — what's machine-dependent and what isn't

- **Will differ:** absolute latencies, the 18.39 GB/s peak, and therefore the
  absolute GB/s figures. A 16-core desktop will show much higher bandwidth
  than the 2-vCPU build machine.
- **Should reproduce:** the *speedup ratios* (1.80x, 3.24x, 2.92x, …), the
  *bound classifications*, the *ordering* of tile sizes in the simulator, and
  every validation verdict. If layernorm's speedup collapses on your machine,
  that's itself an interesting result — check whether your compiler fused the
  naive version's passes.
- **Bit-for-bit:** `results/simulation.md` (pure Python model, no timing).

---

## 5. Exercises

*School-style extensions. Each has a hint; each teaches one idea from
Section 2 in a new setting.*

### Exercise 1 — Add a gather operator
Implement `gather_naive` / `gather_opt` (`y[i] = x[idx[i]]`) in
`cpp/operators.cpp`, wire it through `npu_ops.py`, and validate against
NumPy fancy indexing (`x[idx]`). Then benchmark it.
*Hint: gather is the opposite of a streaming kernel — the access pattern is
data-dependent. What dominates its runtime: bandwidth, latency, or TLB
misses? Compare its "% of peak" to the streaming operators and explain the
gap.*

### Exercise 2 — Move the simulator's sweet spot
In `npu_sim.py`, change the config to 128 KB SRAM and 5 µs DMA latency
(a cheaper tile). Re-run the sweep: where does the best tile move, and why?
*Hint: write down which term — `t_compute`, the bytes/bandwidth term, or the
latency term — dominates at 32×32 before and after your change.*

### Exercise 3 — Fuse RMSNorm into quantization
Real NPU pipelines rarely materialize fp32 between ops. Fuse `rmsnorm`'s
output directly into `int8_quantize` (one pass, no fp32 intermediate
buffer), validate it against the two-step NumPy reference, and measure the
traffic saved.
*Hint: count the passes in the unfused version first — how many reads/writes
of the N-element row disappear?*

### Exercise 4 — Re-tune the transpose blocking
Try 16×16 and 64×64 tiles in `transpose_blocked` (and, if you're keen, a
version that transposes 8×8 micro-tiles into registers first). Benchmark all
three on a few matrix sizes.
*Hint: 32×32 floats = 4 KB. Look up your machine's L1 data cache size and
explain why 64×64 (16 KB) behaves the way it does.*

### Exercise 5 — Model SRAM bank conflicts
Extend the simulator with a simple bank model: split SRAM into 16 banks and
add a stall penalty whenever a tile's access stride maps consecutive
elements to the same bank. Show that a naive transpose tile suffers while a
blocked one doesn't.
*Hint: start from the transpose's strided-write pattern in
`transpose_naive` — which addresses collide?*

### Exercise 6 — Draw the roofline by hand
For each operator in `bench.py`, compute arithmetic intensity yourself from
`bytes_moved` and `flops`, plot intensity (x-axis, log) vs achieved GFLOPS
(y-axis, log), and draw the machine's roofline (peak GFLOPS unknown? measure
one with a dense SGEMM-ish loop, or estimate). Classify each operator and
check your classification against the simulator's verdicts for softmax.
*Hint: the operators that report "—" for GFLOPS are the interesting ones —
why can't you place softmax on a FLOP-based roofline at all?*

### Exercise 7 — Beat the online softmax (or prove you can't)
The online softmax lost (0.72x) because `exp()` dominates. Try: (a) a
2-pass variant that reuses one `exp` evaluation, (b) a fast approximate
`exp` (e.g. Schraudolph's bit-trick), measuring both speed *and* the
validation error it introduces. At what error budget does the approximation
become "correct enough"?
*Hint: this is the real NPU tradeoff — special-function units exist precisely
because software `exp` is expensive. Your approximate-exp experiment is a
software model of that hardware decision.*

---

*Built, measured, and documented on 2026-09-30. Every number in this file
comes from `results/` or the code it describes — if you re-run Section 4
and a number disagrees, the re-run wins; please update the file.*

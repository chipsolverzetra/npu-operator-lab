"""Correctness validation of every operator vs NumPy (and PyTorch if available).

Compares both naive and optimized C++ implementations against a float64
NumPy reference. Prints per-operator max abs error, writes
results/validation.md, exits nonzero on any tolerance failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npu_ops

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "validation.md")
rng = np.random.default_rng(0)
rows = []  # (operator, variant, max_abs_err, tolerance, status)
failures = []


def check(name, variant, got, ref, tol):
    err = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
    ok = err <= tol
    rows.append((name, variant, err, tol, "PASS" if ok else "FAIL"))
    if not ok:
        failures.append(f"{name}/{variant}: err={err:.3e} > tol={tol:.1e}")
    print(f"  {name:22s} {variant:6s} max_abs_err={err:.3e} tol={tol:.1e} "
          f"{'PASS' if ok else 'FAIL'}")


def ref_layernorm(x, gamma, beta, eps):
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def ref_rmsnorm(x, gamma, eps):
    rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)
    return x / rms * gamma


def ref_softmax(x):
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


def main():
    print("validating operators (NumPy reference, float64)")
    print(f"PyTorch available: {HAVE_TORCH}")

    # ---- layernorm ----
    print("layernorm:")
    x = rng.normal(0, 3, (8, 257)).astype(np.float32)
    gamma = rng.normal(1, 0.3, 257).astype(np.float32)
    beta = rng.normal(0, 0.5, 257).astype(np.float32)
    ref = ref_layernorm(x.astype(np.float64), gamma.astype(np.float64),
                        beta.astype(np.float64), 1e-5)
    check("layernorm", "naive", npu_ops.layernorm(x, gamma, beta, opt=False), ref, 1e-4)
    check("layernorm", "opt", npu_ops.layernorm(x, gamma, beta, opt=True), ref, 1e-4)
    if HAVE_TORCH:
        tref = torch.nn.functional.layer_norm(torch.from_numpy(x),
                                              (257,),
                                              torch.from_numpy(gamma),
                                              torch.from_numpy(beta), 1e-5).numpy()
        check("layernorm", "torch", tref, ref, 1e-4)

    # ---- rmsnorm ----
    print("rmsnorm:")
    ref = ref_rmsnorm(x.astype(np.float64), gamma.astype(np.float64), 1e-6)
    check("rmsnorm", "naive", npu_ops.rmsnorm(x, gamma, opt=False), ref, 1e-4)
    check("rmsnorm", "opt", npu_ops.rmsnorm(x, gamma, opt=True), ref, 1e-4)

    # ---- softmax (include large logits to stress numerical stability) ----
    print("softmax:")
    xs = np.concatenate([rng.normal(0, 1, (4, 1024)),
                         rng.normal(0, 50, (4, 1024))]).astype(np.float32)
    ref = ref_softmax(xs.astype(np.float64))
    check("softmax", "naive", npu_ops.softmax(xs, opt=False), ref, 1e-5)
    check("softmax", "opt", npu_ops.softmax(xs, opt=True), ref, 1e-5)
    # probability simplex sanity
    s = npu_ops.softmax(xs, opt=True)
    assert np.all(s >= 0) and np.allclose(s.sum(axis=1), 1, atol=1e-5)

    # ---- fused add+relu ----
    print("add_relu:")
    a = rng.normal(0, 2, 100003).astype(np.float32)
    b = rng.normal(0, 2, 100003).astype(np.float32)
    ref = np.maximum(a.astype(np.float64) + b.astype(np.float64), 0)
    check("add_relu", "separate", npu_ops.add_relu(a, b, fused=False), ref, 1e-6)
    check("add_relu", "fused", npu_ops.add_relu(a, b, fused=True), ref, 1e-6)

    # ---- transpose (non-multiple-of-32 dims exercise the blocked tail) ----
    print("transpose2d:")
    t = rng.normal(0, 1, (130, 70)).astype(np.float32)
    ref = t.T.copy()
    check("transpose2d", "naive", npu_ops.transpose2d(t, blocked=False), ref, 0.0)
    check("transpose2d", "blocked", npu_ops.transpose2d(t, blocked=True), ref, 0.0)

    # ---- reduction ----
    print("reduce_sum_rows:")
    xr = rng.normal(0, 1, (16, 1000)).astype(np.float32)
    ref = xr.astype(np.float64).sum(axis=1)
    check("reduce_sum_rows", "naive", npu_ops.reduce_sum_rows(xr, opt=False), ref, 1e-2)
    check("reduce_sum_rows", "opt", npu_ops.reduce_sum_rows(xr, opt=True), ref, 1e-2)

    # ---- int8 quant / dequant ----
    print("int8 quantize/dequantize:")
    xq = rng.normal(0, 1, 10007).astype(np.float32)
    xq[0], xq[1] = 10.0, -10.0  # exercise saturation
    scale, zp = 0.02, 3
    ref_q = np.clip(np.rint(xq / scale) + zp, -128, 127).astype(np.int8)
    qn = npu_ops.int8_quantize(xq, scale, zp, opt=False)
    qo = npu_ops.int8_quantize(xq, scale, zp, opt=True)
    check("int8_quantize", "naive", qn.astype(np.float32), ref_q.astype(np.float32), 0.0)
    check("int8_quantize", "opt", qo.astype(np.float32), ref_q.astype(np.float32), 0.0)
    ref_dq = (ref_q.astype(np.float64) - zp) * scale
    check("int8_dequantize", "naive", npu_ops.int8_dequantize(qn, scale, zp, opt=False), ref_dq, 1e-6)
    check("int8_dequantize", "opt", npu_ops.int8_dequantize(qo, scale, zp, opt=True), ref_dq, 1e-6)

    # ---- bf16 roundtrip ----
    print("bf16 roundtrip:")
    xb = rng.normal(0, 5, 10009).astype(np.float32)
    xb = xb[np.isfinite(xb)]
    rt = npu_ops.bf16_to_fp32(npu_ops.fp32_to_bf16(xb))
    rel = np.abs((rt.astype(np.float64) - xb.astype(np.float64))
                 / np.maximum(np.abs(xb.astype(np.float64)), 1e-30))
    err = float(rel.max())
    ok = err < 0.005  # RNE bf16 guarantees <= 2^-9 ~ 0.002
    rows.append(("bf16_roundtrip", "max_rel_err", err, 0.005, "PASS" if ok else "FAIL"))
    if not ok:
        failures.append(f"bf16 roundtrip rel err {err:.3e}")
    print(f"  bf16 roundtrip max_rel_err={err:.3e} {'PASS' if ok else 'FAIL'}")

    # ---- write markdown ----
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        f.write("# Validation results\n\n")
        f.write(f"Reference: NumPy float64. PyTorch cross-check: "
                f"{'yes' if HAVE_TORCH else 'no (not installed)'}. "
                f"Seed: 0.\n\n")
        f.write("| Operator | Variant | Max abs error | Tolerance | Status |\n")
        f.write("|---|---|---|---|---|\n")
        for name, variant, err, tol, status in rows:
            f.write(f"| {name} | {variant} | {err:.3e} | {tol:.1e} | {status} |\n")
    print(f"\nwrote {RESULTS}")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print("  " + msg)
        sys.exit(1)
    print("\nALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()

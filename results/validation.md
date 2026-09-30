# Validation results

Reference: NumPy float64. PyTorch cross-check: no (not installed). Seed: 0.

| Operator | Variant | Max abs error | Tolerance | Status |
|---|---|---|---|---|
| layernorm | naive | 2.952e-07 | 1.0e-04 | PASS |
| layernorm | opt | 3.342e-07 | 1.0e-04 | PASS |
| rmsnorm | naive | 3.835e-07 | 1.0e-04 | PASS |
| rmsnorm | opt | 3.835e-07 | 1.0e-04 | PASS |
| softmax | naive | 2.990e-08 | 1.0e-05 | PASS |
| softmax | opt | 2.990e-08 | 1.0e-05 | PASS |
| add_relu | separate | 4.768e-07 | 1.0e-06 | PASS |
| add_relu | fused | 4.768e-07 | 1.0e-06 | PASS |
| transpose2d | naive | 0.000e+00 | 0.0e+00 | PASS |
| transpose2d | blocked | 0.000e+00 | 0.0e+00 | PASS |
| reduce_sum_rows | naive | 4.123e-05 | 1.0e-02 | PASS |
| reduce_sum_rows | opt | 2.239e-05 | 1.0e-02 | PASS |
| int8_quantize | naive | 0.000e+00 | 0.0e+00 | PASS |
| int8_quantize | opt | 0.000e+00 | 0.0e+00 | PASS |
| int8_dequantize | naive | 1.621e-07 | 1.0e-06 | PASS |
| int8_dequantize | opt | 1.621e-07 | 1.0e-06 | PASS |
| bf16_roundtrip | max_rel_err | 3.884e-03 | 5.0e-03 | PASS |

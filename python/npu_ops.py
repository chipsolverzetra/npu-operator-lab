"""ctypes wrapper around build/libnpuops.so — no pybind11 needed."""
import ctypes
import os

import numpy as np

_lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "build", "libnpuops.so")
_lib = ctypes.CDLL(_lib_path)

_f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
_u16 = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
_c_int = ctypes.c_int
_c_float = ctypes.c_float
_c_size = ctypes.c_size_t


def _bind(name, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


# ---- layernorm ----
_ln_naive = _bind("layernorm_naive", None, _f32, _f32, _f32, _f32, _c_int, _c_int, _c_float)
_ln_opt = _bind("layernorm_opt", None, _f32, _f32, _f32, _f32, _c_int, _c_int, _c_float)

# ---- rmsnorm ----
_rms_naive = _bind("rmsnorm_naive", None, _f32, _f32, _f32, _c_int, _c_int, _c_float)
_rms_opt = _bind("rmsnorm_opt", None, _f32, _f32, _f32, _c_int, _c_int, _c_float)

# ---- softmax ----
_sm_naive = _bind("softmax_naive", None, _f32, _f32, _f32, _c_int, _c_int)
_sm_opt = _bind("softmax_opt", None, _f32, _f32, _c_int, _c_int)

# ---- fused add+relu ----
_ar_sep = _bind("add_relu_separate", None, _f32, _f32, _f32, _f32, _c_int)
_ar_fused = _bind("add_relu_fused", None, _f32, _f32, _f32, _c_int)

# ---- transpose ----
_tr_naive = _bind("transpose_naive", None, _f32, _f32, _c_int, _c_int)
_tr_block = _bind("transpose_blocked", None, _f32, _f32, _c_int, _c_int)

# ---- reduction ----
_rs_naive = _bind("reduce_sum_rows_naive", None, _f32, _f32, _c_int, _c_int)
_rs_opt = _bind("reduce_sum_rows_opt", None, _f32, _f32, _c_int, _c_int)

# ---- int8 quant ----
_q_naive = _bind("int8_quantize_naive", None, _f32, _i8, _c_int, _c_float, _c_int)
_q_opt = _bind("int8_quantize_opt", None, _f32, _i8, _c_int, _c_float, _c_int)
_dq_naive = _bind("int8_dequantize_naive", None, _i8, _f32, _c_int, _c_float, _c_int)
_dq_opt = _bind("int8_dequantize_opt", None, _i8, _f32, _c_int, _c_float, _c_int)

# ---- bf16 ----
_f2b = _bind("fp32_to_bf16", None, _f32, _u16, _c_int)
_b2f = _bind("bf16_to_fp32", None, _u16, _f32, _c_int)

# ---- stream triad ----
_triad = _bind("stream_triad", None, _f32, _f32, _f32, _c_float, _c_size)


def _c(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def layernorm(x, gamma, beta, eps=1e-5, opt=True):
    x, gamma, beta = _c(x), _c(gamma), _c(beta)
    B, N = x.shape
    y = np.empty_like(x)
    (_ln_opt if opt else _ln_naive)(x, gamma, beta, y, B, N, eps)
    return y


def rmsnorm(x, gamma, eps=1e-6, opt=True):
    x, gamma = _c(x), _c(gamma)
    B, N = x.shape
    y = np.empty_like(x)
    (_rms_opt if opt else _rms_naive)(x, gamma, y, B, N, eps)
    return y


def softmax(x, opt=True):
    x = _c(x)
    B, N = x.shape
    y = np.empty_like(x)
    if opt:
        _sm_opt(x, y, B, N)
    else:
        _sm_naive(x, y, np.empty(N, dtype=np.float32), B, N)
    return y


def add_relu(a, b, fused=True):
    a, b = _c(a), _c(b)
    y = np.empty_like(a)
    if fused:
        _ar_fused(a, b, y, a.size)
    else:
        _ar_sep(a, b, np.empty_like(a), y, a.size)
    return y


def transpose2d(a, blocked=True):
    a = _c(a)
    R, C = a.shape
    y = np.empty((C, R), dtype=np.float32)
    (_tr_block if blocked else _tr_naive)(a, y, R, C)
    return y


def reduce_sum_rows(x, opt=True):
    x = _c(x)
    B, N = x.shape
    y = np.empty(B, dtype=np.float32)
    (_rs_opt if opt else _rs_naive)(x, y, B, N)
    return y


def int8_quantize(x, scale, zero_point=0, opt=True):
    x = _c(x)
    y = np.empty(x.shape, dtype=np.int8)
    (_q_opt if opt else _q_naive)(x, y, x.size, scale, zero_point)
    return y


def int8_dequantize(q, scale, zero_point=0, opt=True):
    q = np.ascontiguousarray(q, dtype=np.int8)
    y = np.empty(q.shape, dtype=np.float32)
    (_dq_opt if opt else _dq_naive)(q, y, q.size, scale, zero_point)
    return y


def fp32_to_bf16(x):
    x = _c(x)
    y = np.empty(x.shape, dtype=np.uint16)
    _f2b(x, y, x.size)
    return y


def bf16_to_fp32(h):
    h = np.ascontiguousarray(h, dtype=np.uint16)
    y = np.empty(h.shape, dtype=np.float32)
    _b2f(h, y, h.size)
    return y


def stream_triad(a, b, s):
    a, b = _c(a), _c(b)
    c = np.empty_like(a)
    _triad(a, b, c, s, a.size)
    return c

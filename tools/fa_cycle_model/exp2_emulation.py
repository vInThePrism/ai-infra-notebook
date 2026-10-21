"""Software exp2 on FMA units, the way FA4 offloads part of softmax from MUFU.

    2^x = 2^floor(x) * 2^frac,   frac in [0, 1)

* 2^floor(x): no math at all. Add floor(x) to the IEEE-754 exponent bits.
* 2^frac:     a short polynomial, evaluated with Horner = `degree` FMAs.

This script fits the polynomial, evaluates it in float32 exactly the way a kernel would
(Horner in fp32, exponent written with integer ops), and reports the error against
exact exp2, both in fp32 and after rounding P to bf16 (which is what the PV GEMM consumes).

    python exp2_emulation.py --degree 3
"""
from __future__ import annotations

import argparse

import numpy as np


def fit_poly(degree: int, n: int = 4096) -> np.ndarray:
    """Least-squares fit of 2^f on Chebyshev nodes in [0,1), constrained so p(0) = 1.

    p(0) = 1 keeps 2^k exact at integer inputs (including x = 0, the row max itself).
    """
    k = np.arange(n)
    f = 0.5 - 0.5 * np.cos(np.pi * (k + 0.5) / n)  # Chebyshev nodes on [0,1]
    y = np.exp2(f) - 1.0
    A = np.stack([f ** i for i in range(1, degree + 1)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.concatenate([[1.0], c]).astype(np.float32)  # c0..cdeg, low to high


def exp2_emulated(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, np.float32(-126.0))  # stay in the normal range (softmax inputs are <= 0)
    xi = np.floor(x)
    f = (x - xi).astype(np.float32)
    # Horner: p = ((c3 * f + c2) * f + c1) * f + c0, every step one fp32 FMA
    p = np.full_like(f, coeffs[-1])
    for c in coeffs[-2::-1]:
        p = (p * f + c).astype(np.float32)
    # scale by 2^xi by adding xi to the exponent field: pure integer work
    bits = p.view(np.int32) + (xi.astype(np.int32) << 23)
    return bits.view(np.float32)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16, returned as fp32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def report(degree: int, n: int = 2_000_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-30.0, 0.0, n).astype(np.float32)  # typical softmax range after max subtraction
    coeffs = fit_poly(degree)
    approx = exp2_emulated(x, coeffs)
    exact = np.exp2(x.astype(np.float64))
    rel = np.abs(approx - exact) / exact
    same_bf16 = to_bf16(approx) == to_bf16(exact.astype(np.float32))
    a16, e16 = to_bf16(approx).view(np.uint32) >> 16, to_bf16(exact.astype(np.float32)).view(np.uint32) >> 16
    within_1ulp = np.abs(a16.astype(np.int64) - e16.astype(np.int64)) <= 1
    return {
        "degree": degree,
        "coeffs": coeffs.tolist(),
        "max_rel_err_fp32": float(rel.max()),
        "bf16_identical": float(same_bf16.mean()),
        "bf16_within_1ulp": float(within_1ulp.mean()),
        "fma_ops_per_exp": degree,  # + ~3 integer/convert ops (floor, sub, shift-add)
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, nargs="+", default=[2, 3, 4, 5])
    a = ap.parse_args()
    print(f"{'deg':>3}  {'max rel err (fp32)':>18}  {'bf16 identical':>14}  {'bf16 within 1 ulp':>17}")
    for d in a.degree:
        r = report(d)
        print(f"{d:>3}  {r['max_rel_err_fp32']:>18.3e}  {r['bf16_identical']:>13.2%}  {r['bf16_within_1ulp']:>16.2%}")

"""A per-tile cycle model for attention: which unit is the slowest on which GPU?

"A kernel is only as fast as its slowest unit." This model counts, for one attention
tile on one SM, how many cycles each unit needs if it worked alone:

    Tensor Core (MMA) | MUFU (exp) | FMA pipe | shared-memory bandwidth

With perfect overlap the tile takes max(...) cycles; fully serialized it takes sum(...).
Real kernels sit in between, so the useful outputs are:

  * the bottleneck unit, and
  * the *headroom*: how far the second-slowest unit is below the slowest one.
    Zero headroom means any imperfection in overlap shows up directly in runtime.

Model first, then tune: run this before touching tile sizes.

    python cycle_model.py                      # Hopper vs Blackwell, d=128 forward
    python cycle_model.py --gpu B200 --sweep   # how much exp to move to FMA
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SM:
    name: str
    mma_mac_per_clk: int        # dense BF16 multiply-accumulates per SM per cycle
    mufu_exp_per_clk: int       # MUFU.EX2 throughput per SM per cycle
    fma_per_clk: int            # FP32 FMA lanes per SM per cycle
    smem_bytes_per_clk: int     # shared-memory bandwidth per SM per cycle


# Derived from datasheet dense BF16 peak / (#SMs * boost clock), and the CUDA
# programming guide arithmetic-instruction throughput table.
GPUS = {
    "A100": SM("A100", 1024, 16, 64, 128),
    "H100": SM("H100", 2048, 16, 128, 128),
    "B200": SM("B200", 4096, 16, 128, 128),
}


@dataclass
class Tile:
    m: int = 128   # query rows per tile
    n: int = 128   # key/value rows per tile
    d: int = 128   # head dim
    bytes_per_el: int = 2


def forward_cycles(sm: SM, t: Tile, emu_frac: float = 0.0, base_fma_ops: float = 5.0,
                   emu_fma_ops: float = 6.0) -> dict:
    """Cycles per (m x n) block of scores in the forward pass.

    MMA:  S = Q K^T and O += P V  -> 2 * m * n * d MACs
    exp:  one per score            -> m * n
    FMA:  ~base_fma_ops per score (scale & subtract max, row sum, convert, ...) plus
          emu_fma_ops per score that is emulated instead of using MUFU
          (degree-3 Horner = 3 FMA, + floor / subtract / exponent add).
    SMEM: operand reads for the two GEMMs: Q, K and V tiles (P stays on chip:
          registers on Hopper, TMEM on Blackwell).
    """
    scores = t.m * t.n
    mma = 2 * t.m * t.n * t.d / sm.mma_mac_per_clk
    mufu = (1 - emu_frac) * scores / sm.mufu_exp_per_clk
    fma = scores * (base_fma_ops + emu_frac * emu_fma_ops) / sm.fma_per_clk
    smem = (t.m * t.d + 2 * t.n * t.d) * t.bytes_per_el / sm.smem_bytes_per_clk
    units = {"MMA": mma, "MUFU(exp)": mufu, "FMA": fma, "SMEM": smem}
    return _summarize(units)


def backward_cycles(sm: SM, t: Tile, smem_tiles: float = 13.0) -> dict:
    """Backward: 5 GEMMs (S recompute, dP = dO V^T, dV += P^T dO, dK += dS^T Q, dQ += dS K).

    ``smem_tiles`` = how many (n x d) bf16 tiles are read from shared memory per block.
    13 for one CTA and ~10.5 for the 2-CTA (M=256) MMA are the counts reported in the
    FA4 paper for d=128; they depend on the exact schedule, so treat them as inputs.
    """
    mma = 5 * t.m * t.n * t.d / sm.mma_mac_per_clk
    smem = smem_tiles * t.n * t.d * t.bytes_per_el / sm.smem_bytes_per_clk
    exp = t.m * t.n / sm.mufu_exp_per_clk  # P is recomputed from S and the saved LSE
    return _summarize({"MMA": mma, "MUFU(exp)": exp, "SMEM": smem})


def _summarize(units: dict) -> dict:
    ordered = sorted(units.items(), key=lambda kv: -kv[1])
    slowest, second = ordered[0], ordered[1]
    mma = units["MMA"]
    return {
        "units": units,
        "bottleneck": slowest[0],
        "headroom": 1 - second[1] / slowest[1],
        "mma_util_overlapped": mma / slowest[1],       # ceiling with perfect overlap
        "mma_util_serialized": mma / sum(units.values()),  # floor with zero overlap
    }


def best_emulation_fraction(sm: SM, t: Tile, **kw) -> tuple[float, dict]:
    """Smallest exp time over the fraction of exps moved from MUFU to FMA."""
    best = (0.0, forward_cycles(sm, t, 0.0, **kw))
    for i in range(1, 101):
        f = i / 100
        r = forward_cycles(sm, t, f, **kw)
        cur = max(r["units"]["MUFU(exp)"], r["units"]["FMA"])
        old = max(best[1]["units"]["MUFU(exp)"], best[1]["units"]["FMA"])
        if cur < old - 1e-9:
            best = (f, r)
    return best


def _print(title: str, r: dict) -> None:
    print(f"\n{title}")
    for k, v in r["units"].items():
        mark = "  <- slowest" if k == r["bottleneck"] else ""
        print(f"  {k:<10}{v:>8.0f} cycles{mark}")
    print(f"  headroom vs next unit: {r['headroom']:.0%}   "
          f"MMA util: {r['mma_util_serialized']:.0%} (serialized) .. {r['mma_util_overlapped']:.0%} (perfect overlap)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", nargs="+", default=["H100", "B200"])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--sweep", action="store_true", help="find the MUFU->FMA split")
    a = ap.parse_args()
    t = Tile(d=a.d)
    for g in a.gpu:
        sm = GPUS[g]
        _print(f"[{g}] forward, 128x128 tile, d={a.d}", forward_cycles(sm, t))
        if a.sweep:
            f, r = best_emulation_fraction(sm, t)
            _print(f"[{g}] forward with {f:.0%} of exps on FMA", r)
        if g == "B200":
            _print(f"[{g}] backward, 1 CTA", backward_cycles(sm, t, 13.0))
            _print(f"[{g}] backward, 2-CTA MMA", backward_cycles(sm, t, 10.5))

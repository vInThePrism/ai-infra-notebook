# Attention cycle model + software exp2

Model first, then tune. Background: [notes/02](../../notes/02-flashattention-1-to-4.md).

## `cycle_model.py`

For one 128×128 attention tile on one SM, counts the cycles each unit would need on its own:
Tensor Core (MMA), MUFU (exp), FMA pipe, shared-memory bandwidth. Prints the bottleneck, the
**headroom** to the next-slowest unit, and MMA utilization between "fully serialized" and "perfect overlap".

```bash
python cycle_model.py --sweep
```

| | MMA | exp | SMEM | bottleneck | headroom |
|---|---|---|---|---|---|
| H100 fwd | 2048 | 1024 | 768 | MMA | 50% |
| B200 fwd | 1024 | 1024 | 768 | MMA = exp | **0%** |
| B200 fwd, 21% of exp on FMA | 1024 | 809 | 768 | MMA | 21% |
| B200 bwd, 1 CTA | 2560 | 1024 | 3328 | SMEM | |
| B200 bwd, 2-CTA MMA | 2560 | 1024 | 2688 | SMEM | |

The B200 forward and backward numbers reproduce the per-tile counts in the FlashAttention-4 paper.
The FMA cost per score element (`base_fma_ops=5`) and per emulated exp (`emu_fma_ops=6`) are
assumptions: change them to match the SASS of the kernel you are tuning.

## `exp2_emulation.py`

`2^x = 2^floor(x) · p(frac(x))`: the integer part goes into the exponent bits, the fraction into a
Horner polynomial (one FMA per degree). Evaluated in float32 the way a kernel would, then compared to
exact exp2 in fp32 and after rounding to BF16 (what the PV GEMM consumes).

```
deg  max rel err (fp32)  bf16 identical  bf16 within 1 ulp
  2           2.620e-03         75.47%           100.00%
  3           1.154e-04         99.00%           100.00%
  4           4.076e-06         99.97%           100.00%
  5           1.770e-07        100.00%           100.00%
```

The fit is least squares on Chebyshev nodes with p(0) = 1 (so integer inputs, including the row max
itself, are exact). A true minimax (Remez) fit does slightly better at degree 3; the FA4 paper
reports 8.77e-5.

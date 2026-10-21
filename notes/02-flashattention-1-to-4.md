# FlashAttention 1 → 4: four GPUs, four bottlenecks

Code: [`tools/fa_cycle_model`](../tools/fa_cycle_model) (cycle model + software exp2),
[`tools/attn_check`](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/tools/attn_check) (correctness + timing),
[`kernels/window_attention_varlen`](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/demos/window_attention_varlen) (launch-bound fix).

One idea runs through all four versions: *a kernel is only as fast as its slowest unit, and every new
GPU makes a different unit the slowest.* Each version finds the new bottleneck and takes work off it.

| Version | GPU | Slowest unit | Key move |
|---|---|---|---|
| FA1 | A100 | HBM traffic | tiling + online softmax in one fused kernel |
| FA2 | A100 | non-matmul work, idle SMs | Q in the outer loop, fewer non-matmul FLOPs |
| FA3 | H100 | the synchronous execution model | TMA + WGMMA, warp specialization, overlap softmax with GEMM |
| FA4 | B200 | exponential unit (fwd), shared memory (bwd) | TMEM, part of exp on FMA, conditional rescaling, 2-CTA MMA |

When attention is slow in a real job, I check three layers:

1. Are the Tensor Cores fed? (math)
2. Do load and compute overlap? (data movement)
3. Does the speedup reach the hardware at all? (call granularity: launches, syncs, adapters)

FA1–FA4 mostly fix layers 1 and 2. In my own work, the biggest single win came from layer 3.

---

## FA1: stop writing the N×N matrix

Standard attention writes `S = QKᵀ` (N×N) to HBM, reads it back for softmax, writes `P`, and reads `P`
again for `PV`. The FLOPs are not the problem. The traffic is: attention is memory-bound.

- **Tiling:** load tiles of Q, K, V into shared memory and use each score block immediately. One fused
  kernel; the N×N matrix never exists in HBM.
- **Online softmax:** keep a running max `m` and running sum `l` per row. When a new tile raises the
  max, rescale the old sum and the old output by `exp(m_old − m_new)`. Divide by `l` at the end.
- Memory O(N²) → O(N). The backward pass saves one number per row (logsumexp) and recomputes S.
- Same FLOPs, much less HBM traffic ("IO-aware"). Exact, not approximate.

## FA2: why the outer loop is over Q

Start from the formula. Output row `o_i` needs **only** `q_i`, but **all** of K and V.
So `O_i` is an accumulator (with its `m_i`, `l_i`), and K, V are read-only inputs.

**Principle: the accumulator stays on chip; read-only inputs stream past it.**

FA1 put K/V in the outer loop. Then each `O_i` is touched once per KV block: read from HBM, rescale,
write back. With an 8K sequence and 128-wide blocks, that is 64 round trips per output block.
FA2 puts Q in the outer loop. A thread block owns `O_i` in registers from start to end and writes it
once.

Ownership also gives parallelism. In FA1, different KV blocks update the same `O_i`, so splitting the
KV loop across thread blocks means write conflicts; the grid is only `batch × heads`.
In FA2, Q blocks are independent, so the grid is `batch × heads × (N / block_M)`.
Example, long context with small batch on H100 (132 SMs): batch 1, 16 heads → FA1 launches 16 blocks
and leaves ~88% of SMs idle; FA2 at 8K / 128 launches 1024 blocks and fills every SM.

The same rule inside one thread block: FA1 split K across warps (split-K, warps must combine partial
results through shared memory); FA2 splits Q across warps (each warp owns its rows, no inter-warp
communication).

Also: fewer non-matmul FLOPs. Non-matmul math runs far slower than Tensor Core math, so FA2 rescales
by `1/l` once at the end instead of every iteration, and saves only logsumexp for backward.

## FA3: make attention asynchronous (Hopper)

FA2 reaches ~35% of peak on H100 while a good GEMM reaches 80–90%. FA2 is not badly written. Its
execution model (issue, wait, issue) does not match the hardware.

**Fact 1: data movement and math became separate async units.** TMA copies HBM → shared memory,
issued by one thread. WGMMA is an async matmul that reads operands from shared memory.
→ **Warp specialization:** a producer warpgroup only issues TMA into a circular buffer; consumer
warpgroups only do GEMM and softmax; they hand off through mbarriers. The producer needs few
registers, so `setmaxnreg` moves registers to the consumers.

**Fact 2: softmax is about half the GEMM time.** H100: ~989 TFLOP/s BF16 matmul vs ~3.9 TFLOP/s of
`exp` on MUFU (16 per SM per cycle), about 256× apart. With head dim 128 there are 512 matmul FLOPs
per `exp` (256 MACs: 128 in QKᵀ, 128 in PV), so `exp` takes ~50% of the matmul time. On the critical
path, that is half the kernel wasted. FA3 hides it at two levels:

- *Between warpgroups — ping-pong:* named barriers force an order so warpgroup 1 does softmax while
  warpgroup 2 does GEMM, then they swap.
- *Inside a warpgroup — 2-stage pipeline:* the PV GEMM of block j overlaps the softmax of block j+1.
  Cost: one extra S buffer in registers, which competes with tile size.

**Fact 3: FP8.** 2× the FP16 throughput, two catches. Layout: FP8 WGMMA wants K-major operands, so
V is transposed in-kernel (LDSM/STSM). Accuracy: E4M3 has 3 mantissa bits and LLM activations have
outliers, so FA3 uses per-block scales and multiplies Q and K by a random orthogonal (Hadamard)
matrix: `(QM)(KM)ᵀ = QKᵀ` because `MMᵀ = I`, but outliers get spread out. FA3's FP8 path is
forward-only, so training backward does not use it.

Reported result: 1.5–2× over FA2, up to ~740 TFLOP/s BF16 (~75% of peak).

### Layer 3 in practice: the speedup that did not reach the hardware

After switching a vision-language model to FA3, end-to-end speed barely moved. The vision tower uses
**window attention** with 64-token windows, and the adapter launched one kernel per window: about
one million launches in a 10-step replay.

Three signs in Nsight Systems:

1. Total attention GPU time did not change with the faster kernel.
2. Each kernel ran for only a few microseconds.
3. The GPU row was full of gaps while the CPU row was a solid wall of `cudaLaunchKernel`.

The GPU was waiting for the CPU: **launch-bound**. Fix: backport upstream varlen batching. All
windows go into one flat sequence, `cu_seqlens` marks the boundaries, and one launch computes every
window (it never computes cross-window pairs, rather than masking them).
Result: ~1M → 10K launches, attention GPU time 8.8 s → 4.0 s in a fixed 10-step replay.

## FA4: the exponential becomes the bottleneck (Blackwell)

On B200, attention did not speed up as much as the Tensor Cores did. So I built a cycle model first.

Per score element with head dim 128: the two GEMMs do 256 MACs, and there is 1 `exp`.

| | MMA per element | exp per element | MMA : exp |
|---|---|---|---|
| H100 (2048 MAC/clk/SM, 16 exp/clk/SM) | 0.125 cycle | 0.0625 cycle | 2 : 1 |
| B200 (4096 MAC/clk/SM, 16 exp/clk/SM) | 0.0625 cycle | 0.0625 cycle | **1 : 1** |

Per 128×128 forward tile on B200 (these match the FA4 paper): MMA 1024, exp 1024, shared memory 768
cycles. On Hopper, FA3 can hide softmax under a GEMM twice as long. On Blackwell there is **no
headroom**: the exponential unit is exactly as slow as the Tensor Cores, so any imperfection in
overlap goes straight into runtime, and with no overlap at all the kernel reaches only ~50% of matmul
peak. The faster Tensor Cores get, the less attention looks like a matmul problem. *The roofline
changed sides.*

> Careful with the wording: with *perfect* overlap the forward pass could still reach the MMA peak in
> this model. The 50% figure is the no-overlap floor. The real point is zero headroom: the
> exponential unit becomes a co-bottleneck, so FA4 must take work off it.

**Fact 1: TMEM changes who owns the accumulator.** On Hopper, the WGMMA result is spread across the
registers of 128 threads, so one row lives in several threads and a row max needs shuffles. On
Blackwell, `tcgen05.mma` is issued by one thread and writes to Tensor Memory (256 KB per SM). A tile
has 128 rows and a warpgroup has 128 threads, so each softmax thread reads one full row and does its
own max and sum. *The thread is not bigger; the accumulator moved.*

Because S, P and O live in TMEM, FA4 pipelines more stages: a TMA warp loads K/V; an MMA warp computes
QKᵀ into TMEM; softmax warpgroups turn S into P in TMEM; a separate **correction** warpgroup rescales
O; the MMA warp reads P directly for PV. Each CTA works on two Q tiles that alternate between softmax
and MMA.

**Conditional rescaling (an algorithm change, not a pipeline change).** The running max only has to
be a shared reference for numerator and denominator; it does not have to be the true max. FA4 keeps
the old reference and rescales only when the max grows by more than 8 (in log₂, i.e. 256×). Most
rescales never happen. The correction warpgroup moves rescaling *off the critical path*; conditional
rescaling *skips most of it*. Two different fixes.

**Fact 2: compute some exponentials on the FMA pipe.** MUFU is full, FMA is not (128 FMA vs 16 exp
per SM per cycle).

```
2^x = 2^floor(x) · 2^frac(x)
      └ integer: add floor(x) to the float's exponent bits (no math)
                    └ fraction in [0,1): degree-3 polynomial, Horner = 3 FMAs
```

Only ~10–25% of the exponentials move: more would cost registers, instructions and latency. The point
is not that software exp is faster; it is that sharing the work raises total throughput. My
reimplementation ([`exp2_emulation.py`](../tools/fa_cycle_model/exp2_emulation.py)): a least-squares
degree-3 fit gives max relative error 1.2e-4 in fp32, and after rounding P to BF16 99% of values are
bit-identical and 100% are within 1 ULP. (The FA4 paper reports 8.77e-5 with a minimax fit.) The cycle
model puts the MUFU/FMA balance point at ~21% under its FMA-cost assumptions.

**Fact 3: the backward pass is shared-memory bound.** Five GEMMs, but even more operand traffic: per
128 tile, MMA 2560 cycles vs shared memory 3328 (+30%). FA4 pairs two CTAs into one M=256 MMA
(2-CTA UMMA); each CTA stages only half of the shared operand, shared-memory time drops to ~2688
cycles, and dS is exchanged through distributed shared memory, which also cuts dQ atomics.

Reported result (FA4 paper, B200 BF16): 1.1–1.3× over cuDNN 9.13, 2.1–2.7× over Triton, up to
1613 TFLOP/s (~71%), and CuTe DSL compiles in ~2.5 s instead of ~55 s for C++ templates.

### My FA4 work

- **B200, tuning the upstream CuTe DSL kernel:** the cycle model said the forward pass on our shapes
  was MUFU-bound, so I tuned the MUFU/FMA split, the polynomial degree under our accuracy limit, the
  tiles, and the work balance: 1.15× geometric-mean speedup over 8K–64K sequence lengths.
- **B300, FA4 as a first-class backend in LLaMA-Factory:** fixed a packed-sequence path that built a
  dense 4D mask and ran out of memory, moved the CuTe DSL JIT cache off NFS, and measured 8–9× training
  throughput over PyTorch SDPA in controlled A/B runs.
- **Proof:** a 48-shape matrix (ViT MHA / LLM GQA × varlen / causal / varlen+causal × 8K–64K × head
  dim 64/128), plus edge cases sampled by bug probability; normalized L2 ≤ 1e-2 for out, dQ, dK, dV,
  max error reported, NaN/Inf = fail; 50 warm-up / 200 timed / 3 runs; then fixed 8-GPU jobs;
  per-architecture fallback for anything that failed or regressed.

## In one sentence

A kernel is only as fast as its slowest unit, and every new GPU changes which unit that is: model
first, then tune, and prove it end to end.

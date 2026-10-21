# MoE training: grouped GEMM, expert parallelism, and all-to-all

Code: [`kernels/grouped_gemm`](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/kernels/triton/grouped_gemm) (Triton varlen-M grouped GEMM + full MoE FFN,
tested against a per-expert loop).

## The shape flow

```
x [B, S, H] → flatten → [T, H]
router: [T, H] @ [H, E] → softmax → top-k → (expert id, weight) for T·k assignments
dispatch (all-to-all under EP) → permute: sort the T·k rows by expert
grouped GEMM gate+up: [rows_e, H] @ [H, 2I] for every expert e → SwiGLU → [rows_e, I]
grouped GEMM down:    [rows_e, I] @ [I, H]
unpermute → combine (all-to-all back) → weighted sum of each token's k outputs → [T, H]
```

The FFN is **per token**, not element-wise, and there is **one grouped GEMM per linear layer**, not one
kernel for the whole FFN.

## Why a for-loop over experts is slow

A loop launches E small GEMMs. Token counts are uneven, so many GEMMs have tiny M (a cold expert may
get 3 tokens) and leave most SMs idle. And the host needs every count to size each launch, which
forces a device → host sync (`.item()`, `.tolist()`, `nonzero`) in the middle of the step.

## What grouped GEMM does

Two things differ between experts, and each is found a different way:

- **Weights:** every expert's weight has the *same shape*, only different values. A tile finds its
  expert's weight by **indexing**: `w + e * stride_e`.
- **Tokens:** experts get *different numbers* of rows. After the permute, expert e owns rows
  `[m_off[e], m_off[e+1])`, where `m_off` is the **prefix sum** of the counts, computed on the GPU.

All experts' M-tiles form one pool. Program `pid` finds its expert by comparing `pid` with the prefix
sum of *tiles* per expert. The launch grid uses a static bound, `ceil(total_rows / BM) + E`, so the
host never reads the counts: no sync, CUDA-graph friendly, one launch keeps every SM busy. Programs
past the real tile count exit immediately.

## Expert parallelism and its communication

EP places different experts on different GPUs. Each token must travel to the GPUs that hold its top-k
experts and come back: **two all-to-alls per MoE layer** in forward (dispatch, combine), two more in
backward.

Problems with all-to-all:

- heavy and often cross-node (InfiniBand, not NVLink);
- imbalanced: hot experts receive more tokens;
- hard to overlap inside one micro-batch, because the next computation needs the data.

Fixes: overlap across micro-batches (DualPipe-style schedules, or interleaved 1F1B with explicit
stream events), node-limited routing with IB-then-NVLink forwarding, load-balancing losses or bias
terms, FP8 dispatch, and specialized kernels (DeepEP from DeepSeek; NVIDIA's Hybrid-EP builds on the
same ideas). Expert kernels and communication kernels compete for SMs, so the number of SMs reserved
for communication is a tuning knob.

## 64 GPUs, EP = 8: who holds what, who syncs what

- **Attention (and other dense) weights** are replicated across data parallel. Their gradients are
  all-reduced (or reduce-scattered with a distributed optimizer) over all data-parallel ranks.
- **Expert weights** are split across the EP group of 8: each GPU holds E/8 experts. The 64 GPUs form
  8 EP groups, so each expert has 8 replicas, one per EP group. Expert gradients are synchronized only
  across those 8 replicas: the **expert-data-parallel (EDP)** group, not the full DP group.

## A debugging story: rule things out with a control experiment

Setup: multimodal MoE SFT (30B total / 3B active) on 32 H800s, Megatron-Core.

1. **Decompose first.** Nsight Systems on a reproducible 8K-sequence baseline: compute 48%, exposed
   communication 34%, host wait 18%. MoE dispatch/combine was 41% of module time.
2. **The obvious hypothesis was expert imbalance.** Control: force uniform routing. The step got <3%
   faster, so imbalance was not the problem.
3. **What was left:** fragmented per-expert kernels (gather → projection → scatter for each expert)
   and serialized stream dependencies.
4. **Kernels:** integrated a varlen-M grouped GEMM (SonicMoE), adapting the expert weight layout, the
   routing weights, and autograd to the kernel's input contract. Launches per layer 48 → 10; expert FFN
   time 4.8 s → 2.6 s per step.
5. **Communication:** after integrating DeepEP, a full-batch dispatch barrier still blocked overlap.
   Coordinated stream events across micro-batches under interleaved 1F1B and swept 16/20/24 SMs for
   communication: exposed communication 4.9 s → 1.4 s per step.
6. **Proof:** output and expert/input-gradient parity (normalized L2 ≤ 1e-2), three 100-step
   steady-state runs (throughput CV < 2%), 91.1% weak-scaling efficiency from 8 to 32 GPUs.

End to end: 14.4 s → 7.2 s per step, MFU 10.2% → 20.4%, with token budget and sample order fixed.

## In one sentence

Weights have one shape, so index them; token counts differ, so prefix-sum them on the GPU; then one
launch serves every expert, and the communication is hidden across micro-batches, not inside one.

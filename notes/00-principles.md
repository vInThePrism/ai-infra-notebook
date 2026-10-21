# Five principles I use for training performance work

Everything else in this repo is one of these five ideas applied to a specific problem.

## 1. A kernel is only as fast as its slowest unit, and every new GPU changes which unit that is

A GPU is several machines that run side by side: Tensor Cores, the exponential unit (MUFU),
the FP32/FMA pipe, shared-memory bandwidth, HBM bandwidth, and the CPU that launches the kernels.
Speed comes from the slowest one. Each hardware generation grows these units by different amounts,
so the slowest one moves.

- A100: HBM was the limit for attention, so FlashAttention-1 stopped writing the N×N matrix.
- H100: the synchronous issue-and-wait model was the limit, so FlashAttention-3 went async.
- B200: Tensor Cores doubled but the exponential unit did not, so FlashAttention-4 moved part of
  `exp` to the FMA pipe.

How I use it: **model first, then tune.** Before touching tile sizes, count the cycles each unit needs
for one tile ([`tools/fa_cycle_model`](../tools/fa_cycle_model)). The model tells me where to look.

## 2. Count the work from the model, never from the kernels

MFU = logical model FLOPs / (step time × GPUs × dense peak).
With the model, the data and the training semantics fixed, the numerator is fixed. The hardware is
fixed, so the denominator is fixed. The only thing allowed to move MFU is step time.

If switching the attention backend changes the numerator, the meter is broken. It says nothing about
the optimization. So the numerator comes from the model config plus runtime token counts, split into
forward, input-gradient and weight-gradient per module ([`tools/logical_mfu`](https://github.com/vInThePrism/logical-mfu)).

Self-check: **the MFU ratio must equal the throughput ratio.** 14.4 s → 7.2 s per step must give
exactly 2.00× MFU. If not, the numerator moved.

## 3. Accumulators stay on chip; read-only inputs stream past them

Whoever owns the output keeps it in registers or on-chip memory for the whole reduction and writes
it once. Inputs that are only read flow past it.

- FlashAttention-2 puts Q in the outer loop: each thread block owns O_i, m_i, l_i and writes O_i once.
  FA1 (KV outer) wrote and re-read O_i once per KV block.
- Ring attention keeps Q on its GPU and rotates K/V.
- Grouped GEMM gives each tile one expert's output rows and streams that expert's weight.

Ownership also gives parallelism for free: independent owners need no synchronization.

## 4. Decompose the step before optimizing anything

Split step time into **compute**, **exposed communication**, and **host wait** (Nsight Systems,
Holistic Trace Analysis). Then optimize the biggest part, and use *control experiments* to rule
hypotheses out:

- Is MoE slow because of expert imbalance? Force uniform routing. If the step gets <3% faster,
  imbalance is not the problem. Look at kernel fragmentation and stream dependencies instead.
- Is the new attention kernel not helping? Check whether attention GPU time moved at all. If the
  timeline has gaps and the CPU row is full of launches, it is launch-bound: fix the call granularity,
  not the kernel.

## 5. A speedup only counts if it is correct and real

- **Correct:** check output and gradients against an fp32 reference with normalized L2, max error,
  and a hard fail on NaN/Inf. Choose shapes by bug probability (tail blocks, empty sequences,
  q_len ≠ k_len), not only by the production distribution ([`tools/attn_check`](https://github.com/vInThePrism/gpu-kernel-lab/tree/main/tools/attn_check)).
- **Real:** warm up, rotate inputs to beat L2, repeat runs and report the CV, lock clocks.
  Then re-measure inside a fixed end-to-end job with the same token budget and sample order.
  A microbenchmark win that does not show up end to end is not a win.
- **Safe:** keep a per-architecture fallback for every configuration that fails or regresses.

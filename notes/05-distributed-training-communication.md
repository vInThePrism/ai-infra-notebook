# What communicates during training, and how to hide it

## Two families: replicate vs shard

| | What each GPU holds | Gradient sync | Parameter traffic |
|---|---|---|---|
| DP / DDP | full model, full optimizer | **all-reduce** gradients | none |
| ZeRO-1 / distributed optimizer | full params + grads, 1/N of optimizer states | reduce-scatter grads | all-gather updated params |
| ZeRO-2 | + 1/N of gradients | reduce-scatter grads | all-gather params |
| ZeRO-3 / FSDP | + 1/N of parameters | reduce-scatter grads | all-gather params in forward **and again in backward** |

Volumes (Ψ = parameter count): DP, ZeRO-1 and ZeRO-2 move ~2Ψ per step (reduce-scatter + all-gather
is the same as one all-reduce). ZeRO-3 moves ~3Ψ (1.5×) because parameters are gathered twice.

Things that are easy to mix up:

- Plain DDP does **not** split work or memory: every replica holds everything and applies the same
  averaged gradient. Sharding is ZeRO / FSDP.
- DP all-reduces **gradients**, never weights. In ZeRO, parameters are **all-gathered**, never reduced.
- FSDP ≈ ZeRO-3 algorithmically. FSDP1 flattens parameters into one buffer per unit; FSDP2 shards
  each parameter as a DTensor.

## Where the gradient goes (and why 16 bytes per parameter)

Mixed-precision Adam per parameter: BF16 weight (2) + BF16 gradient (2) + FP32 master weight (4) +
FP32 first moment m (4) + FP32 second moment v (4) = **16 bytes**. The gradient updates m and v;
m and v update the FP32 master weight; the master weight is cast back to BF16.
A 7B model needs ~112 GB of model states before a single activation.

## Ring all-reduce

Reduce-scatter (N − 1 steps) + all-gather (N − 1 steps) = 2(N − 1) steps. Each GPU sends
2(N − 1)/N · S ≈ 2S bytes, independent of N: bandwidth-optimal. Ring wins for large messages;
trees have O(log N) latency and win for small messages at scale. NCCL picks with a cost model of size,
rank count and topology (`NCCL_ALGO` overrides).

Why split all-reduce into reduce-scatter + all-gather? Each rank can update only its 1/N shard between
the two halves (the distributed optimizer), and the two halves can be placed at different points in
the step (SP: all-gather before a block, reduce-scatter after it).

## Hiding communication

- **DDP buckets:** gradients are grouped into ~25 MB buckets. An autograd hook fires when a bucket is
  full and starts its all-reduce while backward continues on earlier layers.
- **Megatron-LM:** overlap-grad-reduce and overlap-param-gather (distributed optimizer), TP comm/GEMM
  overlap (Transformer Engine), interleaved 1F1B (smaller bubble: (p − 1)/(v·m) instead of
  (p − 1)/m), scatter-gather of PP sends, and zero-bubble schedules that split backward into
  input-gradient (critical path) and weight-gradient (fills bubbles). The same dgrad / wgrad split
  as in the MFU numerator.

**How to know it is really hidden:** not GPU utilization, because NCCL kernels also occupy SMs and
look "busy". In Nsight Systems, measure **exposed communication**: the time when a communication kernel
runs and no compute kernel does. That number, not bandwidth, is what costs step time.
A common trap: collectives can run at >85% of link bandwidth and still be ~35% of the step, because
they are exposed, not slow.

## Topology decides placement

Rough per-GPU bandwidth: NVLink 4 (H100) ~900 GB/s bidirectional, NVLink 5 (B200) ~1.8 TB/s;
InfiniBand NDR ~50 GB/s per 400 Gb/s NIC; PCIe Gen5 x16 ~64 GB/s per direction.

Rule: **the most frequent, critical-path traffic goes on the fastest links.**
TP (4 all-reduces per layer, on the critical path) inside a node; PP (point-to-point, once per
micro-batch per stage boundary) across nodes; DP outermost (once per step, overlappable).
A standard 64-GPU layout: TP=8 in-node, PP across nodes, DP outermost; for MoE, EP usually stays inside
a node where possible because all-to-all is heavy.

## Pipeline stages: balance by measured cost, not layer count

Equal layer counts per stage are not equal work: the first stage also has the embedding (and, in VLMs,
the vision tower), and the last has the output projection over a large vocabulary. Measure per-layer
cost and cut stages so each stage's time is equal. In a 32K-context job with 4 physical stages × 2
virtual chunks under interleaved 1F1B, cost-based balancing cut the bubble fraction from 24% to 12%.

## Migrating backends without changing the experiment

Moving a job from DeepSpeed ZeRO-3 to Megatron-Core is only a performance result if the model is the
same model. Checks that made it trustworthy: Hugging Face → Megatron weight mapping, then parity on
GQA, RoPE, document-boundary masks, token-weighted loss and gradients at 8/32/64 GPUs, and a matched
300-step run whose final-50-step mean loss differed by <0.5%.

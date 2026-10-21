# Tensor, sequence and context parallelism, derived from the formulas

## Tensor parallelism (Megatron): why column first, then row

MLP: `Y = GeLU(X A)`, `Z = Y B`.

- Split **A by columns** across t GPUs: each GPU computes `GeLU(X A_i)` on its own slice. GeLU is
  element-wise, so no communication is needed before it. (Splitting A by rows would give partial sums,
  and GeLU of a partial sum is wrong, so you would have to all-reduce first.)
- Split **B by rows**: each GPU computes `Y_i B_i`, a partial sum of Z. One **all-reduce** gives Z.

Attention is the same pattern: q/k/v projections split by columns (= by heads, each GPU owns whole
heads), the output projection split by rows, one all-reduce.

Per transformer layer: forward 2 all-reduces (attention, MLP), backward 2 (the conjugate ones on the
input gradients) = **4 per layer**. They sit on the critical path, which is why TP stays inside a
node, on NVLink.

## Sequence parallelism: the part TP left replicated

TP splits activations only *inside* attention and MLP. LayerNorm, dropout and the residual add have no
large weight matrix to split, so under plain TP every GPU holds the full `[S, B, H]` activation for
them. For long sequences that is a lot of memory.

These ops are **per token**: token s does not need token s'. So split them along the sequence
dimension with zero communication inside the op. The only change is at the boundaries:

```
TP:      [full seq] → block → all-reduce → [full seq]
TP + SP: [seq/t] → all-gather → block → reduce-scatter → [seq/t]
```

"All-gather in, reduce-scatter out." An all-reduce *is* a reduce-scatter followed by an all-gather,
so the communication volume is the same as TP; SP just moves the two halves to different places and
divides the replicated activations by t.

## Context parallelism: split the sequence through the whole layer

CP splits the sequence for *every* op in the layer. Per-token ops (MLP, norms, projections) need no
communication. Only attention mixes tokens. Which part has to travel? Read it off the formula for one
output row:

```
s_ij = q_i · k_j / sqrt(d)
p_ij = exp(s_ij) / Σ_j' exp(s_ij')
o_i  = Σ_j p_ij · v_j
```

`q_i` appears once. `k_j` and `v_j` appear for every j. `q_m` (m ≠ i) never appears.
So **`o_i = f(q_i, K, V)`**: each output row needs its own query and *all* keys and values.

Split rows across GPUs: each GPU keeps its own queries and builds its own outputs; **K and V are what
must travel.**

## Ring attention: keep Q, stream K/V, merge exactly

Cut the sum over j into blocks. GPUs form a ring; at each step every GPU computes attention of its Q
against the K/V block it holds, then passes that block to the next GPU (and overlaps the send with
the compute). After CP steps every Q has met every K/V block.

Each block gives a partial result for row i: local max `m`, local sum `l = Σ exp(s − m)`, and
`u = Σ exp(s − m) · v`. Two partials merge exactly:

```
m = max(m_a, m_b)
l = e^(m_a − m) · l_a + e^(m_b − m) · l_b
u = e^(m_a − m) · u_a + e^(m_b − m) · u_b
o_i = u / l   (after the last block)
```

This is the online softmax from FlashAttention, used across GPUs instead of across tiles.

**Causal load balance.** With a causal mask, the GPU that holds the last chunk of the sequence has
work against every earlier block, while the first one has almost none. **Zigzag** chunking cuts the
sequence into 2·CP chunks and gives GPU r chunks r and 2·CP − 1 − r: one early and one late chunk each,
so every GPU gets the same amount of unmasked work.

## When to use which

| | TP + SP | CP |
|---|---|---|
| Splits | weights (TP) + per-token activations (SP) | the sequence, through every op |
| Communication | AG + RS per block, critical path, every layer | K/V ring inside attention only, overlappable |
| Placement | inside a node (NVLink) | can span more GPUs |
| Use it for | models too big for one GPU's weights | sequences too long for one GPU's activations |

Real configs combine them. For 32K-context continued pretraining of a 32B model on 64 H800s, a
measured search ended at TP=2 / PP=4 / CP=2 / DP=4 with SP; co-tuning CP/SP with selective
recomputation cut peak memory 74 → 62 GiB and recompute overhead from 31% to 17% of useful FLOPs.

## In one sentence

Row i of attention needs only query i but all of K and V, so keep Q where it is, move K/V around a
ring, and merge the partial softmaxes exactly.

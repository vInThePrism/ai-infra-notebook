# Attention as a soft database lookup — and why the KV cache falls out of it

The question I started from: during decoding, why do we cache K and V but never Q?

## Q is the output side; K and V are the context side

For a query at position `i`:

```
o_i = Σ_j softmax_j(q_i · k_j / √d) · v_j
```

- `q_i` is used to produce exactly **one** output row, `o_i`. Once `o_i` exists, `q_i` is never needed again.
- `k_j` and `v_j` are read by **every later** query `i ≥ j` (causal masking). Under causal attention, earlier tokens never see later ones, so `k_j`, `v_j` never change once computed.

So K and V are reused, unchanged, by all future steps, and Q is used once. That's the whole reason for a KV cache: cache what will be read again.

## The database analogy (with the parts that are wrong)

Think of the KV cache as a table with one row per past token:

| row (token j) | key column `k_j` | value column `v_j` |
|---|---|---|

A query does something like:

```sql
SELECT v FROM kv_cache WHERE k MATCHES q     -- with a soft, weighted WHERE
```

- **K is the column you match on** (the `WHERE`).
- **V is the column you return** (the `SELECT`). They are different projections of the same token, because "what makes this token relevant" and "what information this token carries" are different things.
- **The match is soft.** Instead of returning matching rows, every row gets weight `softmax(q · k)`, and the result is the weighted average of all `v`. There's no threshold; a row is never fully excluded (except by masking).

What the analogy explains well:

| Database idea | Attention fact |
|---|---|
| append-only table | the KV cache: each decode step appends one row (per layer, per KV head) and never updates old rows |
| query = one-off | Q is not cached |
| table size | KV cache bytes = 2 · layers · seq_len · kv_heads · head_dim · bytes/elem — linear in sequence length, per request |
| cross join | `QKᵀ` compares every query with every key: O(n²) in prefill |
| fewer, wider index columns | GQA/MQA: several query heads share one K/V head, so the table shrinks by the group factor |
| distributed nested-loop join | ring attention (context parallelism): each rank keeps its query block fixed and K/V blocks rotate around the ring |

Where it breaks: in *self*-attention, Q, K and V are all projections of the same sequence, so it's "a table querying itself"; and unlike a database, the "query language" (the projection weights) is learned. In *cross*-attention (encoder–decoder) the analogy is closer: queries come from one source, the table from another.

## Why the same asymmetry shows up in kernels and parallelism

The "Q is the output side" observation decides loop order and data placement in three places:

- **FlashAttention-2:** the outer loop (and the parallelization across thread blocks) is over **Q blocks**. Each block owns its output rows and their running `(m, l, O)` accumulators, and streams K/V tiles through. FA1 had K/V outer, so partial outputs had to be read and rewritten in HBM on every outer iteration.
- **Ring attention:** **K/V circulate, Q stays.** The output and accumulators live where the query lives, so nothing has to come back. Under GQA, K+V is also smaller than Q, so moving K/V is cheaper.
- **Decode:** one query, a long table. The work is reading the whole KV cache once, so decode attention is memory-bound, and flash-decoding splits the *table* (KV length) across blocks and merges the partial `(m, l, O)` — the same monoid merge as online softmax ([note 03](https://github.com/vInThePrism/gpu-kernel-lab/blob/main/notes/03-reductions-are-monoids.md)).

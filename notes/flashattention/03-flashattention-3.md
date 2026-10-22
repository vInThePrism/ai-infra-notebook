# FlashAttention-3: make everything asynchronous on Hopper

**Question:** FA2 is a good kernel, so why does it reach only about 35% of an H100, and what does FA3 change to get to about 75%?

- GPU: Hopper (H100)
- Slowest unit: tensor cores waiting, first on synchronous data loads, then on softmax's exponentials.
- Fix: asynchrony (TMA + WGMMA), warp specialization, two-level overlap of softmax and GEMM, FP8.

Series: [overview](../02-flashattention-1-to-4.md) · [FA1](01-flashattention-1.md) · [FA2](02-flashattention-2.md) · **FA3** · [FA4](04-flashattention-4.md)

---

## 1. Starting point: FA2 reaches about 35% on H100

H100 SXM5: 132 SMs, about **989 TFLOPS** dense FP16/BF16 on tensor cores, about 3.35 TB/s HBM3, up to 228 KB shared memory per SM.

FA2 reaches only about **35%** of peak on H100. FA2 was written for Ampere: it uses `mma.sync` (synchronous tensor-core instructions issued per warp) and doesn't use the hardware Hopper added:

- **TMA** (Tensor Memory Accelerator) for asynchronous data loads
- **WGMMA** (warpgroup MMA) for asynchronous matmul. Hopper's tensor cores only reach full throughput through WGMMA.

In FA2's program order, every step waits for the previous one: load → matmul → softmax → matmul → load …

---

## 2. Hopper's new hardware

- **TMA (Tensor Memory Accelerator):** a dedicated unit that copies multi-dimensional tiles between HBM and shared memory **asynchronously**. Address calculation and bounds checks happen in hardware. One thread can issue a whole tile copy, which frees threads and registers.
- **WGMMA (`wgmma.mma_async`):** a matmul issued by a **warpgroup** (4 warps, 128 threads). It runs **asynchronously** and can read operands directly from shared memory. While it runs, the warpgroup can do other work, such as softmax.
- **mbarrier:** shared-memory barriers with arrival counts / transaction bytes. Producers and consumers signal "buffer full" / "buffer empty" to each other without a block-wide sync.
- **`setmaxnreg`:** lets a warpgroup give up or take registers at runtime. Producers need few registers; consumers (holding S, P, O tiles) need many.

---

## 3. Technique 1: producer–consumer warp specialization

Different warpgroups in the same thread block get **different jobs**.

**Producer warpgroup**

- Uses TMA to copy $K_j, V_j$ (and $Q_i$ once) into a **circular buffer** of shared-memory stages.
- Waits on a stage's "empty" barrier before reusing it.
- Gives up registers with `setmaxnreg`.

**Consumer warpgroups** (usually 2)

- Wait on a stage's "full" barrier, then run WGMMA $S = QK_j^\top$ → softmax → WGMMA $O \mathrel{+}= P V_j$.
- Mark the stage "empty" when done with it.
- Take more registers with `setmaxnreg`.

Result: data movement fully overlaps with compute. While consumers work on stage $k$, TMA is already filling stages $k+1, k+2, \dots$, so the tensor cores no longer wait for loads.

---

## 4. Technique 2: hide softmax behind matmul

### 4.1 Why softmax becomes the bottleneck on Hopper

On H100 SXM5:

- Tensor cores (FP16 matmul): about 989 TFLOPS
- Special function unit, MUFU (exp, log, …): about 3.9 TFLOPS

With head dim $d = 128$, attention does about **512×** more matmul FLOPs than exponentials. But matmul throughput is only about 250× higher than exp throughput, so the exponentials take about **half** as long as the matmuls. If softmax runs serially *after* each matmul, the tensor cores sit idle for all of that time.

Matmul (tensor cores) and exp (MUFU) are **different hardware units** and can run at the same time. FA3 uses two levels of scheduling to overlap them.

### 4.2 Level 1: ping-pong between warpgroups

- Two consumer warpgroups (WG1, WG2) are forced by **named barriers** to alternate:
  - while WG1 runs its GEMMs, WG2 runs softmax;
  - then they swap.
- After warm-up, the tensor cores always have a GEMM to run, while the other warpgroup's softmax runs on the MUFU/FMA units.

### 4.3 Level 2: pipelining inside one warpgroup

Inside a **single** warpgroup, break the in-order dependency chain (Algorithm 2 in the FA3 paper):

```text
naive (FA2-style):  S_j = Q K_jᵀ → softmax(S_j) → O += P_j V_j → S_{j+1} = Q K_{j+1}ᵀ → …

pipelined (FA3), iteration j:
    issue WGMMA  S_{j+1} = Q K_{j+1}ᵀ     (async)
    issue WGMMA  O += P_j V_j             (async)
    wait for S_{j+1}
    softmax(S_{j+1})                      (overlaps with the P_j V_j WGMMA still running)
    wait for P_j V_j, then rescale O
```

- Because WGMMA is asynchronous, the PV matmul of block $j$ keeps the tensor cores busy while the warpgroup computes the softmax of block $j+1$.
- The cost is **more registers**: $S_{j+1}$ and $P_j$ have to be held at the same time, which competes with tile size and register pressure.

---

## 5. Technique 3: low precision (FP8)

Hopper's FP8 tensor cores have about **2×** the FP16 throughput. FP8 attention has two problems, and FA3 handles both.

### 5.1 Layout constraints

- FP8 WGMMA requires operands in a specific **k-major** layout, and $V$ usually isn't (for $PV$, the reduction dimension is the sequence dimension). FA3 **transposes V tiles inside the kernel** after they are loaded into shared memory (using `LDSM`/`STSM`, i.e. ldmatrix/stmatrix), overlapped with other work.
- The FP8 **accumulator layout** of the first GEMM (which produces $P$) doesn't match the **operand layout** the second GEMM needs for $P$. FA3 fixes this with an **in-register byte permute**, avoiding a round trip through shared memory.

### 5.2 Accuracy

FP8 has very few mantissa bits, and LLM activations have **outlier features** that blow up quantization error. Two fixes:

1. **Block quantization:** one scale per tile (e.g. per $B_r \times d$ block of Q) instead of one per tensor.
2. **Incoherent processing:** multiply both $Q$ and $K$ by the same random orthogonal matrix $M$ (a Hadamard matrix with random signs). Since $MM^\top = I$:

$$(QM)(KM)^\top = QK^\top$$

The attention scores don't change, but the outliers get **spread across dimensions**, so quantization error drops. The Hadamard transform costs $O(d \log d)$ and can be fused with RoPE.

Result: FP8 FA3 has about **2.6× lower** numerical error than an FP8 baseline with per-tensor scaling.

---

## 6. Backward pass

- Uses the same TMA-based warp specialization and asynchronous WGMMA.
- As in FA2, backward parallelizes over K/V blocks, and $dQ$ is accumulated across thread blocks with atomic adds.
- About **1.5–1.75×** faster than FA2 backward.

---

## 7. Results (H100 SXM5)

All figures are as reported in the FA3 paper; I have not reproduced them.

- **FP16/BF16 forward:** up to about **740 TFLOPs/s**, about **75%** of theoretical peak (FA2: about 35%).
- Forward is **1.5–2.0×** faster than FA2.
- **FP8 forward:** close to **1.2 PFLOPs/s**.
- **FP8 accuracy:** 2.6× lower error than the FP8 baseline.

---

## 8. Summary

1. FA2 stalls at about 35% on Hopper because it is **synchronous**; Hopper needs the **asynchronous** TMA and WGMMA.
2. **Warp specialization:** a producer warpgroup only moves data (TMA → circular buffer); consumers only compute.
3. **exp is slow** (3.9 vs 989 TFLOPS): **ping-pong** between warpgroups and **pipelining** inside a warpgroup hide softmax behind the matmuls.
4. **FP8:** an in-kernel V transpose and register permutation solve the layout problems; block quantization and Hadamard incoherent processing solve the accuracy problems.
5. Result: about 75% of peak in FP16, about 1.2 PFLOPs/s in FP8.
6. On Blackwell, even overlapped exp is too slow, so FA4 moves part of the exponentials onto the FMA units. See [FA4](04-flashattention-4.md) and the [overview](../02-flashattention-1-to-4.md#fa4-the-exponential-becomes-the-bottleneck-blackwell).

---

## References

- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Dao, *FlashAttention-2* (FA2). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

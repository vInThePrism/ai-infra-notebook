# AI Infra / CUDA Interview Question Bank (Q&A + Hand-Coding)

Compiled 2026-09. Conceptual questions are grouped by competency block; hand-coding problems are in a
table at the end. Each question carries a tag showing where it came from; ★ marks questions that appear
repeatedly across multiple interview reports. All questions are rewritten, not copied.

Answers are intentionally left blank. A question only counts as passed once you have written your own
bullet-point answer and then explained it out loud without notes. My own answers to many of these are
in [`../notes`](../notes).

**Source tags:**

- 〔NV〕 First-hand NVIDIA interview reports (Nowcoder, 1point3acres, Glassdoor).
- 〔ByteDance〕〔Kuaishou〕〔Baidu〕〔Meituan〕〔Alibaba〕〔Tencent〕〔Ant〕〔Xiaohongshu〕〔Tongyi〕 AI Infra interview reports from Chinese tech companies.
- 〔CN〕 Chinese AI Infra interview reports where the post did not name the company.
- 〔HPC〕 Chinese HPC / chip-company interview reports.
- 〔Bank〕 Curated question banks, not a single interview.
- 〔Overseas〕 Interview reports or JDs from overseas companies.
- 〔Added〕 Not in any report; added from job descriptions and a competency checklist.

---

## 1. GPU Architecture and the CUDA Execution Model

- What is SIMT? How does it differ from CPU SIMD? 〔HPC〕
- ★ What is the hierarchy of grid / block / warp / thread? How does an SM schedule threads for execution? 〔NV〕
  - Follow-up: How many warp schedulers does an SM have, and how many instructions can each issue per cycle?
  - Follow-up: When a warp is waiting on memory, how does the scheduler switch? Why is the switch nearly zero-cost?
- ★ How do you choose grid and block dimensions? What is the relationship between block=256 and warps? Why is block size usually a multiple of 32? 〔NV〕〔Baidu〕
- Can one block be scheduled across multiple SMs? How many blocks can reside on one SM at once, and which resources limit this? 〔HPC〕
- ★ What is warp divergence? How is an if-else executed in hardware? What is its performance impact? 〔NV〕〔HPC〕
  - Follow-up: What did independent thread scheduling (Volta onward) change? Why do warp-level primitives take a mask?
  - Follow-up: Odd columns of a matrix compute log, even columns compute cos — when divergence and coalescing conflict, how do you trade them off? 〔NV〕
- ★ What synchronization mechanisms does CUDA offer? What is the scope of `__syncthreads`, `__syncwarp`, cooperative groups, atomics, and memory fences? 〔NV〕
  - Follow-up: What happens if `__syncthreads` is placed inside a branch?
- ★ What is the difference between CUDA Cores and Tensor Cores? When should each be used? 〔Kuaishou〕〔Bank〕
- Why must a kernel include `if (idx < n)`? What problem does a grid-stride loop solve? 〔Baidu〕
- How do GPU and CPU design philosophies differ? Why does the GPU rely on massive numbers of threads to hide latency? 〔Added〕
- GPUs have no push instructions — who is responsible, at the software level, for passing kernel arguments? 〔Alibaba〕
- What is CPU–GPU unified addressing (UVA, Unified Memory)? Where is the cost of on-demand migration? 〔Alibaba〕
- Roughly how many SMs, how much memory capacity and bandwidth, and how much BF16 compute do common cards (A100, H100, B200) have? 〔HPC〕〔CN〕
- At which hardware level are global atomics executed? Why do multiple threads contending for the same address serialize? 〔Added〕
- What prerequisites must be met for grid-level synchronization with cooperative groups? 〔Added〕
- What is predication? When does the compiler rewrite a branch as predicated execution? 〔Added〕

## 2. Memory Hierarchy and Memory Access Optimization

- ★ Explain the capacity, bandwidth, and latency magnitudes of registers, shared memory / L1, L2, and HBM. Roughly how much do GMEM and SMEM latencies differ? 〔NV〕〔Meituan〕
- ★ What is memory coalescing? Which access patterns break coalescing? 〔Alibaba〕〔NV〕
  - Follow-up: When a warp reads 32 consecutive floats, how many sector requests are generated? What about strided reads?
- ★ What is a bank conflict? In what scenarios does it occur? How do you locate it? How many fixes can you think of? 〔NV〕〔Alibaba〕〔CN〕〔Bank〕
  - Follow-up: What is the bank granularity? How many banks are there?
  - Follow-up: If multiple threads in the same warp read the same address, is that a conflict?
  - Follow-up: How are conflicts counted for 64-bit or 128-bit accesses?
  - Follow-up: What are the costs of padding vs. swizzle?
  - Follow-up: Which ncu metric shows bank conflicts?
- ★ Why is reading/writing global memory with `float4` faster? What are the alignment requirements? 〔HPC〕
- What is local memory? When is it used? Where does it physically live? 〔NV〕
- How does register spilling happen? How do you detect and avoid it? What does `__launch_bounds__` do? 〔HPC〕
- Why does a CPU need three levels of cache while a GPU gets by with two? 〔Alibaba〕
- How large is L2 on common cards? What reuse can L2 handle for you? When does manually caching in shared memory become over-engineering? 〔HPC〕〔Overseas〕
- L1 and shared memory share the same SRAM — how is the split configured? 〔Added〕
- What access patterns suit constant memory and the read-only cache? 〔Added〕
- Why does `cudaMalloc` take a double pointer? 〔CN〕
- What techniques can you think of to improve the memory efficiency of a CUDA program? Answer systematically. 〔CN〕
- What techniques can you think of to improve the compute efficiency of a CUDA program? 〔CN〕
- For matrix transpose, how do you make both reads and writes coalesced without bank conflicts? 〔ByteDance〕〔HPC〕
- What is the principle and implementation of double buffering (ping-pong buffer)? 〔Bank〕

## 3. Performance Analysis Methodology

- ★ What is the roofline model? How do you compute a kernel's arithmetic intensity and decide whether it is memory-bound or compute-bound? 〔Bank〕
  - Follow-up: Which side of the roofline do GEMM, softmax, RMSNorm, and decode attention each fall on?
- ★ What determines occupancy? How do you control it? 〔HPC〕
  - Follow-up: Is higher occupancy always better? When is low occupancy actually faster?
  - Follow-up: If achieved occupancy is far below theoretical occupancy, what could the causes be?
- How is Little's Law applied on GPUs? How many in-flight memory accesses are needed to saturate bandwidth? 〔HPC〕
- Where does instruction-level parallelism (ILP) show up on NVIDIA GPUs? Why does having one thread compute multiple outputs speed things up? 〔HPC〕
- ★ How do you prove an operator optimization is actually effective? Should you account for differing peak throughput across precisions? What metric do you use? 〔Meituan〕
- ★ Given a slow kernel, what is your profiling workflow? What do you look at first, and next? 〔NV〕〔Added〕
- What questions do nsys and ncu each answer? 〔Added〕
- Which ncu sections and metrics do you look at most? What do common stall reasons (e.g. long scoreboard, MIO throttle) indicate? 〔CN〕
- What are wave quantization and the tail effect? How do you mitigate them? 〔Added〕
- ★ Why does kernel fusion speed things up? Besides saving memory traffic, what other benefits are there? Give examples. 〔NV〕〔Bank〕
- What should you watch out for when writing a kernel to measure chip FLOPS? 〔NV〕
  - Follow-up: How do you keep the compiler from optimizing the computation away? How many FLOPs is an FMA? How large should the grid and iteration count be? Should Tensor Cores be considered?
- How do you write a memory-bandwidth microbenchmark? 〔Added〕
- What is a general methodology for operator optimization? 〔ByteDance〕
- How do you debug CUDA programs? What tools are there (compute-sanitizer, cuda-gdb, etc.)? 〔HPC〕

## 4. Asynchrony, Streams, and Host-Side Orchestration

- ★ Is a kernel launch synchronous or asynchronous with respect to the host? Which APIs introduce implicit synchronization? 〔Added〕
- ★ Roughly how large is launch overhead? What does launch-bound mean? How do you fix it? 〔NV〕
  - Follow-up: Why does CUDA Graph reduce launch overhead? What restrictions does it impose on dynamic shapes and addresses?
- What ordering is guaranteed within a single stream versus across streams? How do you express cross-stream dependencies? 〔Added〕
- What is the difference between the legacy default stream and the per-thread default stream? 〔Added〕
- ★ What is pinned memory? Why does DMA need it? What is the cost of allocating too much? 〔CN〕
  - Follow-up: In PyTorch, why should `pin_memory=True` and `non_blocking=True` be used together?
- What is a copy engine? What is the hardware prerequisite for overlapping compute and copies? 〔Added〕
- What is the difference between `cudaMemcpyAsync` and Ampere's `cp.async`? 〔Added〕
- How do you use multiple streams to pipeline H2D, compute, and D2H? What bounds the speedup? 〔Added〕
- What can CUDA events do? What are the pitfalls of using them for timing? 〔Added〕
- What can cause gaps (idle bubbles) on the GPU timeline? How do you rule them out one by one? 〔Added〕
- When multiple processes share one GPU, what is the difference between MPS and MIG? 〔Added〕

## 5. GEMM and Tensor Cores

**From naive to register tiling**

- ★ Why is naive GEMM slow? Going from naive to shared-memory tiling to register tiling, what bottleneck does each step solve? 〔NV〕〔Kuaishou〕〔Meituan〕〔Xiaohongshu〕
- ★ What constrains the tile size? 〔HPC〕
- Why have one thread compute 8×8 outputs? Will registers spill? How many computations happen within one tile? 〔HPC〕
- What is the impact on the kernel when K ≫ M, N? What problem does split-K solve? 〔HPC〕〔Added〕
- What does `#pragma unroll` do? 〔HPC〕
- How do you optimize transposes inside a matmul? Why is A often stored transposed in shared memory? 〔ByteDance〕
- Where does your GEMM fall short of cuBLAS and CUTLASS? 〔HPC〕

**Tensor Cores**

- ★ What are the ways to use Tensor Cores? What is the difference between WMMA and MMA (`mma.sync`)? 〔Alibaba〕
- What does `ldmatrix` do? Why is it needed? 〔Added〕
- How does a multi-stage pipeline hide global-to-shared latency? 〔Added〕
- Have you done low-bit (int8 / FP8 / FP4) GEMM? At which step does dequantization happen? 〔HPC〕
- What are persistent kernels and Stream-K? 〔Added〕

**Hopper / Blackwell**

- ★ What are the advantages of Hopper's TMA? How is it invoked? Does data pass through L1? 〔CN〕
- ★ What is the difference between WGMMA and MMA? 〔CN〕
- ★ What is warp specialization? How are producer and consumer warps divided? 〔CN〕
- What is swizzle? Compared with padding, what are the pros and cons of each for resolving bank conflicts? 〔CN〕
- What are thread block clusters and distributed shared memory? 〔Added〕
- What did Blackwell's tcgen05 and TMEM change relative to Hopper? 〔Added〕

## 6. Attention and DL Operators

**Softmax and normalization**

- ★ Why subtract the max before softmax? How many passes does safe softmax need? 〔Alibaba〕〔CN〕
- ★ How does online softmax reduce it to a single pass? How are two partial results merged? 〔CN〕
- For softmax along the K dimension of an M×K matrix, how do you avoid overflow and precision loss? How do you tile when K is large? 〔Alibaba〕
- How do LayerNorm and RMSNorm differ computationally? How do you derive the RMSNorm backward pass? 〔CN〕〔Added〕
- How do you compute variance in a numerically stable way (Welford)? 〔Meituan〕
- How do you write fused activation kernels such as SwiGLU or GeLU+bias? What does fusion save? 〔Bank〕

**FlashAttention**

- ★ What is the core idea of FlashAttention? Why does it both save memory and run faster? 〔CN〕
- ★ What changed from FA1 to FA2? Why is FA2's outer loop over Q? 〔CN〕
- Which Hopper features does FA3 exploit? What does FA4 do on Blackwell? 〔Added〕
- Why does the FA backward pass recompute? Which statistics does the forward pass save? 〔Added〕
- How does FA save compute under a causal mask? What if the mask is random? 〔Bank〕
- What problem does Flash Decoding solve? Roughly what share of time does the combine kernel take? 〔CN〕
- What are the differences among MHA, MQA, GQA, and MLA? How do they affect KV cache size and TP sharding? 〔Added〕
- How do you estimate the compute-to-memory ratio of MLA decode? Does it depend on seqlen and batch size? 〔CN〕
- How does the PagedAttention kernel handle non-contiguous KV blocks? 〔Baidu〕

**Quantization, MoE, and correctness**

- How do symmetric and asymmetric quantization differ in computation? How do you choose among per-tensor, per-channel, and per-group? 〔Alibaba〕
- After quantization, how do you trade off performance and accuracy? 〔CN〕
- Implement an MoE layer and spell out the dimensions at each step. How are dispatch and combine done? 〔NV〕
- How are top-k and argmax parallelized on GPU? 〔Added〕
- After writing a forward or backward kernel, how do you verify numerical correctness? How do you set tolerances? 〔Overseas〕

## 7. DL Frameworks and Core Libraries

**PyTorch internals**

- What is PyTorch's dispatch mechanism? What are ATen, c10, and torch.library each responsible for? 〔Bank〕
- ★ How do you implement a custom CUDA operator and call it from PyTorch? What are the full steps? 〔Bank〕
- ★ What strategy does PyTorch's CUDA Caching Allocator use? How do you analyze and reduce memory fragmentation? 〔Bank〕
- Why does activation checkpointing save memory? Where is the extra compute? 〔Bank〕
- What is the difference between `torch.jit.trace` and `torch.jit.script`? When does each fail? 〔Bank〕
- What are the components of `torch.compile`? What is a graph break? 〔Added〕
- How does autograd build the graph and execute the backward pass? 〔Added〕
- What is the difference between `torch.repeat` and `torch.expand`? 〔CN〕
- What are the differences among contiguous, stride, view, and reshape? 〔Added〕
- What launch arguments does torchrun take? How do you kill all torchrun processes in bulk on Linux? 〔CN〕
- Design a GPU memory management mechanism. 〔Ant〕

**Core libraries and compilers**

- PyTorch is row-major and cuBLAS is column-major — how do you call cuBLAS correctly? 〔Added〕
- What is the difference between cuBLASLt and cuBLAS? What is epilogue fusion? 〔Added〕
- What is cuDNN's graph API / frontend? How does it select algorithms for convolution or attention? 〔Added〕
- What is CUTLASS's layered structure? What problem does CuTe's layout abstraction solve? 〔Overseas〕〔Added〕
- What are the pros and cons of Triton vs. CUDA? When should you use Triton? 〔Added〕
- In mixed-precision training, what problem does loss scaling solve? How do BF16 and FP16 differ? How is scaling done for FP8 training? 〔Added〕
- After a framework-level change, how do you guarantee numerics match the original? How do you ensure determinism? 〔Added〕
- If a library must support multiple GPU architectures, how are build and test organized? How do fatbin and PTX JIT relate? 〔Added〕

## 8. Distributed Training and Collective Communication

**Parallelism strategies**

- ★ What are the differences among DP, TP, and PP? How do you choose the ratio among them? 〔Kuaishou〕
- How does Megatron's TP shard the MLP and attention? How many all-reduces are there in the forward and backward pass of one transformer layer? 〔Kuaishou〕〔Baidu〕
- ★ How are gradients computed under pipeline parallelism? What is the difference between GPipe and 1F1B? How is the bubble ratio computed? What does interleaved 1F1B improve? 〔Kuaishou〕
- What problems do sequence parallelism and context parallelism (ring attention) each solve? 〔Added〕
- ★ What do ZeRO-1, ZeRO-2, and ZeRO-3 each shard? What is the communication volume of each? Why does ZeRO-3 add an extra copy? 〔Added〕
- What is the relationship between FSDP and ZeRO-3? 〔Added〕
- In MoE expert parallelism, what problems does all-to-all communication have? How are they solved? 〔Meituan〕
- How do you split a matmul too large for one node across multiple nodes? How do they communicate? 〔NV〕
- Walk through the overall flow of distributed training. 〔Kuaishou〕

**Collective communication**

- ★ What does all-reduce do? How do you derive the number of steps and per-GPU communication volume of ring all-reduce? 〔Baidu〕
  - Follow-up: What message sizes suit ring vs. tree? How does NCCL choose an algorithm?
- Why is reduce-scatter + all-gather equivalent to all-reduce? 〔Added〕
- How does DDP work? How do buckets and gradient hooks overlap communication with backward? What is the difference between NCCL and Horovod? 〔Tongyi〕
- What communication optimizations does Megatron-LM make? 〔CN〕〔Kuaishou〕
- How do you tell whether communication has been hidden behind compute? 〔Added〕
- Roughly what are the bandwidth magnitudes of NVLink, NVSwitch, InfiniBand, and PCIe? How does topology affect parallelism strategy? 〔Added〕
- What is SHARP? How does hierarchical all-reduce work? 〔Added〕

**Stability and RL**

- How is fault tolerance handled in large-scale training? How is asynchronous checkpointing done? How do you locate straggler nodes? 〔Added〕
- What are the steps of RL training? Where is the bottleneck in the sampling (rollout) stage? 〔CN〕

## 9. Memory, Compute, and Training-Time Estimation

- ★ For a 7B model trained with Adam in mixed precision, how much memory do parameters, gradients, and optimizer states each take? 〔Kuaishou〕〔Tencent〕
- ★ How do you estimate activation memory? How does it relate to seq len, batch, and hidden size? How much does recomputation save? 〔Added〕
- ★ How much memory does the KV cache take during inference? How do GQA and MLA affect it? 〔Added〕
- ★ Given the number of training tokens, how do you estimate training time? 〔CN〕
- ★ What is the difference between MFU and HFU? How is MFU computed? Which peak is used as the denominator? 〔Added〕
- Where are the parameters in a Transformer? Which part has the most parameters and which has the most compute, and why? 〔CN〕
- How are the FLOPs of a GEMM computed? How does attention compute scale with seq len? 〔Added〕
- What are the compute-to-memory ratios of prefill and decode? Why is decode memory-bound? 〔Added〕
- How does batch size affect training convergence, generalization, and throughput? 〔NV〕
- What are the memory capacity and bandwidth of common GPUs? 〔CN〕

## 10. Inference Optimization

- ★ How does the KV cache work? What are common optimizations (prefix cache, compression, quantization, offload)? 〔Kuaishou〕〔ByteDance〕
- ★ What is PD (prefill-decode) disaggregation? Multi-node PD disaggregation incurs KV cache transfer overhead — why do it anyway? 〔Meituan〕〔CN〕
- How is Mooncake's KV-cache-centric architecture designed? 〔CN〕
- ★ What problem does PagedAttention solve? 〔Baidu〕
- How do continuous batching and chunked prefill work? 〔Baidu〕
- How is CUDA Graph used in inference engines? 〔Baidu〕
- What optimization techniques apply to the prefill and decode stages respectively? 〔CN〕
- What is two-batch overlap? When does it become a de-optimization? 〔CN〕
- How does speculative decoding work? What affects the acceptance rate? 〔Added〕
- What communication problems does MoE inference have? How are they solved? 〔Meituan〕
- Which DeepSeek Open Source Week projects (e.g. FlashMLA, DeepEP, DeepGEMM) do you know? 〔Meituan〕
- What is the core design of the vLLM or SGLang scheduler? 〔Added〕
- How is inference-deployment traffic scheduled? 〔Kuaishou〕
- How do DiT inference frameworks and LLM inference frameworks compare in design? 〔CN〕
- What is the difference between diffusion language models (dLLM) and autoregressive models? 〔CN〕

## 11. C++ and Systems Fundamentals

**C++**

- ★ How are polymorphism, virtual functions, and vtables implemented? 〔NV〕
- C++ templates: specialization, SFINAE. Should template definitions go in the header or the .cpp, and why? 〔NV〕
- ★ RAII and smart pointers: how do you pass a `unique_ptr` to a function? Where is `shared_ptr` mainly used? What does `std::move` do underneath? 〔NV〕
- What is the difference between `malloc` and `new`? What ways of managing memory exist in C++? 〔CN〕
- What is the difference between `vector` and `list`? How do you efficiently delete elements from a `vector`? 〔NV〕
- How do you check whether two floats are equal? 〔CN〕
- Why doesn't floating-point arithmetic satisfy associativity? What is catastrophic cancellation? What problem does Kahan summation solve? 〔NV〕〔Added〕
- Why do C++ projects need header files? 〔CN〕
- How are arguments passed in x86 function calls? 〔Alibaba〕

**Systems and concurrency**

- What thread synchronization mechanisms do you know? Have you written a thread pool? 〔NV〕
- On a 16-core multithreaded processor where every core frequently calls the same handler, how do you keep it from becoming a bottleneck? 〔NV〕
- What is copy-on-write? How is a computer system layered? 〔NV〕
- What is the difference between big-endian and little-endian? How do you implement a ring buffer? 〔NV〕
- What is the basic usage of OpenMP? 〔NV〕
- In Docker, how do image, container, and Dockerfile relate? 〔NV〕

## 12. HPC Theory and Open-Ended Design Questions

Interviewers often bring a few open-ended questions. First narrow the scope, then pick the level of
abstraction (algorithm, system, or micro-optimization).

- ★ What do Amdahl's Law and Gustafson's Law each say? How are strong vs. weak scaling, speedup, and parallel efficiency computed? 〔NV〕
- How would you parallelize a compression algorithm based on hash-table matching? How do you trade off parameters like hash-table size and match length? 〔NV〕
- How do you parallelize graph algorithms? What data structure represents the graph? Do atomics count as true parallelism? Can synchronization be removed? Is the goal throughput or latency? 〔NV〕
- How do you parallelize pairwise IoU computation across a large number of bounding boxes? What algorithmic optimizations are possible? 〔NV〕
- How do you optimize the power method for eigenvalues? How should the mat-vec kernel be written? 〔Overseas〕
- How is parallel sorting (bitonic, radix) done? 〔NV〕
- For scatter-add by ID (histograms, embedding pooling), how do you trade off atomic contention vs. privatization? 〔CN〕
- Design a 10,000-GPU distributed training system. How do you handle communication bottlenecks? 〔Tongyi〕
- Given hardware parameters and a workload, estimate its runtime (performance modeling). 〔Overseas〕
- Brain teasers: measure a target amount of water with two containers; find the poisoned bottle with a few mice; river-crossing puzzles. 〔NV〕

## 13. Project Deep-Dive Templates

Almost every interview opens with a résumé deep-dive. Follow-up chains to prepare for any training-
performance project:

- How is your MFU defined and measured? Which peak is the denominator?
- What was the bottleneck before optimization? How did you locate it (nsys timeline, ncu, PyTorch Profiler)?
- If there was host wait: was it slow CPU launch, sync points, or data loading? How do you prove which one?
- How much did each optimization contribute? Why did you do them in that order?
- After integrating a new kernel, how did you prove numerical correctness and a real end-to-end gain?
- How was the parallel configuration chosen? What share of time is communication, and how is it hidden?
- How would the plan change at 2× the GPUs, or on the next GPU generation?
- Given two more weeks, where would the next bottleneck be?
- Tell the same project in three lengths: 30 seconds, 3 minutes, and 15 minutes.

## 14. Hand-Coding Problem List and Follow-Ups

Interviews usually provide no runtime environment: write the kernel body, grid / block config and the
launch, then explain how to optimize further.

| Problem | Source | Common follow-ups |
|---|---|---|
| vector add | 〔Baidu〕 | Why the bounds check; why not use shared memory; relationship between block=256 and warps |
| Fused activations like GeLU + bias | 〔Bank〕 | Vectorized loads/stores; how much memory traffic fusion saves |
| 4D elementwise multiply with broadcasting | 〔CN〕 | Stride computation under broadcasting; reducing indexing overhead |
| reduce sum / max | 〔NV〕〔Baidu〕 | Bottleneck of each version; warp shuffle; atomic vs. second kernel for the final step |
| Find the single distinct element in an array | 〔NV〕 | XOR reduction |
| Variance | 〔Meituan〕 | Parallel merge in Welford |
| all-reduce | 〔Baidu〕〔Alibaba〕 | Ring step count and communication volume |
| Matrix transpose | 〔ByteDance〕〔HPC〕 | Coalesced reads and writes; padding vs. swizzle |
| Prefix sum (scan) | 〔CN〕〔HPC〕 | Hillis-Steele vs. Blelloch; stitching across blocks |
| softmax (row-wise, or along K) | 〔Alibaba〕〔CN〕 | Safe vs. online; tiling when K is large |
| RMSNorm / LayerNorm | 〔CN〕 | Backward derivation; FP16 overflow |
| top-k / argmax | 〔Added〕 | Warp-level top-k |
| int8 per-channel quantization | 〔Added〕 | Computing and storing scales |
| Chained (1,256)×(256,128)×(128,256) | 〔CN〕 | Association order; essentially two GEMVs |
| GEMM: naive → tiled → register tiling | 〔NV〕〔Kuaishou〕〔Meituan〕〔Xiaohongshu〕 | See Section 5 |
| Tensor Core GEMM | 〔Alibaba〕 | WMMA vs. MMA; ldmatrix layout |
| FlashAttention forward (in CUDA) | 〔Kuaishou〕 | Why the outer loop is over Q; skipping blocks under causal mask |
| Histogram / embedding pooling by ID | 〔CN〕 | Atomic contention; privatization; sort then segmented reduce |
| GEMV / mat-vec | 〔Overseas〕 | Whether L2 already handles reuse; fusion with normalize |
| Vector outer product | 〔HPC〕 | Write-back bandwidth is the bottleneck |
| FLOPS microbenchmark | 〔NV〕 | Preventing compiler elimination; grid and iteration count; Tensor Cores |
| log on odd columns, cos on even columns | 〔NV〕 | Trade-off between divergence and coalescing |
| Image rotation | 〔NV〕 | 2D indexing; boundaries; coalesced access |
| MLP forward, backward + SGD | 〔Tongyi〕〔Overseas〕 | Derive gradients by hand |
| Implement an MoE layer with dimensions | 〔NV〕 | Dispatch and combine |
| C++ memory pool (supporting new[]/delete[]) | 〔CN〕 | Alignment; where the array element count is stored |
| C++ recursive tensor stride computation | 〔NV〕 | Non-contiguous tensors |

---

## Sources

Questions were rewritten from public interview reports and question banks, including: NVIDIA reports on
1point3acres, Nowcoder and Glassdoor; AI-infra interview roundups on Nowcoder and Zhihu; Anthropic's
public performance take-home (github.com/anthropics/original_performance_takehome) and its blog post on
AI-resistant technical evaluations; and the repositories xiaoba17/ai-infra-interview-305,
Tongkaio/CUDA_Kernel_Samples, xlite-dev/LeetCUDA and Imironman123/llm-interview-notes.

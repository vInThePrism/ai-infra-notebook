# AI Infra Notebook

**First-principles notes on attention, distributed training, and GPU performance models.**

[Reading paths](#reading-paths) · [Explore by question](#explore-by-question) · [Try a model](#try-a-model) · [Practice explaining](#practice-explaining)

How does a mathematical operation become a fast training system? These notes follow the work from attention and MoE layers to memory traffic, communication, and hardware bottlenecks. Each explanation starts with a question, works through the mechanism, and connects it to code or a concrete example.

## Reading paths

| Goal | Suggested path |
|---|---|
| Build an intuition for attention | Soft database lookup → FlashAttention overview → FA1 / FA2 / FA3 deep dives → ring attention |
| Understand distributed training | MoE → TP/SP/CP → gradient and parameter communication |
| Reason about performance | Five principles → optimization methodology → attention cycle model → measurement case studies |

For CUDA indexing and synchronization, begin with the companion [GPU Kernel Lab](https://github.com/vInThePrism/gpu-kernel-lab).

## Explore by question

| Question | Note |
|---|---|
| Which ideas keep recurring in performance work? | [Five principles](notes/00-principles.md) |
| How do you derive a performance design from first principles? | [Optimization methodology](notes/01-first-principles-optimization-methodology.md) |
| Why cache K and V, but not Q? | [Attention as a soft database lookup](notes/attention-as-a-soft-database-lookup.md) |
| Why do attention kernels change across GPU generations? | [FlashAttention 1–4](notes/02-flashattention-1-to-4.md) |
| Why is attention memory-bound, and how do tiling and online softmax fix it? | [FlashAttention-1](notes/flashattention/01-flashattention-1.md) |
| Why does FA2 put Q in the outer loop, and what else makes it ~2× faster? | [FlashAttention-2](notes/flashattention/02-flashattention-2.md) |
| How does FA3 use Hopper's asynchronous hardware to hide softmax? | [FlashAttention-3](notes/flashattention/03-flashattention-3.md) |
| How does FA4 handle the exponential bottleneck on Blackwell? | [FlashAttention-4](notes/flashattention/04-flashattention-4.md) (in progress) |
| How do uneven expert workloads become grouped GEMMs? | [MoE, grouped GEMM, and expert parallelism](notes/03-moe-grouped-gemm-and-expert-parallelism.md) |
| Why does context parallelism move K/V? | [TP, SP, CP, and ring attention](notes/04-tp-sp-cp-and-ring-attention.md) |
| What actually communicates during training? | [Distributed training communication](notes/05-distributed-training-communication.md) |
| Where do data loading and recovery enter the system? | [Data pipelines and checkpointing](notes/09-data-pipeline-and-checkpointing.md) |

## Try a model

The [attention cycle model](tools/fa_cycle_model/) estimates per-tile demand on matrix multiplication, exponentials, scalar arithmetic, and shared memory.

```bash
git clone https://github.com/vInThePrism/ai-infra-notebook.git
cd ai-infra-notebook
python tools/fa_cycle_model/cycle_model.py --gpu H100 B200 --sweep
```

This command uses Python's standard library and needs no GPU. It prints estimated cycle counts, a modeled bottleneck, and overlap bounds. These are analytical estimates, not measured kernel latency. The accompanying polynomial-exp2 experiment additionally requires NumPy.

## Practice explaining

The [question bank](interview-prep/question-bank.md) and [practice prompts](interview-prep/) support a simple loop:

**Explain aloud → correct the explanation → retrieve it without hints → revisit it later.**

The prompts cover a voice mock interview, a recall worksheet, and a narrated explanation. They are reusable text templates, not an automated worksheet or video application. The question bank includes source tags and intentionally leaves answers open for the reader.

## From concepts to evidence

- [GPU Kernel Lab](https://github.com/vInThePrism/gpu-kernel-lab): implement and inspect an operator.
- [Logical MFU](https://github.com/vInThePrism/logical-mfu): understand the assumptions behind FLOPs accounting.
- [Training Performance Field Notes](https://github.com/vInThePrism/training-perf-field-notes): follow a hypothesis through a training experiment, including negative results.

Hardware and framework details are version-dependent. Read assumptions alongside conclusions; distinguish paper results, analytical models, and personal experiments.

## Contributing and reuse

Useful contributions clarify a derivation, identify a counterexample, or connect a claim to a primary source. Keep one main question per note and explain the conditions under which the answer holds.

Code is MIT-licensed. Notes and practice material are CC BY 4.0; preserve attribution when adapting them.

## Continue learning and contributing

See [where new work belongs](docs/content-routing.md), [contribution guidance](CONTRIBUTING.md), [release validation](VALIDATION.md), and the [license and source map](LICENSES.md).

# Data pipelines and checkpoint recovery at scale

A training system must coordinate data preparation, host-to-device transfer, compute, and durable recovery. Slow steps can arise anywhere in this pipeline.

## Data preparation and transfer

Measure remote reads, CPU preprocessing, packing, and host-to-device copies separately. Initialize thread pools inside each worker process when using a fork-based loader: parent threads do not survive the fork. Bound look-ahead and pinned-memory use so that the loader does not exhaust host resources. Pinned memory and nonblocking copies enable asynchronous transfer; actual overlap depends on scheduling, stream dependencies, and workload.

Cold-start work deserves its own measurement: parallel shard parsing, streaming tokenization, and content-hash caches solve different costs. Hash both input data and preprocessing configuration so that changes invalidate stale cached results.

## Checkpoints as a recovery protocol

Separate a short foreground snapshot into staged host memory from background persistence. Budget staging memory alongside loader and pinned buffers. A background writer slower than the checkpoint interval creates backpressure.

A recovery manifest should bind model, optimizer, RNG state, data cursor, and packing buffers. Publish it atomically only after the constituent shards are durable. Fault injection and comparison of post-resume batch identities test whether recovery preserves the experiment. Saving only model weights cannot establish exact continuation.

## Evidence and measurements

The original data-loading and recovery measurements, including their workload-specific settings and numbers, are preserved in [the companion field-note supplement](https://github.com/vInThePrism/training-perf-field-notes/blob/main/notes/supplements/data-pipeline-and-recovery-evidence.md). Snapshot pause, final checkpoint drain, and total runtime are different quantities; see the [checkpoint investigation](https://github.com/vInThePrism/training-perf-field-notes/blob/main/notes/10-async-checkpoint-measurement.md).

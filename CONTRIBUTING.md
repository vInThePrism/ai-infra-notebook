# Contributing

Notes develop general mechanisms; tools/fa_cycle_model contains analytical CPU models; interview-prep contains reusable text prompts. The exp2 experiment requires NumPy. A cycle estimate is not a GPU benchmark.

Start with the question your change answers. Explain the mechanism and its limits, then link to the relevant implementation or primary source. Update an existing explanation when appropriate. See [content routing](docs/content-routing.md) for companion-repository boundaries.

For numerical or performance work, include input shapes, precision, reference, tolerance, device/toolchain, warm-up and timing boundaries. Separate kernel measurements from complete-job outcomes. A configuration flag alone does not establish that the intended implementation ran.

Use the existing validation path:

```bash
python tools/fa_cycle_model/cycle_model.py --gpu H100 B200 --sweep
```

State checks you could not run. GPU-only paths must retain that status until exercised on suitable hardware. Keep raw traces and private data out of contributions. No extra test suite or dependency is needed for a documentation correction.

Before sharing adapted material, retain the applicable source notice from [LICENSES.md](LICENSES.md). New prose in this release uses CC BY 4.0; new source edits retain the MIT terms of the code they modify.

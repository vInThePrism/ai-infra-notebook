# Release validation

Validated on 2026-10-21 on macOS arm64 with Python 3.14.0. The isolated validation environment used pytest 9.1.1, NumPy 2.5.4, and PyTorch 2.14.1 where needed. These are the versions exercised for this release, not a compatibility matrix.

Original private datasets, raw cluster traces, and historical run logs are not bundled. Historical measurements in notes were not independently reproduced in this release.

## Checks completed

- `python tools/fa_cycle_model/cycle_model.py --gpu H100 B200 --sweep` executes on CPU.
- `python tools/fa_cycle_model/exp2_emulation.py --degree 3` executes with NumPy: sampled max relative fp32 error **1.154e-4**, BF16 identical **99.00%**, within 1 ULP **100.00%** for the script's seeded sample.

## Boundaries

Cycle counts are analytical estimates under the inputs and assumptions in the code. They are not measured H100/B200 timings. The prompts are text templates; no PDF/video generation pipeline or live interview service is implemented here.

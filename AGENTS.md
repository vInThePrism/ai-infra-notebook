# Working in ai-infra-notebook

Purpose: AI infrastructure concepts and performance models.

Notes develop general mechanisms; tools/fa_cycle_model contains analytical CPU models; interview-prep contains reusable text prompts. The exp2 experiment requires NumPy. A cycle estimate is not a GPU benchmark.

## Commands

Run from the repository root unless a command says otherwise.

```bash
python tools/fa_cycle_model/cycle_model.py --gpu H100 B200 --sweep
```

## Editing rules

- Keep changes focused on the request. Do not add test files, test dependencies, or scaffolding unless requested or required by an existing build. Run relevant existing checks.
- Keep README commands and relative links aligned with the actual files.
- Report measured, estimated, source-reported, and unverified results accurately. Include environment and assumptions with performance claims.
- Preserve source copyright and license provenance; consult LICENSES.md before moving or adapting material.
- Never commit credentials, private paths, datasets, raw internal logs, or generated caches.
- Use docs/content-routing.md to keep the four companion repositories distinct.

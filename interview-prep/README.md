# A learning system for AI-infra interviews (in a second language)

I prepare for AI-infrastructure / CUDA interviews in English, which is not my first language.
Studying in English slows me down enough that I actually understand things. This folder is the system
I built around that, plus the question bank it runs on.

## The loop

```
question bank ──► voice mock interview ──► improved answer + key terms + 1-line summary
                                               │
             ┌─────────────────────────────────┴────────────────────────────┐
             ▼                                                              ▼
 fill-in-the-blank worksheet (PDF)                       first-principles narrated video
 cued recall → free recall → answer key                  built only from MY improved answers,
 spaced review: Day 1 / 3 / 7 / 14                       in MY logic, ordered from one root fact
```

1. **Mock interview** ([`mock-interview-prompt.md`](mock-interview-prompt.md)). One question at a time,
   answered out loud. Feedback comes in a fixed order: *improved version* of my own answer (my ideas,
   corrected, in short sentences I can repeat) → *key English terms and expressions* → *the question +
   a one-line summary to memorize*.
2. **Worksheet** ([`worksheet-prompt.md`](worksheet-prompt.md)). The one-line summaries become a PDF:
   blanks on the words that carry the answer (terms, numbers, causal phrases), then the same questions
   in a different order with no hints (free recall, closer to a real interview), then the answer key.
   Fill-in alone is cued recall; free recall is what an interview tests.
3. **Video** ([`narrated-video-prompt.md`](narrated-video-prompt.md)). The improved answers from a session
   are re-ordered from one first principle (e.g. "fast memory is small; big memory is slow") into a
   narrated slide video I can listen to and repeat. Rule: only my answers and my structure, never the
   interviewer's model answers. If I can't say it, it doesn't go in.

The notes in [`../notes`](../notes) are the written result of this loop.

## Files

- [`question-bank.md`](question-bank.md): ~200 conceptual questions in 13 sections plus 27 hand-coding
  problems, each tagged with where it came from (NVIDIA reports, Chinese AI-infra interviews, HPC
  companies, curated banks, overseas JDs). ★ = reported repeatedly.
- [`mock-interview-prompt.md`](mock-interview-prompt.md), [`worksheet-prompt.md`](worksheet-prompt.md),
  [`narrated-video-prompt.md`](narrated-video-prompt.md): the prompts, usable with any capable LLM.

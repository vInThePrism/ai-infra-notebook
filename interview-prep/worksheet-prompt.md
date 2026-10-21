# Mock interview → fill-in-the-blank worksheet (PDF)

Use this after a session run with `mock-interview-prompt.md`. Turn the session into a printable practice PDF.

## Input

- The mock interview (or a transcript).
- For each question, the **Question + summary** block from the feedback. If a question has no summary
  yet, write one in the same style: one or two lines, simple English.
- Optional: a "weak list" of questions missed in earlier sessions (see Part 4).

## Output

One PDF named `mock-YYYY-MM-DD.pdf`, English only, selectable text, letter size, space to write by hand.
Never split one question across two pages.

### Header (page 1)

- Date and number of questions.
- Score boxes: `Fill-in: __ / __ blanks` and `Free recall: __ / __ questions`.
- Review checkboxes: `□ Day 1  □ Day 3  □ Day 7  □ Day 14`.

### Part 1 — Fill in the blanks

For each question: number, section tag from the question bank (e.g. `§2 Memory`), the full question,
and the summary with key parts replaced by numbered blanks: `______ (1)`.

Rules for blanks:

1. Blank 2–4 parts per summary; never more than about 40% of the words.
2. Blank what carries the answer: key terms (`bank conflict`, `grid-stride loop`), numbers and limits
   (`32 banks`, `128 or 256`), and the key logic as a short phrase (`hides latency by ______`).
3. Do not blank filler words, or words that grammar alone gives away.
4. All blanks have the same length, so the length gives no hint.
5. Keep every other word exactly the same as the summary.

### Part 2 — Free recall

The same questions **in a different order**, with no summary and no blanks, five empty lines each.
Instruction: *"Answer out loud in English first (30–60 seconds). Then write only the keywords."*

### Part 3 — Answer key (new page)

For each question: the full summary with blanked parts in **bold**; one line `(1) … (2) … (3) …` with
`also OK: …` alternatives; the key English terms (one short line).

### Part 4 — Review from earlier sessions (only with a weak list)

Add these at the end of Part 1 under "Review", with **different** blanks from last time; add them to
Parts 2 and 3 as well.

## Why this design

Fill-in-the-blank is *cued* recall: it checks recognition. An interview asks for *production* with no
cues, which is what Part 2 trains. Blanking causal phrases, not just nouns, tests whether you know
*why*, and the Day 1/3/7/14 boxes turn one session into spaced repetition.

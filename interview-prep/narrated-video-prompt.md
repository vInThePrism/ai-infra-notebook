# Mock interview → first-principles narrated slide video

Turns one mock-interview session into a 10–15 minute narrated slide video for listening practice.

## Step 1: Gather only my material

- Collect every question I actually answered.
- For each, take the **improved version of my own answer**. Do not use the interviewer's separate
  explanations or model answers.
- List the questions I did not answer at the end of the script, and ask whether to add them.
- Note my own structure (e.g. level by level: registers → shared memory / L1 → L2 → HBM). The video
  follows my logic, not a new one.

## Step 2: Write the script (show it before building anything)

- Start from **one first principle** that the whole topic follows from, and frame every section as an
  answer to it. Examples: "Fast memory is small; big memory is slow." "A kernel is only as fast as its
  slowest unit." "Count the work from the model, never from the kernels."
- Order sections by my logic: the principle → the map → one part per sub-topic → putting it together →
  a one-sentence closing line to remember.
- First person, simple spoken English, short sentences. Keep my own phrases.
- Rough timing per part; about 10–15 minutes total.

## Step 3: Slides

- One data file with a list of slides: `part`, `title`, `body` (HTML), `say` (narration).
- 15–20 slides, rendered with HTML/CSS at 1920×1080 and captured with a headless browser (Playwright).
- Dark background, one accent color for the main idea, one warning color for costs. Diagrams over bullet
  walls: hierarchy stacks, a row of 32 banks, load/compute timelines, ring diagrams.
- Make a contact sheet of all slides and check it for overflow before moving on.

## Step 4: Narration

- Any local TTS works (I used Kokoro, voice `am_michael`, speed 0.92, for clear, slightly slow speech).
- Spell technical terms out in the `say` text only, never on the slide: `cp.async` → "C P async",
  HBM → "H B M", A100 → "A one hundred", numbers as words.
- Short silences between paragraphs and at slide boundaries.

## Step 5: Assemble

- One segment per slide (still image + its audio) at a low frame rate, then concatenate:
  ```bash
  ffmpeg -y -loop 1 -framerate 5 -i sNN.png -i aNN.wav -c:v libx264 -tune stillimage \
    -pix_fmt yuv420p -r 5 -c:a aac -b:a 160k -ar 48000 -shortest vNN.mp4
  ffmpeg -y -f concat -safe 0 -i list.txt -c copy final.mp4
  ```
- Verify with `ffprobe` (h264 + aac), `volumedetect` (not silent), and one extracted frame at a known time.

## Step 6: Publishing (optional)

- Title: `Ace the AI INFRA Interview: <Topic> Explained from First Principles`.
- Description: the first principle in one line, a paragraph of searchable terms, chapters computed from
  the real audio length of each slide (first chapter `0:00`, each ≥ 10 s).
- Upload and sign-in are done by me. Never click Publish on my behalf.

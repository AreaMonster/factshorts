# factshorts — Project Spec

You are building **factshorts**: a local, 100% free tool that automatically produces
short animated fact videos (vertical 9:16) with spoken narration, from auto-generated
or user-provided ideas, with strong fact-checking against real sources. It is
controlled through a local Web UI and a CLI.

Read this whole spec before writing code. Work in the phases at the bottom and
stop after each phase to report.

---

## 1. Hard constraints

- **100% free.** No paid APIs, no API keys, no accounts, no cloud services.
  Allowed network use: Wikipedia, Wikidata and Wikimedia Commons APIs, plus
  one-time downloads (Ollama models, Piper voices, Hugging Face models, pip packages).
  Everything else runs locally.
- **Target OS:** Windows 11 first, must also run on Linux. Python 3.11+.
  No hardcoded paths, use `pathlib` everywhere.
- **The LLM is never the source of truth.** Every factual claim in a video must be
  traceable to a fetched source text and pass automated verification.
- **Metric units only** in all generated content (km, m, kg, °C, km/h).
  If a source uses imperial units, convert to metric.
- **No copyrighted media.** Only self-generated graphics, or Wikimedia Commons
  files with a compatible license (see 6.3).
- **Honesty:** never claim the output is guaranteed correct. The README must say that
  a local LLM plus verification reduces errors but cannot eliminate them,
  and that the review step is the recommended default.
- **Git:** if you commit, write normal commit messages with **no AI or Claude attribution
  lines** (no Co-Authored-By, no "Generated with").
- When unsure about a library's API (Piper, Ollama, MoviePy and others change often),
  check the installed version's docs or `--help` instead of guessing.

## 2. Architecture

```
src/factshorts/
  core/            # pipeline: ideas, sourcing, script, verify, tts, render, export
  sources/         # wikipedia.py, wikidata.py, commons.py, user_source.py
  verify/          # programmatic.py, nli.py, llm_judge.py, aggregate.py
  render/          # scenes, captions, themes, ffmpeg wrapper
  web/             # FastAPI app, templates, static (vendored JS/CSS)
  cli.py
  config.py        # loads config.toml, validated with pydantic
  db.py            # SQLite: ideas, jobs, videos, rejections, cache
tests/
assets/            # fonts (OFL-licensed), music/, themes/
config.toml
pyproject.toml
README.md
```

- One core library. The CLI and the Web UI are thin layers that call the same
  functions, so they behave identically.
- Every pipeline step is a pure-ish function with typed input and output
  (pydantic models), so it can be tested and re-run on its own.
- Each job has its own working folder `work/<job_id>/` with intermediate files
  (`source.json`, `script.json`, `verification.json`, `audio/`, `frames` or
  segments, `final.mp4`). A job can **resume** from the last completed step.
- Cache Wikipedia/Wikidata responses and TTS output (keyed by text, voice and settings)
  in SQLite or on disk.
- Logging: rotating log file plus console, and a per-job log shown in the UI.
- Retries with backoff on network errors. Clear error messages, no raw tracebacks
  in the UI.

## 3. Pipeline

### 3.1 Ideas

**Auto-generated**
- Topics come from a configurable list in `config.toml`
  (e.g. space, human body, animals, geography, history, technology, Switzerland).
- The Ollama LLM proposes N candidate ideas per topic as structured JSON
  (use Ollama's `format` JSON-schema / structured output).
- Each idea must map to a concrete, existing Wikipedia article. Resolve it via
  Wikipedia search, and discard ideas without a good match.
- Dedupe against all past ideas in SQLite: exact plus fuzzy match (normalized title,
  plus embedding or token similarity). Same article plus same angle counts as a duplicate.

**Custom (user-provided)** — first-class input
- Single idea, multiple ideas (one per line), or an `ideas.txt` file
  with optional per-line settings: `Why is Venus hotter than Mercury? | length=45 | tone=funny`.
- Optional own source: URL or pasted text or uploaded file (.txt/.md/.pdf).
- Source resolution order: user source → Wikipedia search → if nothing reliable,
  reject with a clear message asking for a source.
- **False premise handling:** if the idea contains a claim the source contradicts,
  do not silently fix it. Stop the job in state `needs_input`, show what the source
  actually says, and let me choose "make corrected version" or "cancel".
- Opinion or creative ideas without factual claims can run with fact-check **off**.
  Those outputs are clearly labeled `UNVERIFIED` (filename, metadata, UI badge).
- Idea inbox/queue: custom ideas wait in a queue. Batch runs process the inbox first,
  then fill up with auto-generated ideas.
- Dedupe applies to custom ideas too, with a `force` override.

### 3.2 Sourcing

- Fetch the Wikipedia article (REST/Action API, proper `User-Agent` with a contact
  placeholder from config). Store the plain text, URL, title and **revision ID**.
- Prefer claims from the lead and the first ~3 sections (best-maintained text).
  Configurable.
- Fetch matching Wikidata entity statements (numbers/dates such as population,
  height, length, area, mass, dates, distances) for cross-checking.
- For user-provided sources: extract text, store it with a hash. Mark these as
  "user source" in the metadata.

### 3.3 Script

- Generated by the LLM **only from the provided source text** (RAG prompt:
  "Use only facts stated in SOURCE. If unsure, leave it out. Do not add outside knowledge.").
- Structured JSON (validated with pydantic, retry on invalid JSON):
  - `title`, `hook` (first 2 s, must grab attention), `scenes[4–7]`, `outro`
  - per scene: `narration` (1–2 short sentences), `on_screen_text` (max ~8 words),
    `visual` (one of the scene types in 3.5), `claims[]`
  - per claim: `text`, `source_quote` (verbatim sentence from source), `type`
    (number/date/name/general)
- Length targets: 30/45/60 s → word budget at ~150 words per minute, enforced.
- Tone options: neutral, funny, dramatic, kid-friendly. Tone may change wording,
  never facts.
- Hooks must not be clickbait that misrepresents the facts.

### 3.4 Verification (the core feature)

A claim passes only if **all enabled checks pass**. Run in this order (cheap first):

1. **Quote check:** `source_quote` exists in the source text
   (normalized whitespace/quotes/dashes, small fuzzy tolerance for punctuation only).
2. **Entity check:** every number, year, date and proper noun in the narration appears
   in the cited quote or its paragraph. Normalize number formats
   (`8,849` / `8849` / `8 849` / "eight thousand…"), unit conversions
   (m↔km, kg↔t, °F→°C, mi→km) with rounding tolerance, and spelled-out numbers.
3. **Wikidata cross-check:** for numeric/date claims with a matching Wikidata
   property, values must agree within a tolerance. On disagreement, drop the claim.
4. **NLI check:** local entailment model on CPU (e.g. a DeBERTa-v3 NLI cross-encoder
   via `transformers`/`sentence-transformers`): premise = quote and paragraph,
   hypothesis = narration claim. Require "entailment" above a configurable threshold.
5. **LLM judge:** a second strict Ollama pass, claim vs. source →
   `SUPPORTED` / `NOT_SUPPORTED` / `UNCLEAR` with a short reason.
   Anything other than `SUPPORTED` fails.

Then:
- A failed claim triggers a targeted rewrite of that scene (max 3 retries).
  If it still fails, remove the scene if the video stays coherent, else reject the idea.
- Also check the whole narration for **unsupported sentences without claims**
  (the LLM sneaking facts into "non-claim" sentences). Every sentence is either
  linked to a claim or classified as non-factual filler by the checks.
- Save `verification.json` with each claim, each check's result, scores and reasons.
- Rejected ideas go to a `rejections` table with the exact reason
  (e.g. `"number 8849 m not found in source paragraph"`).
- The pipeline is configurable but **safe by default**: all checks on, review required.

### 3.5 Voice (Piper TTS)

- `piper-tts` Python package. Auto-download the selected voice (.onnx + .json)
  from the `rhasspy/piper-voices` Hugging Face repo on first use. Default to a good
  English voice (medium or high quality), switchable in settings.
- **Text normalization before TTS:** numbers, years, units, abbreviations and symbols
  become speakable text (e.g. `8,849 m` → "eight thousand eight hundred forty-nine metres",
  `°C` → "degrees Celsius", `km/h` → "kilometres per hour"). Keep a pronunciation
  override dictionary in config for names.
- Synthesize per sentence to get exact durations. Insert configurable pauses
  between sentences and scenes.
- Optional: run **faster-whisper** (local, CPU) on the final narration for word-level
  timestamps to sync captions exactly. Fall back to a proportional estimate by
  character count if it's disabled.

### 3.6 Animation and render

- 1080×1920, 30 fps, H.264 + AAC MP4. Use hardware encoding if available
  (auto-detect NVENC / AMF / QSV), fallback `libx264`.
- Frame generation: Pillow + numpy streamed into ffmpeg via a pipe, or MoviePy if
  clearly simpler. ffmpeg does the final encode either way. Render scenes in parallel
  worker processes, then concat.
- Visual system:
  - Animated backgrounds: gradients, particles, subtle noise, theme-based
  - Kinetic on-screen text: fade/scale/slide in, easing curves
  - **Word-by-word highlighted captions** in the vertical **safe zone**
    (avoid the bottom ~20 % and the right edge where TikTok/Shorts UI sits)
  - Scene types: `big_number` (animated count-up), `comparison` (two bars or objects
    scaled to each other), `timeline`, `map_pin` (simple self-drawn shapes),
    `icon_list`, `image` (Ken Burns pan/zoom on a Commons image), `quote`
  - Progress bar, subtle transitions between scenes, end card
- Themes as TOML/JSON files in `assets/themes/` (colors, fonts, animation speed).
  Use only OFL/free fonts bundled in `assets/fonts/`.
- **Wikimedia Commons images (optional, per job):** search Commons for the article's
  images. Allow only Public Domain, CC0, CC BY and CC BY-SA. Store author, license
  and URL, and put the attribution in the video description and `sources.json`.
  Skip images of living people.
- Audio: optional royalty-free background music from `assets/music/`, ducked under
  the voice (ffmpeg `sidechaincompress`). Normalize the final mix to about −14 LUFS
  (ffmpeg `loudnorm`).

### 3.7 Export

For every video, write next to the MP4:
- `sources.json`: article URL plus revision ID, Wikidata IDs, every claim with its quote
  and verification results, image attributions, verified/unverified flag
- `metadata.json` and `description.txt`: title, short description, hashtags,
  **source links and image credits** (ready to paste into TikTok/YouTube/Instagram)
- `thumbnail.jpg`: auto-generated from the best frame plus title text
- Optional: `.srt` subtitle file

Folder layout: `output/pending/`, `output/approved/`, `output/rejected/`.

## 4. Web UI

`factshorts ui` → local web interface at `http://127.0.0.1:8000`.

- Stack: **FastAPI + Jinja2 + HTMX**, Server-Sent Events for live progress.
  No Node build step, no CDN (vendor all JS/CSS locally so it works offline).
- Binds to localhost by default. `--host 0.0.0.0` for LAN access, **only allowed if
  a password is set** in config (simple session login, hashed password).
- Dark theme, clean, mobile-friendly.

**Pages**
1. **Create**
   - Text box for one or more custom ideas, optional source URL, text or file upload
   - Settings: length, tone, voice, theme, music on/off, Commons images on/off,
     fact-check on/off (off = labeled unverified)
   - "Generate ideas" button: pick topics and count, shows the proposed ideas with
     their Wikipedia article. Tick or untick, edit, then send to the queue.
2. **Queue / Jobs**
   - Background job queue (one render worker by default, configurable)
   - Live progress per job: idea → sourcing → script → verification → voice → render → export,
     with step logs, cancel, retry from failed step
   - `needs_input` jobs (false premise) show the source excerpt plus action buttons
   - Rejections with exact reasons
3. **Review**
   - Video player
   - Side panel: script by scene; each claim next to its source quote with
     per-check verdicts (quote / entities / Wikidata / NLI / LLM) and a link to the
     article revision
   - Actions: Approve, Reject, Regenerate scene, **Edit narration**
     (edits re-run verification → TTS → render for that scene only), change theme or voice
     and re-render
4. **Library**
   - Approved videos, search/filter by topic, date and verified status,
     download MP4 plus description, delete
5. **Settings**
   - Ollama URL plus model dropdown (fetched from Ollama), with a hint to use the largest
     model that fits my VRAM/RAM
   - Piper voice manager (list, download, preview sample)
   - Verification thresholds, topic list, defaults, output folder, password
   - **Health panel**: same checks as `factshorts doctor`

State (jobs, queue, settings) lives in SQLite, so a restart loses nothing and
unfinished jobs resume.

## 5. CLI

```
factshorts make --topic space --count 3
factshorts make --idea "Why is Venus hotter than Mercury?" [--length 45 --tone funny]
factshorts make --idea "..." --source-url https://... | --source-file notes.txt
factshorts make --ideas-file ideas.txt
factshorts make --idea "..." --no-factcheck        # labeled UNVERIFIED
factshorts queue add "idea" | queue list | queue clear
factshorts batch --per-topic 2                      # inbox first, then auto ideas
factshorts review                                   # terminal review fallback
factshorts rerender <job_id> [--theme X --voice Y]
factshorts ui [--host --port]
factshorts doctor
```

`doctor` checks: Python version, ffmpeg (and HW encoders), Ollama reachable, model
pulled, Piper voice present, NLI model downloadable/cached, write permissions, and
**fails if any config value looks like an API key or a non-local LLM endpoint**.

## 6. Config (`config.toml`)

All tunables in one place, with comments: Ollama URL and model, topics, default
length/tone/voice/theme, verification checks plus thresholds, retries, music volume,
LUFS target, encoder preference, output paths, Web UI host/port/password hash,
Wikipedia user agent, cache TTL.

## 7. Tests and quality

- `pytest`, runs **fully offline** with mocked LLM, TTS, Wikipedia, Wikidata and NLI.
- Unit tests:
  - number/unit/date normalization and conversion (many edge cases)
  - quote matching (curly quotes, dashes, whitespace)
  - entity check catching invented numbers and names
  - verification aggregation (one failing check → claim fails)
  - TTS text normalization
  - caption timing and safe-zone layout
  - dedupe logic
  - false-premise detection flow
- API tests for the Web UI endpoints (FastAPI TestClient).
- One **golden test**: a fixed source text plus a fixed mocked script renders a short
  video. Assert duration, resolution, audio stream present and audio/video length
  match within 100 ms (via `ffprobe`).
- Include a few **adversarial fixtures**: scripts with a wrong number, a wrong year,
  a swapped name and an unsupported extra sentence. All must be caught.
- Lint/format with `ruff`. Type hints everywhere.

## 8. README

- What it does, with an honest limitations section
- Windows setup step by step: Python, ffmpeg (`winget`), Ollama install plus model pull,
  first run of `doctor`, starting the UI
- How to pick an Ollama model based on GPU VRAM / RAM
- How fact-checking works (short diagram of the checks)
- How to add themes, music, voices and topics
- Optional: reaching the UI from a phone on the same Wi-Fi (password required),
  with a security warning
- Licensing notes: Wikipedia text is CC BY-SA, so videos must credit sources
  (the description file handles this)

## 9. Phases — work in this order, stop after each and report

- **Phase 0 – Plan:** summarize your understanding, list open questions and
  assumptions, propose the final file tree and key libraries with versions.
  Wait for my OK.
- **Phase 1 – Core vertical slice (CLI):** one custom idea → Wikipedia source →
  script → quote and entity checks → Piper → simple render → MP4 with sources.json.
  Produce one real sample video and verify audio/video sync with ffprobe.
- **Phase 2 – Full verification:** Wikidata, NLI, LLM judge, rewrite loop,
  rejections, adversarial tests.
- **Phase 3 – Ideas and queue:** auto-generation, dedupe, inbox, batch,
  false-premise flow, `--no-factcheck`.
- **Phase 4 – Visual polish:** all scene types, themes, Commons images, word-level
  captions (faster-whisper), music ducking, loudness, thumbnails, HW encoding.
- **Phase 5 – Web UI:** all pages, SSE progress, review with claim panel,
  edit-and-rerender, settings, health panel, optional password.
- **Phase 6 – Hardening:** resume/retry, caching, README, full test run,
  final `doctor` run, 3 sample videos on different topics.

After each phase: run the tests, show results, list what's done, what's next and
any problems found. Don't move on with failing tests.

## 10. Definition of done

- `factshorts ui` starts, and I can create, track, review, edit and approve videos
  entirely from the browser.
- Auto and custom ideas both work. Custom ideas never skip fact-checking unless
  explicitly turned off, and then they're labeled.
- Every verified video has a complete `sources.json` and a paste-ready description
  with credits.
- All adversarial fixtures are caught. All tests pass offline.
- Nothing in the project requires a paid service or an API key.

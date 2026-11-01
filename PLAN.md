# factshorts — Phase 0 Plan

Status: **approved with the decisions in section 7.** Next: Phase 1.

## 1. What I understand we're building

A local, free tool that turns an idea ("Why is Venus hotter than Mercury?") into a
30–60 s vertical (1080×1920) animated fact video with Piper narration, where every
factual statement is traceable to a fetched source (Wikipedia by default, or a source
you supply) and has passed a chain of automated checks. The LLM (Ollama) only writes
and judges; it is never the source of truth. Output lands in `output/pending/` with a
`sources.json`, a paste-ready description with credits, a thumbnail and optional `.srt`,
and you approve or reject it via a local Web UI (or the CLI).

The heart of the project is verification. A claim survives only if it passes, in order:
quote check → entity check → Wikidata cross-check → NLI entailment → strict LLM judge.
Failures trigger a targeted scene rewrite (max 3), then scene removal, then rejection
with an exact reason. Sentences that carry no claim must be provably non-factual filler.

## 2. Environment I found on this machine

| Item | Found | Notes |
|---|---|---|
| Python | 3.11.9 (also 3.13 Store, uv-managed 3.11.15) | Target 3.11 |
| uv | 0.11.3 | I'll use uv for the venv and lockfile; plain `pip install -e .` also works |
| ffmpeg / ffprobe | 9.0.2 full build (winget) | `h264_nvenc`, `h264_amf`, `h264_qsv`, `libx264` all present |
| GPU | RTX 5070 Ti, 16 GB VRAM | NVENC will be auto-selected |
| CPU / RAM | Ryzen 7 9800X3D (8C/16T), 31 GB | 8 parallel scene renderers is realistic |
| Ollama | 0.34.0 (update available), models: `qwen3.5:4b`, `llama3.1:8b` | Writer `qwen3:14b` to be pulled |
| Disk | 298 GB free on C: | fine |
| Git | installed | Repo initialized in Phase 0 |
| Location | project lives inside **OneDrive** | Runtime data goes to `C:\factshorts-data` |

## 3. Key libraries (latest on PyPI today, I'll pin `>=` lower bounds and lock with uv)

| Purpose | Library | Version |
|---|---|---|
| Models / config | pydantic, pydantic-settings, stdlib `tomllib` (+ `tomli-w` for writing settings) | 2.14.2, 2.16.2 |
| HTTP (Wikipedia, Wikidata, Commons, Ollama REST) | httpx (+ own retry/backoff) | 0.28.1 |
| Ollama | direct REST `/api/chat` with `format=<JSON schema>` via httpx (no extra SDK) | — |
| CLI | typer | 0.27.3 |
| Web | fastapi, uvicorn, jinja2, python-multipart, sse-starlette, itsdangerous (sessions), argon2-cffi (password hash) | 0.162.0, 0.54.0, 3.1.6, 0.0.33, 3.6.0, 2.2.0, 25.1.0 |
| Front-end (vendored, no CDN at runtime) | htmx 2.x + htmx SSE extension, one small hand-written CSS file | — |
| TTS | piper-tts (Python API, `PiperVoice`) + voices from `rhasspy/piper-voices` via huggingface-hub | 1.8.0, 2.4.0 |
| Word timestamps (optional) | faster-whisper (CPU, `int8`) | 1.2.1 |
| NLI | transformers + torch **CPU wheel**, model `cross-encoder/nli-deberta-v3-base` (configurable, e.g. a DeBERTa-v3-large MNLI/FEVER/ANLI model for higher accuracy) | 5.22.1, 2.15.0 |
| Fuzzy matching / dedupe | rapidfuzz (token similarity; no embedding model needed for v1) | 3.14.6 |
| Numbers → words (TTS) | num2words; words → numbers I write myself (small, fully tested) | 0.5.14 |
| User sources | trafilatura (URL → main text), pypdf (PDF text) | 2.3.1, 6.21.4 |
| Frames | Pillow + numpy, raw RGB piped to ffmpeg; `ProcessPoolExecutor` per scene | 13.0.0, 2.4.6 |
| Storage | stdlib `sqlite3` (WAL mode), no ORM | — |
| Quality | pytest, ruff (lint + format), pytest-asyncio only if needed | 9.2.0, 0.17.3 |

Deliberately **not** used: MoviePy (Pillow → ffmpeg pipe is simpler and faster to control),
spaCy (entity extraction is regex + capitalization heuristics + number parser, which is
enough for "every number/year/proper noun must appear in the quote" and is easy to test),
pint (a small explicit unit table with rounding tolerances is clearer), Ollama Python SDK.

## 4. Proposed file tree

```
pyproject.toml            # package "factshorts", entry point `factshorts`, ruff config
config.toml               # all tunables, commented
README.md
PLAN.md
src/factshorts/
  __init__.py
  cli.py                  # typer app: make, queue, batch, review, rerender, ui, doctor
  config.py               # pydantic models for config.toml, load/save, API-key/remote-LLM guard
  db.py                   # SQLite schema + migrations: ideas, jobs, job_steps, videos,
                          #   rejections, http_cache, settings
  logging_setup.py        # rotating file + console + per-job log handler
  models.py               # shared pydantic types: Idea, Source, Script, Scene, Claim,
                          #   CheckResult, ClaimVerdict, Verification, AudioPlan, RenderPlan...
  core/
    pipeline.py           # step registry, Job runner, resume-from-last-step, cancel
    ideas.py              # auto ideas (Ollama structured output), Wikipedia resolution
    custom_ideas.py       # parse single/multi/ideas.txt with `| length=45 | tone=funny`
    dedupe.py             # normalized title + rapidfuzz token similarity + article/angle key
    sourcing.py           # source resolution order: user → Wikipedia → reject
    premise.py            # false-premise detection → `needs_input`
    script.py             # RAG prompt, JSON schema, word budget enforcement, retries
    rewrite.py            # targeted scene rewrite loop
    tts.py                # sentence synthesis, pauses, duration plan, TTS cache
    textnorm.py           # numbers/units/abbreviations → speakable text, pronunciation overrides
    captions.py           # word timings (whisper or proportional), caption layout
    export.py             # sources.json, metadata.json, description.txt, thumbnail, .srt
    llm.py                # Ollama client: chat with JSON schema, retries, model listing
    http.py               # httpx client w/ User-Agent, backoff, SQLite/disk cache
    workdir.py            # work/<job_id>/ layout and atomic JSON writes
  sources/
    wikipedia.py          # search, extract (plain text by section), revision ID, QID
    wikidata.py           # entity statements → typed numeric/date values with units
    commons.py            # image search + license filtering + attribution
    user_source.py        # URL / pasted text / .txt .md .pdf, hashed
  verify/
    normalize.py          # quotes/dashes/whitespace, number parsing (digits, words, separators)
    units.py              # unit table + conversions (m↔km, kg↔t, °F→°C, mi→km, ft→m, lb→kg)
    entities.py           # extract numbers/years/dates/proper nouns from narration
    programmatic.py       # quote check + entity check
    wikidata_check.py     # claim ↔ Wikidata property cross-check
    nli.py                # lazy-loaded cross-encoder, batched, CPU
    llm_judge.py          # SUPPORTED / NOT_SUPPORTED / UNCLEAR + reason
    sentences.py          # every sentence → claim or provably non-factual filler
    aggregate.py          # all enabled checks must pass; builds verification.json
  render/
    themes.py             # load assets/themes/*.toml
    easing.py             # easing curves
    layout.py             # 1080×1920 safe zone, text wrapping/fitting
    background.py         # gradients, particles, noise (numpy, seeded)
    text.py               # kinetic text (fade/scale/slide)
    captions_draw.py      # word-by-word highlighted captions
    scenes/               # big_number, comparison, timeline, map_pin, icon_list, image, quote
    endcard.py, progress.py, transitions.py
    encoder.py            # NVENC/AMF/QSV detection with a real test encode, libx264 fallback
    ffmpeg.py             # pipe writer, concat, mux, loudnorm, sidechain ducking, ffprobe
    renderer.py           # parallel scene rendering → concat → mux
  web/
    app.py                # FastAPI app factory, auth middleware (only when host != localhost)
    routes/               # create.py, jobs.py, review.py, library.py, settings.py, sse.py
    worker.py             # background job worker(s), configurable concurrency
    templates/            # Jinja2 + HTMX partials
    static/               # htmx.min.js, sse.js, app.css, icons (all vendored)
  doctor.py               # shared by CLI and Settings health panel
assets/
  fonts/                  # OFL fonts + their OFL.txt (e.g. Inter, Montserrat, Bebas Neue)
  themes/                 # default.toml, neon.toml, paper.toml ...
  music/                  # empty by default (you add royalty-free tracks)
  icons/                  # self-drawn / CC0 SVG-like shapes rendered by code
tests/
  conftest.py             # offline: fake LLM, fake TTS (sine/silence WAV), fake Wikipedia,
                          #   fake Wikidata, fake NLI; network calls blocked
  fixtures/               # source texts, scripts, adversarial scripts, ideas.txt
  test_normalize.py, test_units.py, test_quote_match.py, test_entities.py,
  test_aggregate.py, test_textnorm.py, test_captions.py, test_layout.py,
  test_dedupe.py, test_premise.py, test_adversarial.py, test_pipeline_resume.py,
  test_web_api.py, test_golden_render.py, test_doctor.py
```

Runtime folders (created on demand, configurable, git-ignored):
`work/<job_id>/`, `output/{pending,approved,rejected}/`, `cache/` (HTTP, TTS, models), `logs/`,
`data/factshorts.db`.

## 5. Key design decisions

- **Pipeline steps** are `idea → sourcing → script → verification → voice → render → export`.
  Each step is a function `(ctx, input_model) -> output_model`, writes its result
  atomically into `work/<job_id>/`, and records `done` + an input hash in `job_steps`.
  Resume = skip steps whose output exists and whose input hash still matches.
  "Edit narration" / "Regenerate scene" invalidate only that scene's downstream artefacts.
- **One job runner** used by both CLI (runs inline, prints progress) and Web UI
  (background worker thread, progress pushed via SSE). Same code path, same results.
- **Claims and sentences:** the script schema links each narration sentence to claim IDs.
  The entity check runs on the *sentence*, not only on the claim text, so a number slipped
  into narration without a claim is caught. A sentence with no claim passes only if it has
  no numbers/dates/proper nouns **and** the LLM judge classifies it as non-factual filler.
- **Wikidata cross-check** uses a configurable property map (elevation P2044, height P2048,
  length P2043, area P2046, population P1082, mass P2067, diameter P2386, inception P571,
  birth/death P569/P570, etc.). If no matching property exists the check is `skipped`
  (not failed); if it exists and disagrees beyond tolerance, the claim is dropped.
- **False premise:** before scripting, the LLM extracts the idea's embedded presuppositions
  ("Venus is hotter than Mercury"), each is verified against the source with the same
  NLI + judge checks; a *contradiction* sets the job to `needs_input` with the excerpt.
- **Rendering:** each scene renders in its own process to a video-only segment
  (raw RGB → ffmpeg pipe), segments are concatenated, then muxed with narration;
  loudness via two-pass `loudnorm` to −14 LUFS; optional music ducked with
  `sidechaincompress`. Captions are drawn into frames (not burned by ffmpeg) so they can
  be word-highlighted.
- **Encoder detection** tries a 1-frame test encode with each HW encoder, because
  "listed in `ffmpeg -encoders`" doesn't guarantee the driver works.
- **Safety defaults:** all checks on, review required, localhost only. `--host 0.0.0.0`
  refuses to start without a password hash in config. `doctor` fails on anything that
  looks like an API key or a non-local LLM endpoint.

## 6. Assumptions (tell me if any is wrong)

1. Video language is **American English** only, from **English Wikipedia**.
2. Default Ollama endpoint `http://127.0.0.1:11434`.
3. NLI runs on CPU as the spec says (avoids a CUDA/Blackwell torch wheel). A `device`
   setting allows `cuda` later.
4. The bundled fonts and the vendored htmx file are downloaded **once by me during
   development** and committed into `assets/` / `web/static/` with their licenses.
   At runtime nothing is fetched from a CDN.
5. Piper voices and the NLI model download on first use into `cache/models/`.
6. `piper-tts` 1.8 is GPL-3.0 licensed. Fine for local personal use; I'll note it in the README.
7. Hashtags in descriptions are generated from the topic and article title, not invented facts.
8. Tests that need ffmpeg (the golden test) run locally since ffmpeg is installed; they
   still need no network.

## 7. Decisions (answered 2026-11-01)

1. **Runtime data** lives outside OneDrive in `C:\factshorts-data\` (config `data_dir`,
   defaults to a platform-appropriate path on Linux): `work/`, `output/`, `cache/`, `logs/`,
   `factshorts.db`. The venv also lives there via `UV_PROJECT_ENVIRONMENT`.
   Code stays in the OneDrive project folder.
2. **American English**: "meters", "kilometers" on screen and in TTS normalization,
   default `en_US` Piper voice (high quality where available).
3. **LLM models**: writer = `qwen3:14b` (9.3 GB, you pull it), judge = `llama3.1:8b`
   (already installed, different model family).
4. **Music**: built-in procedural ambient generator (copyright-free); files in
   `assets/music/` are used instead when present.
5. **Git**: initialized; one commit per finished phase, no AI attribution lines.
6. **Phase 1 sample idea**: "How tall is Mount Everest?"

## 8. Phase 1 scope (next)

CLI `factshorts make --idea "..."` → Wikipedia article (text, URL, revision ID) →
script via Ollama with JSON schema and word budget → quote + entity checks →
Piper per sentence → simple render (gradient background, kinetic on-screen text,
proportional captions, progress bar) → `final.mp4` + `sources.json` →
one real sample video, with an ffprobe check that audio and video durations match
within 100 ms. Tests for normalization, quote matching, entity check and TTS text
normalization run offline with ruff clean.

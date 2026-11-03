# factshorts

Make vertical (9:16) fact videos for TikTok, YouTube Shorts and Instagram Reels, entirely on
your own computer, with narration, AI illustrations, captions, music and a fact-check of every
sentence against real sources.

- **Free and local.** No paid APIs, no API keys, no accounts, no cloud. The language models
  run in [Ollama](https://ollama.com); voice, pictures and the fact-checking model run with
  PyTorch on your machine.
- **The LLM is never the source of truth.** Every fact in a video must come from a fetched
  source (Wikipedia, or a page/file/text you provide) and pass automatic checks.
- **Metric units only** (km, m, kg, °C, km/h), American English.
- Every video comes in two versions, with and without background music (the music is made
  as its own track first), plus a short `tiktok.txt` caption to paste into TikTok, a full
  `description.txt` with source credits (for YouTube), `sources.json`, `metadata.json`,
  subtitles (`.srt`) and a `thumbnail.jpg`.
- Use it from the browser (`factshorts ui`) or the command line.

## Limitations (please read)

- **The checks reduce mistakes; they cannot rule them out.** A local LLM writes the script and
  local models check it. Five independent checks catch most invented numbers, names and
  claims, but a subtle misreading can still pass all of them. **Watch every video and read its
  claims before you publish it.** That is why every video waits in `output/pending/` for
  your review, and the Review page shows each claim next to its source quote.
- Sources can be wrong too. Wikipedia is usually well maintained, but the video can only be
  as correct as the article revision it used (linked in `sources.json`).
- The illustrations are AI-generated and stylized, not photos. They are checked by a vision
  model against the narration, but can still show details that are not exactly right. The
  description says the visuals are AI-generated.
- Some topics do not have enough verifiable facts for a one-minute video. The tool then
  drops what it cannot verify, so the video can come out shorter (it tells you) or the idea is
  rejected with the reason.
- It needs a fairly strong PC: a modern NVIDIA graphics card with 12-16 GB of VRAM is
  recommended. A 75-second video takes about 10-15 minutes on an RTX 5070 Ti.
- Opinion or creative videos can be made with fact-checking turned off. They are always
  labeled **UNVERIFIED** (file name, metadata, description and UI).

## Setup on Windows, step by step

You need about 70 GB of free disk space for the models (the two Ollama models about 27 GB,
the image model about 31 GB, voice, depth and fact-checking models about 1.5 GB).

1. **Python 3.11 or newer** and **uv** (the package manager):

   ```bash
   winget install Python.Python.3.11
   ```

   ```bash
   winget install astral-sh.uv
   ```

2. **ffmpeg** (video encoding):

   ```bash
   winget install Gyan.FFmpeg
   ```

3. **Ollama** (runs the language models locally). Install it, then pull a writer and a judge
   model (see "Which models?" below):

   ```bash
   winget install Ollama.Ollama
   ```

   ```bash
   ollama pull qwen3:14b
   ```

   ```bash
   ollama pull gemma4:26b
   ```

4. **factshorts itself.** Get the code (needs `winget install Git.Git`), then create an
   environment in the project folder and install. Keeping the environment outside OneDrive or
   Dropbox folders avoids sync trouble with large files:

   ```bash
   git clone https://github.com/AreaMonster/factshorts.git
   ```

   ```bash
   cd factshorts
   ```

   ```bash
   uv venv C:\factshorts-data\.venv --python 3.11
   ```

   ```bash
   uv pip install --python C:\factshorts-data\.venv\Scripts\python.exe -e ".[dev]"
   ```

   With an NVIDIA RTX 50xx card, PyTorch comes from the CUDA 13 index configured in
   `pyproject.toml`.

5. **Edit `config.toml`:** put your contact address in `[wikipedia] user_agent` (Wikimedia asks
   for one). Everything else has working defaults; data goes to `C:\factshorts-data`.

6. **Check the setup:**

   ```bash
   C:\factshorts-data\.venv\Scripts\factshorts doctor
   ```

   It checks Python, ffmpeg and hardware encoders, Ollama and the two models, the voice, the
   fact-checking and image models, the GPU, folder permissions, and that the config contains
   no API keys or remote LLM addresses. Models that are not downloaded yet are fetched on
   first use (the image model is about 31 GB, so the first video takes a while).

7. **Start the UI** and open http://127.0.0.1:8000:

   ```bash
   C:\factshorts-data\.venv\Scripts\factshorts ui
   ```

## Using it

**In the browser** (`factshorts ui`):

- **Create:** write one or more ideas (one per line). Optional per-line settings:
  `Why is Venus hotter than Mercury? | length=60 | tone=funny`. You can attach your own
  source (web page, .txt/.md/.pdf file or pasted text). "Generate ideas" lets the LLM propose
  ideas for topics, each tied to a real Wikipedia article; tick and edit them, then make them
  or put them in the inbox.
- **Jobs:** live progress of every job, step by step, with cancel and retry. If an idea assumes
  something false ("Why is the Great Wall visible from space?"), the job stops and shows what
  the source says; you choose "make a corrected video" or cancel. The inbox, batch runs and
  rejection reasons are here too.
- **Review:** the video next to its script; every claim with its source quote, the result
  of each check and a link to the exact article revision. Switch the music on or off while
  watching; Download saves the version you picked. The TikTok caption is there with a Copy
  button. It is written from the checked script only: if the LLM's line contains a number or
  name that is not in the script, the checked opening line is used instead. Approve or
  reject, edit a scene's narration (it is fact-checked again, and only what changed is
  redone), regenerate a scene, or re-render with another theme or voice.
- **Library:** approved videos, searchable, with downloads of the MP4 and description.
- **Settings:** models, voice (with previews), checks, defaults, topics, password, and a
  health panel with the same checks as `doctor`.

**On the command line:**

```text
factshorts make --idea "How tall is Mount Everest?" [--length 60 --tone funny]
factshorts make --idea "..." --source-url https://... | --source-file notes.pdf
factshorts make --ideas-file ideas.txt
factshorts make --topic space --count 3
factshorts make --idea "..." --no-factcheck        # labeled UNVERIFIED
factshorts queue add "idea" | queue generate --topic space | queue list | queue clear
factshorts batch --per-topic 2                      # inbox first, then new auto ideas
factshorts answer <job_id> corrected|cancel         # for false-premise ideas
factshorts resume <job_id> [--from verification]    # continue after a crash, or redo from a step
factshorts resume <job_id> --from visuals            # draw all pictures again
factshorts rerender <job_id> --theme watercolor --voice af_heart --no-music
factshorts review                                   # approve/reject in the terminal
factshorts jobs
factshorts doctor
factshorts ui [--host --port]
factshorts password [--remove]
```

Jobs resume from their last finished step: if the PC restarts mid-video, `factshorts resume`
(or Retry/Resume in the UI) continues where it stopped without re-writing or re-checking the
script. Tasks started in the UI resume on their own when the UI starts again.

## Which models?

Use the largest models that fit your graphics card. Bigger models follow the fact rules
better and invent less. Rough guide for the **writer** (quantized, as Ollama ships them):

| VRAM | Writer model size | Examples |
|---|---|---|
| 8 GB | 7-8B | `qwen3:8b`, `llama3.1:8b` |
| 12-16 GB | 12-14B | `qwen3:14b` (default) |
| 24 GB+ | 27-32B | `qwen3:32b`, `gemma4:26b` |

The **judge** checks claims and pictures. It should come from a different model family than
the writer, so it does not share its blind spots. `gemma4:26b` (default) was right on every
case of a labeled test set where `llama3.1:8b` passed a wrong number. If it does not fit
your VRAM, Ollama runs part of it on the CPU: slower, but it works. The image and voice
models load only after the LLM has been unloaded, so they do not compete for VRAM.

## How fact-checking works

```text
 idea ──► source (Wikipedia article revision, or your source) ──► premise check
                                                                      │
          script written ONLY from the source; every sentence cites   ▼
          the exact source sentence(s) it relies on  ◄────────────────┘
                     │
                     ▼   every claim must pass ALL enabled checks, cheapest first
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ 1. quote     the cited sentence really is in the source                   │
   │ 2. entities  every number, date and name is in that sentence (with unit   │
   │              conversion, rounding and spelled-out numbers)                │
   │ 3. Wikidata  numbers and dates agree with Wikidata, where it has them     │
   │ 4. NLI       a local entailment model: the source sentence implies it     │
   │ 5. LLM judge a second model, strictly: SUPPORTED / NOT_SUPPORTED / UNCLEAR │
   └──────────────────────────────────────────────────────────────────────────┘
          sentences without a claim must be fact-free filler (the judge checks)
                     │
     fail ──► targeted rewrite of that scene (up to 3 times) ──► still failing:
              drop the scene (if the story still works) or reject the idea with the reason
                     │
                     ▼
     pass ──► voice ──► illustrations (checked by a vision model) ──► video ──► your review
```

All results (every claim, each check's verdict, scores and reasons) are saved in
`verification.json` and `sources.json`. Videos are labeled **VERIFIED**,
**PARTIALLY_VERIFIED** (a check could not run, e.g. Wikidata was unreachable) or
**UNVERIFIED** (fact-check turned off).

## Customizing

- **Topics** for automatic ideas: `[ideas] topics` in `config.toml`, or Settings in the UI.
- **Themes:** copy a file in `assets/themes/` (for example `watercolor.toml`), change the
  colors, fonts, animation speed and the `[art] style` sentence that describes the
  illustration style, and save it under a new name. It then appears in every theme list. Fonts
  must be free (OFL) files in `assets/fonts/`.
- **Music:** put royalty-free `.mp3`, `.wav`, `.ogg`, `.flac` or `.m4a` files you are allowed to
  use in `assets/music/`. Without files, factshorts generates a calm ambient pad itself (no
  copyright). Music is ducked under the voice and the mix is normalized to -14 LUFS.
- **Voices:** the default engine is Kokoro (`am_michael`). Other voices, such as `af_heart`
  or `bm_george`, can be previewed and picked in Settings. Piper voices (`tts.engine = "piper"`,
  names like `en_US-ryan-high`) download from Hugging Face on first use or from Settings.
- **Pronunciation:** add names the voice says wrong to `[tts] pronunciations`.

## Using it from a phone on the same Wi-Fi

1. Set a password: `factshorts password` (or on the Settings page).
2. Start with `factshorts ui --host 0.0.0.0` and open `http://<your PC's IP>:8000` on the
   phone (Windows asks once whether to allow it through the firewall: allow it for private
   networks only). Without a password the UI refuses to start on a network address.

**Security warning:** anyone on the same network can reach the login page. The connection is
plain HTTP, not encrypted, so only do this on a network you trust (your home Wi-Fi, never a
public one). Use a strong password and stop the UI when you no longer need it. The UI
can start jobs and delete videos, and it has no further user management.

## Licensing notes

- **Wikipedia text is licensed CC BY-SA.** Videos built from it must credit the sources:
  `description.txt` lists the articles (with links to the exact revisions) and is meant to be
  pasted into the video's description. Keep it there.
- Your own sources stay your responsibility: only use pages and files you may use.
- The illustrations are generated locally by AI and are labeled as such in the description.
  The generated ambient music is original. Files you add to `assets/music/` must be ones
  you may use.
- Bundled fonts (Bebas Neue, Montserrat) are under the SIL Open Font License (see
  `assets/fonts/`). htmx is under the Zero-Clause BSD license.
- Models are downloaded from their publishers and keep their own licenses. Kokoro-82M,
  Z-Image-Turbo and Depth-Anything-V2-Small are Apache 2.0; check each model's page on
  Hugging Face or Ollama, especially before commercial use.
- `piper-tts` (the fallback voice engine) is GPL-3.0, so factshorts is GPL-3.0-or-later too.
  That is fine for personal use and for publishing the videos you make.

## Development

```bash
C:\factshorts-data\.venv\Scripts\python -m pytest
```

The tests run offline with mocked LLM, voice, Wikipedia, Wikidata and NLI. They include a
golden render test checked with `ffprobe` and adversarial scripts (a wrong number, a wrong
year, a swapped name, an unsupported extra sentence) that must all be caught. Tests against the
real local models are opt-in: set `FACTSHORTS_MODEL_TESTS=1`. Lint with `ruff check src tests`.

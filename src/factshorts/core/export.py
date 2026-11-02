"""Write the deliverables next to the MP4: sources.json, metadata.json, description.txt, .srt."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from factshorts import __version__
from factshorts.core.captions import caption_chunks
from factshorts.models import (
    AudioPlan,
    ExportResult,
    Idea,
    RenderResult,
    Script,
    SourceDoc,
    Verification,
    VisualsPlan,
    utcnow,
)
from factshorts.render.themes import load_theme
from factshorts.render.thumbnail import frame_at, make_thumbnail

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Facts were checked automatically against the cited source. Automated checks reduce "
    "errors but cannot eliminate them; please report mistakes."
)
WIKIPEDIA_LICENSE = "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "video"


def source_label(source: SourceDoc) -> str:
    if source.kind == "wikipedia":
        titles = " and ".join(f"“{r.title}”" for r in source.refs)
        return f"Source: Wikipedia, {titles}"
    if source.kind == "none":
        return "Opinion video: not fact-checked"
    return f"Source: {source.title}"


def hashtags(script: Script, source: SourceDoc, topic: str | None) -> list[str]:
    tags = ["#facts", "#shorts", "#learnontiktok"]
    for word in (source.title, topic or ""):
        tag = re.sub(r"[^A-Za-z0-9]", "", word.title())
        if tag and f"#{tag.lower()}" not in [t.lower() for t in tags]:
            tags.append(f"#{tag}")
    return tags


def build_sources_json(
    idea: Idea,
    source: SourceDoc,
    script: Script,
    verification: Verification | None,
    verified: bool,
) -> dict:
    claims = []
    by_id = {c.claim_id: c for c in verification.claims} if verification else {}
    for seg in script.segments:
        for claim in seg.claims:
            v = by_id.get(claim.id)
            claims.append(
                {
                    "id": claim.id,
                    "segment": seg.id,
                    "text": claim.text,
                    "type": claim.type,
                    "source_quote": claim.source_quote,
                    "paragraph": v.paragraph if v else None,
                    "passed": v.passed if v else None,
                    "checks": [c.model_dump() for c in v.checks] if v else [],
                }
            )
    return {
        "generator": f"factshorts {__version__}",
        "generated_at": utcnow().isoformat(),
        "verified": verified,
        "label": verification.label if verification else "UNVERIFIED",
        "checks_not_run": verification.skipped_checks if verification else [],
        "disclaimer": DISCLAIMER,
        "idea": idea.text,
        "source": {
            "kind": source.kind,
            "title": source.title,
            "url": source.url,
            "revision_id": source.revision_id,
            "revision_url": source.revision_url,
            "wikidata_id": source.wikidata_id,
            "sha256": source.sha256,
            "license": WIKIPEDIA_LICENSE if source.kind == "wikipedia" else None,
            "user_source": source.kind == "user",
            "documents": [
                {
                    "title": r.title,
                    "url": r.url,
                    "revision_id": r.revision_id,
                    "revision_url": r.revision_url,
                    "wikidata_id": r.wikidata_id,
                }
                for r in source.refs
            ],
        },
        "enabled_checks": verification.enabled_checks if verification else [],
        "claims": claims,
        "sentences": [s.model_dump() for s in verification.sentences] if verification else [],
        "images": [],
    }


def ai_visuals_note(visuals: VisualsPlan | None) -> str | None:
    if visuals is None or visuals.mode == "off" or not visuals.scenes:
        return None
    return (
        "Illustrations and animation are AI-generated (stylized, not real footage); "
        "the narration is checked against the sources below."
    )


def build_description(
    script: Script,
    source: SourceDoc,
    tags: list[str],
    label: str,
    not_run: list[str],
    visuals: VisualsPlan | None = None,
) -> str:
    lines = [script.title, "", script.hook.narration]
    if label == "UNVERIFIED":
        lines += ["", "UNVERIFIED: this video was not fact-checked."]
    elif label == "PARTIALLY_VERIFIED":
        lines += ["", f"PARTIALLY VERIFIED: these checks did not run: {', '.join(not_run)}."]
    lines += ["", "Sources:"]
    if source.kind == "wikipedia":
        for r in source.refs:
            lines.append(f"- Wikipedia, “{r.title}” (revision {r.revision_id}): {r.revision_url}")
        lines.append(f"  Text license: {WIKIPEDIA_LICENSE}")
    elif source.kind == "none":
        lines.append("- none (opinion/creative video, not fact-checked)")
    else:
        lines.append(f"- {source.title}" + (f": {source.url}" if source.url else ""))
    note = ai_visuals_note(visuals)
    if note:
        lines += ["", note]
    if label != "UNVERIFIED":
        lines += ["", DISCLAIMER]
    lines += ["", " ".join(tags)]
    return "\n".join(lines) + "\n"


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(audio: AudioPlan) -> str:
    blocks = []
    n = 0
    for seg in audio.segments:
        for sentence in seg.sentences:
            for chunk in caption_chunks(sentence.words, max_words=7, max_chars=42):
                n += 1
                text = " ".join(w.text for w in chunk)
                blocks.append(
                    f"{n}\n{_srt_time(chunk[0].start)} --> {_srt_time(chunk[-1].end)}\n{text}\n"
                )
    return "\n".join(blocks)


def export(
    job_id: str,
    idea: Idea,
    source: SourceDoc,
    script: Script,
    verification: Verification | None,
    audio: AudioPlan,
    render: RenderResult,
    output_dir: Path,
    verified: bool,
    sync: dict[str, float] | None = None,
    visuals: VisualsPlan | None = None,
    thumbnail_base: Path | None = None,
    theme: str | None = None,
) -> ExportResult:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    full_label = verification.label if verification and verified else "UNVERIFIED"
    verified = full_label == "VERIFIED"
    label = {"VERIFIED": "", "PARTIALLY_VERIFIED": "_PARTIAL"}.get(full_label, "_UNVERIFIED")
    slug = slugify(script.title)
    folder = output_dir / "pending" / f"{stamp}_{slug}{label}_{job_id[-6:]}"
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / f"{slug}{label}.mp4"
    shutil.copy2(render.video, video)

    tags = hashtags(script, source, idea.topic)
    sources = build_sources_json(idea, source, script, verification, verified)
    sources["video"] = {
        "file": video.name,
        "duration_s": render.duration,
        "frames": render.frames,
        "encoder": render.encoder,
        "sync": sync,
        "music": render.music,
        "loudness_lufs": render.lufs,
    }
    if visuals is not None and visuals.mode != "off":
        sources["visuals"] = {
            "ai_generated": True,
            "mode": visuals.mode,
            "models": visuals.models,
            "style": visuals.style,
            "scenes": [
                {
                    "segment": sc.segment_id,
                    "kind": sc.kind,
                    "prompt": sc.prompt,
                    "motion": sc.motion if sc.kind == "wan" else None,
                    "seed": sc.seed,
                }
                for sc in visuals.scenes
            ],
        }
    sources_path = folder / "sources.json"
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

    description = build_description(
        script, source, tags, full_label, sources["checks_not_run"], visuals
    )
    desc_path = folder / "description.txt"
    desc_path.write_text(description, encoding="utf-8")

    metadata = {
        "job_id": job_id,
        "title": script.title,
        "description": script.hook.narration,
        "hashtags": tags,
        "verified": verified,
        "label": sources["label"],
        "topic": idea.topic,
        "length_target_s": idea.length,
        "tone": idea.tone,
        "duration_s": render.duration,
        "voice": audio.voice,
        "source_links": [r.revision_url for r in source.refs if r.revision_url],
        "image_credits": [],
        "ai_generated_visuals": bool(visuals and visuals.mode != "off" and visuals.scenes),
        "status": "pending",
    }
    meta_path = folder / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    srt_path = folder / f"{slug}{label}.srt"
    srt_path.write_text(build_srt(audio), encoding="utf-8")

    thumb = folder / "thumbnail.jpg"
    try:
        base = (
            Image.open(thumbnail_base).convert("RGB")
            if thumbnail_base
            else frame_at(video, min(1.0, render.duration / 2))
        )
        make_thumbnail(base, script.title, load_theme(theme or "default"), thumb)
    except Exception as exc:  # a missing thumbnail must not lose the finished video
        log.warning("thumbnail failed: %s", exc)
        thumb = None

    return ExportResult(
        folder=str(folder),
        video=str(video),
        sources_json=str(sources_path),
        metadata_json=str(meta_path),
        description_txt=str(desc_path),
        srt=str(srt_path),
        thumbnail=str(thumb) if thumb else None,
        verified=verified,
    )

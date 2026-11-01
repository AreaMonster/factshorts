"""User-provided sources: pasted text, a .txt/.md/.pdf file, or a web page URL."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from factshorts.models import Section, SourceDoc

MIN_WORDS = 60


class UserSourceError(ValueError):
    pass


def _clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links → their text
    text = re.sub(r"^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+", "", text, flags=re.M)  # markers
    return re.sub(r"(\*\*|__|\*|_|`)", "", text)


def from_text(text: str, title: str = "User-provided text", url: str | None = None) -> SourceDoc:
    """Paragraphs are separated by blank lines; line breaks inside a paragraph are joined."""
    blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", text)]
    paragraphs = [b for b in blocks if b]
    body = "\n".join(paragraphs)
    if len(body.split()) < MIN_WORDS:
        raise UserSourceError(
            f"the source has only {len(body.split())} words; at least {MIN_WORDS} are needed "
            "to fact-check a video"
        )
    return SourceDoc(
        kind="user",
        title=title,
        url=url,
        sections=[Section(index=0, title=title, text=body)],
        sha256=hashlib.sha256(body.encode()).hexdigest(),
    )


def from_file(path: Path) -> SourceDoc:
    if not path.is_file():
        raise UserSourceError(f"source file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        text = "\n\n".join(pages)
    elif suffix in (".txt", ".md", ".markdown"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix != ".txt":
            text = _clean_markdown(text)
    else:
        raise UserSourceError(f"unsupported source file type '{suffix}' (use .txt, .md or .pdf)")
    return from_text(text, title=path.stem)


def from_url(url: str, http: Any) -> SourceDoc:
    """Main text of a web page (boilerplate removed by trafilatura)."""
    import trafilatura

    html = http.get_text(url)
    text = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_precision=True
    )
    if not text:
        raise UserSourceError(f"no readable article text found at {url}")
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else url).strip()
    # trafilatura separates paragraphs with single newlines
    return from_text(text.replace("\n", "\n\n"), title=title, url=url)

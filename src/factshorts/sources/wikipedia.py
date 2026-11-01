"""Wikipedia: search, plain-text extract split into sections, revision ID and Wikidata QID."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from factshorts.config import WikipediaConfig
from factshorts.core.http import CachedHttp
from factshorts.models import Section, SourceDoc

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


class ArticleNotFound(LookupError):
    pass


@dataclass
class SearchHit:
    title: str
    snippet: str
    page_id: int


class Wikipedia:
    def __init__(self, http: CachedHttp, cfg: WikipediaConfig) -> None:
        self.http = http
        self.cfg = cfg
        self.api = f"https://{cfg.language}.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int | None = None) -> list[SearchHit]:
        data = self.http.get_json(
            self.api,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit or self.cfg.search_limit,
                "srprop": "snippet",
                "format": "json",
                "formatversion": 2,
            },
        )
        hits = data.get("query", {}).get("search", [])
        return [
            SearchHit(
                title=h["title"],
                snippet=re.sub(r"<[^>]+>", "", h.get("snippet", "")),
                page_id=h["pageid"],
            )
            for h in hits
        ]

    def fetch(self, title: str) -> SourceDoc:
        data = self.http.get_json(
            self.api,
            {
                "action": "query",
                "prop": "extracts|revisions|pageprops|info",
                "explaintext": 1,
                "exsectionformat": "wiki",
                "titles": title,
                "redirects": 1,
                "rvprop": "ids",
                "inprop": "url",
                "ppprop": "wikibase_item|disambiguation",
                "format": "json",
                "formatversion": 2,
            },
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise ArticleNotFound(f"Wikipedia article '{title}' not found")
        page = pages[0]
        props = page.get("pageprops", {})
        if "disambiguation" in props:
            raise ArticleNotFound(f"'{page['title']}' is a disambiguation page")
        extract = page.get("extract", "")
        if not extract.strip():
            raise ArticleNotFound(f"Wikipedia article '{page['title']}' has no text")
        sections = select_sections(
            split_sections(extract), self.cfg.max_sections, self.cfg.skip_sections
        )
        text = "\n\n".join(s.text for s in sections)
        return SourceDoc(
            kind="wikipedia",
            title=page["title"],
            url=page.get("fullurl") or page.get("canonicalurl"),
            language=self.cfg.language,
            revision_id=page.get("lastrevid") or page["revisions"][0]["revid"],
            page_id=page.get("pageid"),
            wikidata_id=props.get("wikibase_item"),
            sections=sections,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


def split_sections(extract: str) -> list[tuple[int, str, str]]:
    """Split a TextExtracts plain-text extract into (level, title, text); lead is level 1."""
    out: list[tuple[int, str, str]] = []
    pos = 0
    level, title = 1, "Lead"
    for m in HEADING_RE.finditer(extract):
        out.append((level, title, extract[pos : m.start()]))
        level, title, pos = len(m.group(1)), m.group(2), m.end()
    out.append((level, title, extract[pos:]))
    return [(lv, t, clean_text(body)) for lv, t, body in out]


def select_sections(
    parts: list[tuple[int, str, str]], max_sections: int, skip: list[str]
) -> list[Section]:
    """Keep the lead plus the first `max_sections` top-level sections (with subsections)."""
    skip_lower = {s.lower() for s in skip}
    selected: list[Section] = []
    top_seen = 0
    skipping = False
    for level, title, text in parts:
        if level == 2:
            top_seen += 1
            skipping = title.lower() in skip_lower
            if not skipping and top_seen > max_sections:
                break
        elif level > 2 and title.lower() in skip_lower:
            continue
        if skipping or not text:
            continue
        selected.append(Section(index=len(selected), title=title, text=text))
    return selected


def clean_text(text: str) -> str:
    # Collapse whitespace inside paragraphs but keep paragraph breaks.
    paragraphs = [re.sub(r"[ \t ]+", " ", p).strip() for p in text.split("\n")]
    return "\n".join(p for p in paragraphs if p)

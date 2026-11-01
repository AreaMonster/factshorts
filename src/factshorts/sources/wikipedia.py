"""Wikipedia: search, plain-text extract split into sections, revision ID and Wikidata QID."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from factshorts.config import WikipediaConfig
from factshorts.core.http import CachedHttp
from factshorts.models import Section, SourceDoc
from factshorts.verify.normalize import normalize_title

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

    def fetch(self, title: str, focus: str | None = None) -> SourceDoc:
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
        if focus:  # the article's own name is in every section; it says nothing about relevance
            own = set(normalize_title(page["title"]).split())
            focus = " ".join(w for w in normalize_title(focus).split() if w not in own)
        sections = select_sections(
            split_sections(extract),
            self.cfg.max_sections,
            self.cfg.skip_sections,
            focus,
            self.cfg.focus_sections,
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
    parts: list[tuple[int, str, str]],
    max_sections: int,
    skip: list[str],
    focus: str | None = None,
    focus_sections: int = 2,
) -> list[Section]:
    """Keep the lead, the first `max_sections` top-level sections (best-maintained text) and up
    to `focus_sections` further sections that best match the `focus` keywords (the idea)."""
    skip_lower = {s.lower() for s in skip}
    groups: list[list[tuple[int, str, str]]] = []  # lead, then each top-level section
    for part in parts:
        if part[0] <= 2 or not groups:
            groups.append([part])
        else:
            groups[-1].append(part)
    lead, rest = groups[0], [g for g in groups[1:] if g[0][1].lower() not in skip_lower]
    chosen = set(range(min(max_sections, len(rest))))
    keywords = normalize_title(focus).split() if focus else []
    if keywords:

        def score(group: list[tuple[int, str, str]]) -> int:
            titles = " ".join(t for _, t, _ in group).lower()
            body = " ".join(x for _, _, x in group).lower()
            return sum(3 * (k in titles) + (k in body) for k in keywords)

        extra = sorted(
            (i for i in range(len(rest)) if i not in chosen),
            key=lambda i: score(rest[i]),
            reverse=True,
        )
        chosen |= {i for i in extra[:focus_sections] if score(rest[i]) >= 3}
    selected: list[Section] = []
    for group in [lead, *(rest[i] for i in sorted(chosen))]:
        for _, title, text in group:
            if text and title.lower() not in skip_lower:
                selected.append(Section(index=len(selected), title=title, text=text))
    return selected


def clean_text(text: str) -> str:
    # Collapse whitespace inside paragraphs but keep paragraph breaks.
    paragraphs = [re.sub(r"[ \t ]+", " ", p).strip() for p in text.split("\n")]
    return "\n".join(p for p in paragraphs if p)

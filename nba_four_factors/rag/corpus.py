"""Corpus loading and chunking.

RAG step one: turn a pile of documents into retrievable pieces.
Chunking is the most underrated design decision in RAG. Too small and
a chunk lacks the context to answer anything; too big and retrieval
gets blurry because one vector has to summarize several topics. We
split markdown on headings first (sections are natural semantic
units), then window long sections with overlap so a fact near a
boundary appears in two chunks instead of being cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DEFAULT_ROOTS = [REPO / "docs"]

EXTRA_FILES = [
    REPO / "job" / "Thesis - MSDS590.docx",
    REPO / "job" / "project_writeup.docx",
]

CHUNK_CHARS = 1100
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    source: str
    section: str
    text: str


def _read_markdown(path: Path) -> str:
    return path.read_text(errors="ignore")


def _read_docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def load_documents() -> list[tuple[str, str]]:
    """Return (relative source name, full text) for every corpus file."""
    docs = []
    for root in DEFAULT_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            docs.append((str(p.relative_to(REPO)), _read_markdown(p)))
    for p in EXTRA_FILES:
        if p.exists():
            docs.append((str(p.relative_to(REPO)), _read_docx(p)))
    return docs


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on ## headings."""
    parts = re.split(r"^(#{1,3} .+)$", text, flags=re.MULTILINE)
    sections = []
    heading = "(start)"
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            heading = part.lstrip("# ").strip()
        elif part.strip():
            sections.append((heading, part.strip()))
    return sections


def window(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Greedy paragraph-boundary windows with character overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = cur[-overlap:] + "\n" + p if overlap else p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_documents():
        for heading, body in split_sections(text):
            for piece in window(body):
                chunks.append(Chunk(source=source, section=heading, text=piece))
    return chunks

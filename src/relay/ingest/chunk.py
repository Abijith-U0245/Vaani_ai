"""Deterministic, page-local text chunking with source offsets."""

from __future__ import annotations

from dataclasses import dataclass

from .extract import PageText, SectionSpan


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    page: int
    page_number: int
    section: str | None
    start: int
    end: int
    text: str
    search_text: str


def _search_text(text: str) -> str:
    import re

    text = re.sub(r"(?<![\w-])([A-Za-z]+)-\n(?=[a-z])", r"\1", text)
    return re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "-", text)


def chunk_pages(
    pages: list[PageText] | tuple[PageText, ...],
    paper_id: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Split each page into overlapping slices whose text is source-addressable."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for page in pages:
        if not page.text:
            continue
        spans = page.section_spans or (
            SectionSpan(0, len(page.text), page.section),
        )
        for span in spans:
            start = span.start
            while start < span.end:
                end = min(start + chunk_size, span.end)
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper_id}-p{page.page_index}-c{len(chunks)}",
                        paper_id=paper_id,
                        page=page.page_index,
                        page_number=page.page_number,
                        section=span.section,
                        start=start,
                        end=end,
                        text=page.text[start:end],
                        search_text=_search_text(page.text[start:end]),
                    )
                )
                if end == span.end:
                    break
                start += step
    return chunks

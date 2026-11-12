"""Deterministic, page-local text chunking with source offsets."""

from __future__ import annotations

from dataclasses import dataclass

from .extract import PageText


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
        start = 0
        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            applicable_headings = [
                heading.title for heading in page.headings if heading.start <= start
            ]
            section = applicable_headings[-1] if applicable_headings else page.section
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}-p{page.page_index}-c{len(chunks)}",
                    paper_id=paper_id,
                    page=page.page_index,
                    page_number=page.page_number,
                    section=section,
                    start=start,
                    end=end,
                    text=page.text[start:end],
                )
            )
            if end == len(page.text):
                break
            start += step
    return chunks

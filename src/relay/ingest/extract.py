"""Layout-aware PDF extraction with page-local provenance offsets."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf


@dataclass(frozen=True)
class SectionHeading:
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class PageText:
    page_index: int
    page_number: int
    text: str
    headings: tuple[SectionHeading, ...]
    dropped_blocks: int = 0
    section: str = "unknown"


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    bold: bool


@dataclass(frozen=True)
class _Block:
    x0: float
    y0: float
    x1: float
    text: str
    lines: tuple[_Line, ...]


_NUMBER = r"\d+(?:\.\d+)*"
_NUMBERED_HEADING = re.compile(
    rf"^(?P<number>{_NUMBER})(?:\s*\.?\s*)(?P<title>[A-Za-z][A-Za-z0-9 :,'()\-–—/&]+)$"
)
_NUMBER_ONLY = re.compile(rf"^(?P<number>{_NUMBER})\.?$")
_APPENDIX_HEADING = re.compile(
    r"^Appendix\s+(?P<number>[A-Z](?:\.\d+)*)\.\s+(?P<title>[A-Za-z][A-Za-z0-9 :,'()\-–—/&]+)$",
    re.IGNORECASE,
)
_APPENDIX_ONLY = re.compile(r"^(?:Appendix\s+)?(?P<number>[A-Z])\.?$", re.IGNORECASE)
_COMMON_HEADING = re.compile(
    r"^(?:ABSTRACT|INTRODUCTION|CONCLUSION(?:S)?|REFERENCES|ACKNOWLEDGMENTS?)$",
    re.IGNORECASE,
)
_CAPTION = re.compile(r"^(?:FIGURE|TABLE)\s+\d+\b", re.IGNORECASE)
_UNIT_LINE = re.compile(r"^\d+(?:\.\d+)?\s+(?:HZ|KHZ|MHZ|GB|MB|KB|MS|S)$", re.IGNORECASE)


def _block_from_raw(raw: dict[str, Any]) -> _Block | None:
    lines: list[_Line] = []
    for raw_line in raw.get("lines", []):
        spans = raw_line.get("spans", [])
        if not spans:
            continue
        # Preserve the PDF text layer exactly. Missing spaces in the source remain missing.
        text_parts: list[str] = []
        for span in spans:
            span_text = str(span.get("text", ""))
            if int(span.get("flags", 0)) & 1 and span_text.strip():
                leading = span_text[: len(span_text) - len(span_text.lstrip())]
                trailing = span_text[len(span_text.rstrip()) :]
                core = span_text.strip()
                span_text = f"{leading}^{{{core}}}{trailing}"
            text_parts.append(span_text)
        text = "".join(text_parts).strip()
        if not text:
            continue
        sizes = [float(span.get("size", 0.0)) for span in spans]
        bold = any(int(span.get("flags", 0)) & 16 for span in spans)
        lines.append(_Line(text=text, size=max(sizes), bold=bold))
    if not lines:
        return None
    return _Block(
        x0=float(raw["bbox"][0]),
        y0=float(raw["bbox"][1]),
        x1=float(raw["bbox"][2]),
        text="\n".join(line.text for line in lines),
        lines=tuple(lines),
    )


def _is_figure_block(block: _Block) -> bool:
    short_lines = sum(len(line.text.strip()) <= 2 for line in block.lines)
    return (
        short_lines >= 1
        and short_lines / len(block.lines) >= 0.6
        and not any(line.bold for line in block.lines)
    )


def _ordered_blocks(page: pymupdf.Page) -> tuple[list[_Block], int]:
    blocks = [
        block
        for raw in page.get_text("dict")["blocks"]
        if (block := _block_from_raw(raw)) is not None
    ]
    dropped = sum(_is_figure_block(block) for block in blocks)
    blocks = [block for block in blocks if not _is_figure_block(block)]
    page_width = page.rect.width
    left = [block for block in blocks if (block.x0 + block.x1) / 2 < page_width / 2]
    right = [block for block in blocks if (block.x0 + block.x1) / 2 >= page_width / 2]
    separated = left and right and max(block.x1 for block in left) < min(
        block.x0 for block in right
    )
    if separated:
        ordered = sorted(left, key=lambda block: (block.y0, block.x0))
        ordered.extend(sorted(right, key=lambda block: (block.y0, block.x0)))
    else:
        ordered = sorted(blocks, key=lambda block: (block.y0, block.x0))
    return ordered, dropped


def _heading_title(number: str, title: str) -> str:
    return f"{number} {title}".strip()


def _valid_section_number(number: str) -> bool:
    parts = [int(part) for part in number.split(".")]
    return (
        0 < parts[0] <= 9
        and all(part > 0 and part <= 20 for part in parts[1:])
    )


def _is_heading(line: _Line, dominant_size: float) -> tuple[str, str] | None:
    title = line.text.strip()
    if not title or len(title) > 100 or title.endswith("."):
        return None
    if (
        _CAPTION.match(title)
        or _UNIT_LINE.fullmatch(title)
        or re.search(r"\b(?:FIGURE|TABLE)\b", title, re.IGNORECASE)
        or "," in title
    ):
        return None
    match = _NUMBERED_HEADING.fullmatch(title)
    if match is not None:
        number, heading = match.group("number"), match.group("title").strip()
        if not _valid_section_number(number):
            return None
        if number.count(".") == 0 and title.startswith(f"{number}. "):
            return None
        if line.bold or line.size >= dominant_size + 0.5 or number.count(".") >= 1:
            return number, _heading_title(number, heading)
        return None
    appendix = _APPENDIX_HEADING.fullmatch(title)
    if appendix is not None and (line.bold or line.size >= dominant_size + 0.5):
        number = appendix.group("number").upper()
        return number, f"Appendix {number}. {appendix.group('title').strip()}"
    if _COMMON_HEADING.fullmatch(title) and (
        line.bold or line.size >= dominant_size + 0.5
    ):
        return "", title
    return None


def _headings(blocks: list[_Block], text: str) -> tuple[SectionHeading, ...]:
    sizes = [line.size for block in blocks for line in block.lines]
    dominant_size = statistics.median(sizes) if sizes else 0.0
    result: list[SectionHeading] = []
    offset = 0
    pending_number: tuple[str, int, _Line] | None = None
    for block in blocks:
        for line in block.lines:
            line_start = offset
            line_end = line_start + len(line.text)
            number_only = _NUMBER_ONLY.fullmatch(line.text.strip())
            appendix_only = _APPENDIX_ONLY.fullmatch(line.text.strip())
            if pending_number is not None:
                number, number_start, number_line = pending_number
                candidate = _is_heading(line, dominant_size)
                if (
                    candidate is not None
                    and candidate[0] == ""
                    and (
                        number.startswith("Appendix ")
                        or _valid_section_number(number)
                    )
                ):
                    result.append(
                        SectionHeading(
                            title=_heading_title(number, candidate[1]),
                            start=number_start,
                            end=line_end,
                        )
                    )
                    pending_number = None
                    offset = line_end + 1
                    continue
                continuation_is_heading = (
                    bool(re.match(r"^[A-Za-z]", line.text.strip()))
                    and not line.text.strip().endswith(".")
                    and "," not in line.text
                    and not re.search(
                        r"\b(?:FIGURE|TABLE)\b", line.text, re.IGNORECASE
                    )
                )
                if (
                    (
                        number.startswith("Appendix ")
                        or _valid_section_number(number)
                    )
                    and continuation_is_heading
                    and (line.bold or line.size >= dominant_size + 0.5)
                ):
                    result.append(
                        SectionHeading(
                            title=_heading_title(number, line.text.strip()),
                            start=number_start,
                            end=line_end,
                        )
                    )
                    pending_number = None
                    offset = line_end + 1
                    continue
                pending_number = None
            if number_only is not None:
                pending_number = (number_only.group("number"), line_start, line)
            elif appendix_only is not None:
                pending_number = (
                    f"Appendix {appendix_only.group('number').upper()}",
                    line_start,
                    line,
                )
            else:
                candidate = _is_heading(line, dominant_size)
                if candidate is not None:
                    result.append(
                        SectionHeading(
                            title=candidate[1],
                            start=line_start,
                            end=line_end,
                        )
                    )
            offset = line_end + 1
    return tuple(result)


def extract_pdf(path: Path | str) -> list[PageText]:
    """Extract pages with normalized text and provenance offsets.

    ``start``/``end`` offsets refer to this processed page text, including
    superscripts represented as ``^{...}``, not the raw PDF text layer.
    """
    pages: list[PageText] = []
    active_section = "unknown"
    saw_references = False
    with pymupdf.open(Path(path)) as document:
        for page_index, page in enumerate(document):
            blocks, dropped = _ordered_blocks(page)
            text = "\n".join(block.text for block in blocks)
            headings = _headings(blocks, text)
            if headings:
                active_section = headings[-1].title
                if active_section.lower() == "references":
                    saw_references = True
            elif saw_references and active_section == "References":
                active_section = "unknown"
            pages.append(
                PageText(
                    page_index=page_index,
                    page_number=page_index + 1,
                    text=text,
                    headings=headings,
                    dropped_blocks=dropped,
                    section=active_section,
                )
            )
    return pages


def page_to_dict(page: PageText) -> dict[str, Any]:
    """Return a JSON-compatible representation for processed artifacts."""
    return {
        "page_index": page.page_index,
        "page_number": page.page_number,
        "text": page.text,
        "dropped_blocks": page.dropped_blocks,
        "section": page.section,
        "headings": [
            {"title": heading.title, "start": heading.start, "end": heading.end}
            for heading in page.headings
        ],
    }

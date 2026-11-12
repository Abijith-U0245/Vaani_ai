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
class SectionSpan:
    start: int
    end: int
    section: str


@dataclass(frozen=True)
class PageText:
    page_index: int
    page_number: int
    text: str
    headings: tuple[SectionHeading, ...]
    dropped_blocks: int = 0
    section: str = "unknown"
    section_spans: tuple[SectionSpan, ...] = ()
    running_headers: tuple[str, ...] = ()


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
    r"^Appendix\s+(?P<number>[A-Z](?:\.\d+)*)(?:\.\s*|\s+)(?P<title>[A-Za-z][A-Za-z0-9 :,'()\-–—/&]+)$",
    re.IGNORECASE,
)
_APPENDIX_ONLY = re.compile(r"^(?:Appendix\s+)?(?P<number>[A-Z])\.?$", re.IGNORECASE)
_COMMON_HEADING = re.compile(
    r"^(?:ABSTRACT|INTRODUCTION|CONCLUSION(?:S)?|REFERENCES|ACKNOWLEDGMENTS?)$",
    re.IGNORECASE,
)
_CAPTION = re.compile(r"^(?:FIGURE|TABLE)\s+\d+\b", re.IGNORECASE)
_UNIT_LINE = re.compile(r"^\d+(?:\.\d+)?\s+(?:HZ|KHZ|MHZ|GB|MB|KB|MS|S)$", re.IGNORECASE)
_TABLE_MARKERS = frozenset("●◐✓✗")
_EQUATION_MARKERS = frozenset("∑√ℰℳ𝐿")


def _block_from_raw(
    raw: dict[str, Any],
    excluded_headers: frozenset[str] = frozenset(),
) -> _Block | None:
    lines: list[_Line] = []
    for raw_line in raw.get("lines", []):
        spans = raw_line.get("spans", [])
        if not spans:
            continue
        raw_line_text = "".join(str(span.get("text", "")) for span in spans).strip()
        if raw_line_text in excluded_headers:
            continue
        dominant_span = max(spans, key=lambda span: float(span.get("size", 0.0)))
        dominant_baseline = float(dominant_span.get("origin", (0.0, 0.0))[1])
        text_parts: list[str] = []
        index = 0
        while index < len(spans):
            span = spans[index]
            span_text = str(span.get("text", ""))
            is_superscript = bool(int(span.get("flags", 0)) & 1)
            if not is_superscript or not span_text.strip():
                text_parts.append(span_text)
                index += 1
                continue
            previous = "".join(text_parts).rstrip()
            run: list[str] = []
            while index < len(spans):
                candidate = spans[index]
                candidate_text = str(candidate.get("text", ""))
                if not (int(candidate.get("flags", 0)) & 1):
                    break
                run.append(candidate_text)
                index += 1
            value = "".join(run)
            next_text = (
                str(spans[index].get("text", "")) if index < len(spans) else ""
            )
            run_origins = [
                float(candidate.get("origin", (0.0, 0.0))[1])
                for candidate in spans[index - len(run) : index]
            ]
            clearly_raised = (
                run_origins
                and dominant_baseline - min(run_origins) >= 1.5
                and len(value.strip()) < 8
            )
            clearly_lowered = (
                run_origins
                and max(run_origins) - dominant_baseline >= 1.5
                and max(
                    float(candidate.get("size", 0.0))
                    for candidate in spans[index - len(run) : index]
                )
                < float(dominant_span.get("size", 0.0)) * 0.9
            )
            in_math_italic = any(
                0x1D400 <= ord(character) <= 0x1D7FF
                for character in value
            )
            if (
                previous
                and previous[-1].isalpha()
                and value.strip().isdigit()
                and next_text.lstrip().startswith(".")
            ):
                text_parts.append(value)
                continue
            if (not clearly_raised and not clearly_lowered) or in_math_italic:
                text_parts.append(value)
                continue
            if clearly_raised and value.strip().isdigit() and previous.endswith("="):
                text_parts.append(value)
                continue
            leading = value[: len(value) - len(value.lstrip())]
            trailing = value[len(value.rstrip()) :]
            core = value.strip()
            marker = "^" if clearly_raised else "_"
            text_parts.append(f"{leading}{marker}{{{core}}}{trailing}")
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
    if any(marker in block.text for marker in _TABLE_MARKERS | _EQUATION_MARKERS):
        return False
    short_lines = sum(len(line.text.strip()) <= 2 for line in block.lines)
    return (
        short_lines >= 1
        and short_lines / len(block.lines) >= 0.6
        and not any(line.bold for line in block.lines)
    )


def _ordered_blocks(
    page: pymupdf.Page,
    excluded_headers: frozenset[str] = frozenset(),
) -> tuple[list[_Block], int]:
    blocks = [
        block
        for raw in page.get_text("dict")["blocks"]
        if (block := _block_from_raw(raw, excluded_headers)) is not None
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


def _outline_title(title: str) -> str:
    return re.sub(
        r"^(?:Appendix\s+)?(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip().lower()


def _filter_headings(
    headings: tuple[SectionHeading, ...],
    page_number: int,
    outline: set[tuple[str, int]],
) -> tuple[SectionHeading, ...]:
    filtered: list[SectionHeading] = []
    for heading in headings:
        title = heading.title.strip()
        if title[:1].isdigit():
            if (_outline_title(title), page_number) not in outline:
                continue
        elif title.lower() not in {"abstract", "references"} and not title.lower().startswith(
            "appendix "
        ):
            continue
        elif title.lower().startswith("appendix ") and (
            _outline_title(title), page_number
        ) not in outline:
            continue
        filtered.append(heading)
    return tuple(filtered)


def extract_pdf(path: Path | str) -> list[PageText]:
    """Extract pages with normalized text and provenance offsets.

    ``start``/``end`` offsets refer to this processed page text, including
    superscripts represented as ``^{...}``, not the raw PDF text layer.
    """
    pages: list[PageText] = []
    with pymupdf.open(Path(path)) as document:
        header_counts: dict[tuple[str, float], int] = {}
        for page in document:
            seen: set[tuple[str, float]] = set()
            for raw in page.get_text("dict")["blocks"]:
                for line in raw.get("lines", []):
                    text = "".join(
                        str(span.get("text", "")) for span in line.get("spans", [])
                    ).strip()
                    if text and float(line["bbox"][1]) <= 100:
                        seen.add((text, round(float(line["bbox"][1]), 1)))
            for key in seen:
                header_counts[key] = header_counts.get(key, 0) + 1
        repeated_headers = {
            text
            for (text, _), count in header_counts.items()
            if len(document) > 1 and count > len(document) / 2
        }
        outline = {
            (_outline_title(title), page)
            for _, title, page in document.get_toc()
        }
        active_section = "unknown"
        for page_index, page in enumerate(document):
            blocks, dropped = _ordered_blocks(
                page, frozenset(repeated_headers)
            )
            text = "\n".join(block.text for block in blocks)
            headings = _filter_headings(
                _headings(blocks, text), page_index + 1, outline
            )
            section_spans: list[SectionSpan] = []
            cursor = 0
            for heading in headings:
                if cursor < heading.start:
                    section_spans.append(
                        SectionSpan(cursor, heading.start, active_section)
                    )
                active_section = heading.title
                cursor = heading.start
            if cursor < len(text):
                section_spans.append(SectionSpan(cursor, len(text), active_section))
            if headings:
                active_section = headings[-1].title
            pages.append(
                PageText(
                    page_index=page_index,
                    page_number=page_index + 1,
                    text=text,
                    headings=headings,
                    dropped_blocks=dropped,
                    section=active_section,
                    section_spans=tuple(section_spans),
                    running_headers=tuple(sorted(repeated_headers)),
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
        "section_spans": [
            {
                "start": span.start,
                "end": span.end,
                "section": span.section,
            }
            for span in page.section_spans
        ],
        "running_headers": list(page.running_headers),
    }

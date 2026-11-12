"""Print a compact, human-readable extraction spot check for downloaded papers."""

from __future__ import annotations

import argparse
import contextlib
import random
import sys
from pathlib import Path

import pymupdf

from relay.ingest.download import PAPERS
from relay.ingest.extract import _Line, _is_heading, _ordered_blocks, extract_pdf


def _plain_heading(title: str) -> str:
    import re

    return re.sub(
        r"^(?:Appendix\s+)?(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )


def _outline_comparison(document: pymupdf.Document, pages: list) -> None:
    toc = document.get_toc()
    print("  outline:")
    if not toc:
        print("    none")
        return
    for level, title, page in toc:
        print(f"    level={level} title={title!r} page={page}")
    detected = {
        (_plain_heading(heading.title).lower(), page.page_number)
        for page in pages
        for heading in page.headings
    }
    outline = {
        (_plain_heading(title).lower(), page)
        for level, title, page in toc
    }
    numbered_outline = {
        (_plain_heading(title).lower(), page)
        for level, title, page in toc
        if title[:1].isdigit()
    }
    missing = sorted(numbered_outline - detected)
    extras = sorted(detected - outline)
    recall = (
        (len(numbered_outline) - len(missing)) / len(numbered_outline)
        if numbered_outline
        else 1.0
    )
    print(f"  numbered_outline_recall: {recall:.3f}")
    print(f"  headings_missing_from_detection: {missing}")
    print(f"  detected_headings_not_in_numbered_outline: {extras}")


def _print_regression_lines(document: pymupdf.Document) -> None:
    targets = ("3.2.1", "3.2.2", "3.3.1", "3.3.2", "3.4.1", "3.4.2", "3.4.3")
    print("  Moshi regression heading diagnostics:")
    for page_number, page in enumerate(document, start=1):
        sizes = [
            float(span["size"])
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        dominant = sorted(sizes)[len(sizes) // 2] if sizes else 0.0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line.get("spans", []))
                if not text.startswith(targets):
                    continue
                size = max(float(span["size"]) for span in line["spans"])
                bold = any(int(span["flags"]) & 16 for span in line["spans"])
                accepted = _is_heading(
                    _Line(text=text, size=size, bold=bold), dominant
                )
                rule = "accepted" if accepted else (
                    "font-size/bold gate (single-level headings require emphasis)"
                )
                print(
                    f"    page={page_number} size={size:.2f} bold={bold} "
                    f"text={text!r} rule={rule}"
                )


def _print_false_positive_diagnostics(
    document: pymupdf.Document, paper_id: str
) -> None:
    targets = (
        {23: ("kbps",)}
        if paper_id == "2410.00037"
        else {36: ("Cross Attention", "KV"), 17: ("x more data efficient than isotropic model",)}
    )
    print("  false-positive diagnostics:")
    for page_number, needles in targets.items():
        page = document[page_number - 1]
        sizes = [
            float(span["size"])
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        dominant = sorted(sizes)[len(sizes) // 2] if sizes else 0.0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line.get("spans", []))
                if not any(needle.lower() in text.lower() for needle in needles):
                    continue
                size = max(float(span["size"]) for span in line["spans"])
                bold = any(int(span["flags"]) & 16 for span in line["spans"])
                bbox = tuple(round(value, 2) for value in block["bbox"])
                print(
                    f"    page={page_number} size={size:.2f} bold={bold} "
                    f"bbox={bbox} text={text!r} "
                    f"rule=outline-filtered-unless-numbered-match"
                )


def _print_superscript_evidence(document: pymupdf.Document) -> None:
    for page_number in (6, 8):
        raw_lines = [
            line
            for block in document[page_number - 1].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if "O(n" in "".join(span["text"] for span in line["spans"])
            or "10" in "".join(span["text"] for span in line["spans"])
        ]
        print(f"  superscript_page_{page_number}_before_after:")
        processed = extract_pdf_from_document_page(document, page_number)
        for line in raw_lines[:12]:
            before = "".join(span["text"] for span in line["spans"])
            print(f"    before={before!r}")
        for line in processed.splitlines():
            if "O(n" in line or "10^{" in line:
                print(f"    after={line!r}")


def extract_pdf_from_document_page(
    document: pymupdf.Document, page_number: int
) -> str:
    """Extract one already-open document page using the public extractor path."""
    page = document[page_number - 1]
    blocks, _ = _ordered_blocks(page)
    return "\n".join(block.text for block in blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir", type=Path, default=Path("data/pdfs"), help="PDF directory"
    )
    parser.add_argument(
        "--out", type=Path, help="write UTF-8 output to this path instead of stdout"
    )
    args = parser.parse_args()
    output = args.out.open("w", encoding="utf-8") if args.out else None
    stream = output or sys.stdout
    if hasattr(stream, "reconfigure") and output is None:
        stream.reconfigure(encoding="utf-8", errors="replace")
    rng = random.Random(20261111)

    try:
        with contextlib.redirect_stdout(stream):
            for paper_id, version in PAPERS:
                path = args.pdf_dir / f"{paper_id}v{version}.pdf"
                print(f"Paper: {paper_id}v{version}")
                if not path.exists():
                    print(f"  missing: {path}")
                    continue
                pages = extract_pdf(path)
                print(f"  page_count: {len(pages)}")
                headings = [
                    f"p{page.page_number}: {heading.title}"
                    for page in pages
                    for heading in page.headings
                ]
                print(
                    f"  dropped_blocks_by_page: "
                    f"{[page.dropped_blocks for page in pages]}"
                )
                print(f"  headings: {headings}")
                probes = {
                    "2410.00037": (45, 59),
                    "2507.07955": (30, 38),
                }
                if paper_id in probes:
                    for page_number in probes[paper_id]:
                        print(
                            f"  section_page_{page_number}: "
                            f"{pages[page_number - 1].section}"
                        )
                with pymupdf.open(path) as document:
                    _outline_comparison(document, pages)
                    if paper_id == "2410.00037":
                        _print_regression_lines(document)
                    if paper_id == "2410.00037":
                        _print_false_positive_diagnostics(document, paper_id)
                    if paper_id == "2507.07955":
                        _print_false_positive_diagnostics(document, paper_id)
                    if paper_id == "1706.03762":
                        _print_superscript_evidence(document)
                chosen = sorted(rng.sample(range(len(pages)), min(3, len(pages))))
                for page_index in chosen:
                    print(f"  full_text_page_{pages[page_index].page_number}:")
                    print(pages[page_index].text)
    finally:
        if output is not None:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

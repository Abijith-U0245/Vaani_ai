from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from relay.ingest.chunk import chunk_pages
from relay.ingest.download import download_papers, verify_arxiv_version
from relay.ingest.extract import PageText, SectionHeading

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_version_assertion() -> None:
    xml = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/1706.03762v7</id><title>Test paper</title></entry>
    </feed>"""
    assert verify_arxiv_version(
        "1706.03762", 7, opener=lambda *args, **kwargs: _Response(xml)
    ) is None


def test_version_mismatch_raises() -> None:
    xml = b"""<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/1706.03762v6</id><title>Test paper</title></entry>
    </feed>"""
    with pytest.raises(ValueError, match="version mismatch"):
        verify_arxiv_version(
            "1706.03762", 7, opener=lambda *args, **kwargs: _Response(xml)
        )


def test_saved_hash_matches_manifest(tmp_path: Path) -> None:
    api = b"""<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/1706.03762v7</id><title>Test paper</title></entry>
    </feed>"""
    pdf = b"synthetic pdf bytes"

    def opener(url: str, **kwargs: object) -> _Response:
        return _Response(api if "api/query" in url else pdf)

    download_papers(tmp_path, papers=(("1706.03762", 7),), opener=opener)
    saved = tmp_path / "1706.03762v7.pdf"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == manifest[0]["sha256"]


def test_changed_saved_hash_raises(tmp_path: Path) -> None:
    api = b"""<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/abs/1706.03762v7</id><title>Test paper</title></entry>
    </feed>"""
    pdf = b"synthetic pdf bytes"

    def opener(url: str, **kwargs: object) -> _Response:
        return _Response(api if "api/query" in url else pdf)

    download_papers(tmp_path, papers=(("1706.03762", 7),), opener=opener)
    (tmp_path / "1706.03762v7.pdf").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="saved PDF hash changed"):
        download_papers(tmp_path, papers=(("1706.03762", 7),), opener=opener)


def test_chunk_offset_round_trip() -> None:
    text = "Introduction\n" + ("0123456789 " * 20)
    page = PageText(
        page_index=3,
        page_number=4,
        text=text,
        headings=(SectionHeading("Introduction", 0, 12),),
    )
    chunks = chunk_pages([page], "1706.03762", chunk_size=37, overlap=9)
    assert chunks
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert all(
        chunk.page == 3
        and chunk.page_number == 4
        and chunk.paper_id == "1706.03762"
        for chunk in chunks
    )


def test_chunking_is_deterministic() -> None:
    page = PageText(page_index=0, page_number=1, text="abcdefghij" * 10, headings=())
    first = chunk_pages([page], "paper", chunk_size=19, overlap=4)
    second = chunk_pages([page], "paper", chunk_size=19, overlap=4)
    assert first == second


def test_two_column_pdf_order_and_page_index(tmp_path: Path) -> None:
    path = tmp_path / "columns.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_textbox((40, 40, 280, 100), "INTRODUCTION\nleft column")
    page.insert_textbox((320, 40, 560, 100), "right column")
    document.save(path)
    document.close()

    from relay.ingest.extract import extract_pdf

    pages = extract_pdf(path)
    assert len(pages) == 1
    assert pages[0].page_index == 0
    assert pages[0].page_number == 1
    assert pages[0].text.index("left column") < pages[0].text.index("right column")
    assert "INTRODUCTION" in pages[0].text


def test_number_then_title_heading_merge() -> None:
    page = PageText(
        page_index=0,
        page_number=1,
        text="3.1\nEncoder and Decoder Stacks\nBody text",
        headings=(),
    )
    from relay.ingest.extract import _headings

    blocks = [
        type("Block", (), {
            "lines": (
                type("Line", (), {"text": "3.1", "size": 12.0, "bold": True})(),
                type("Line", (), {
                    "text": "Encoder and Decoder Stacks",
                    "size": 12.0,
                    "bold": True,
                })(),
                type("Line", (), {"text": "Body text", "size": 10.0, "bold": False})(),
            )
        })()
    ]
    headings = _headings(blocks, page.text)
    assert [heading.title for heading in headings] == ["3.1 Encoder and Decoder Stacks"]


def test_body_sentence_is_not_heading() -> None:
    page = PageText(
        page_index=0,
        page_number=1,
        text="2014 English-French dataset consisting of 36M sentences",
        headings=(),
    )
    from relay.ingest.extract import extract_pdf
    path = Path("tests") / "_body_sentence_fixture.pdf"
    document = pymupdf.open()
    pdf_page = document.new_page()
    pdf_page.insert_text((40, 40), page.text)
    document.save(path)
    document.close()
    try:
        assert extract_pdf(path)[0].headings == ()
    finally:
        path.unlink()


def test_figure_block_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "figure.pdf"
    document = pymupdf.open()
    pdf_page = document.new_page()
    pdf_page.insert_text((40, 40), "Body text")
    pdf_page.insert_textbox((300, 100, 400, 220), "h\n_\ne\nn")
    document.save(path)
    document.close()
    from relay.ingest.extract import extract_pdf

    extracted = extract_pdf(path)[0]
    assert extracted.dropped_blocks >= 1
    assert "Body text" in extracted.text
    assert "h\n_\ne\nn" not in extracted.text


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/1706.03762v7.pdf").exists(),
    reason="downloaded paper PDFs are not present",
)
def test_attention_unicode_and_superscripts() -> None:
    from relay.ingest.extract import extract_pdf

    pages = extract_pdf(PROJECT_ROOT / "data/pdfs/1706.03762v7.pdf")
    assert "•" in pages[4].text
    assert "∈" in pages[4].text
    assert "10^{20}" in pages[7].text
    assert "10^{18}" in pages[7].text
    assert "O(n^{2}" in pages[5].text


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/1706.03762v7.pdf").exists(),
    reason="downloaded paper PDF is not present",
)
def test_attention_page_five_preserves_bullet_and_membership() -> None:
    from relay.ingest.extract import extract_pdf

    text = extract_pdf(PROJECT_ROOT / "data/pdfs/1706.03762v7.pdf")[4].text
    assert "•" in text
    assert "∈" in text


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf").exists()
    or not (PROJECT_ROOT / "data/pdfs/2507.07955v2.pdf").exists(),
    reason="downloaded paper PDFs are not present",
)
def test_false_positive_headings_are_not_detected() -> None:
    from relay.ingest.extract import extract_pdf

    moshi = extract_pdf(PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf")
    hnet = extract_pdf(PROJECT_ROOT / "data/pdfs/2507.07955v2.pdf")
    assert "1.5kbps" not in {h.title for h in moshi[22].headings}
    hnet_p36 = {h.title for h in hnet[35].headings}
    hnet_p17 = {h.title for h in hnet[16].headings}
    assert "KV" not in hnet_p36
    assert "Cross Attention" not in hnet_p36
    assert "3.6x more data efficient than isotropic model" not in hnet_p17


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf").exists()
    or not (PROJECT_ROOT / "data/pdfs/2507.07955v2.pdf").exists(),
    reason="downloaded paper PDFs are not present",
)
def test_reference_and_appendix_boundary_sections() -> None:
    from relay.ingest.extract import extract_pdf

    moshi = extract_pdf(PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf")
    hnet = extract_pdf(PROJECT_ROOT / "data/pdfs/2507.07955v2.pdf")
    assert moshi[44].section == "References"
    assert hnet[30].section == "Appendix A Related Work"
    assert {heading.title for heading in moshi[52].headings} >= {
        "Appendix A. Additional ablation on Mimi codec",
        "Appendix B. Audio matching and deduplication",
    }
    assert {heading.title for heading in hnet[34].headings} >= {
        "Appendix B FLOPs Computation",
        "Appendix C Learning Rate Modulation",
    }


@pytest.mark.skipif(
    not all(
        (PROJECT_ROOT / f"data/pdfs/{name}").exists()
        for name in ("1706.03762v7.pdf", "2410.00037v1.pdf", "2507.07955v2.pdf")
    ),
    reason="downloaded paper PDFs are not present",
)
def test_non_ascii_counts_match_except_allowlisted_figure_drops() -> None:
    from relay.ingest.extract import extract_pdf

    # These are isolated diagram/equation glyphs removed by the figure filter.
    allowed = {
        # Moshi p14 contains private-use figure glyphs from the PDF's diagram.
        "2410.00037v1.pdf": {
            14: {"U+F8F3": 1, "U+F8F4": 1},
            # These are diagram labels, not prose or equations.
            56: {"U+23B5": 2},
            57: {"U+23B5": 1},
        },
        "2507.07955v2.pdf": {
            # These isolated marks belong to the figure on page 3.
            3: {"U+0302": 1, "U+0305": 2},
            # This standalone ellipsis is part of the figure on page 38.
            38: {"U+2026": 1},
        },
    }
    for name in ("1706.03762v7.pdf", "2410.00037v1.pdf", "2507.07955v2.pdf"):
        with pymupdf.open(PROJECT_ROOT / "data/pdfs" / name) as document:
            raw_pages = [
                Counter(ch for ch in page.get_text("text") if ord(ch) > 127)
                for page in document
            ]
        processed_pages = extract_pdf(PROJECT_ROOT / "data/pdfs" / name)
        differences = {}
        for page_number, (raw, processed) in enumerate(
            zip(raw_pages, processed_pages), start=1
        ):
            page_diff = {
                f"U+{ord(ch):04X}": raw[ch] - processed.text.count(ch)
                for ch in raw
                if raw[ch] != processed.text.count(ch)
            }
            if page_diff:
                differences[page_number] = page_diff
        assert differences == allowed.get(name, {})


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf").exists(),
    reason="downloaded paper PDF is not present",
)
def test_negative_exponents_and_footnote_markers() -> None:
    from relay.ingest.extract import extract_pdf

    pages = extract_pdf(PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf")
    all_text = "\n".join(page.text for page in pages)
    assert "10^{−6}" in all_text
    assert "10^{−}^{6}" not in all_text

    attention = extract_pdf(PROJECT_ROOT / "data/pdfs/1706.03762v7.pdf")[7].text
    assert "GPU 5." in attention
    assert "GPU ^{5}." not in attention


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf").exists()
    or not (PROJECT_ROOT / "data/pdfs/2507.07955v2.pdf").exists(),
    reason="downloaded paper PDFs are not present",
)
def test_appendix_sections_do_not_inherit_references() -> None:
    from relay.ingest.extract import extract_pdf

    moshi = extract_pdf(PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf")
    hnet = extract_pdf(PROJECT_ROOT / "data/pdfs/2507.07955v2.pdf")
    assert moshi[44].section == "References"
    assert moshi[58].section.startswith("Appendix")
    assert hnet[29].section == "References"
    assert hnet[37].section.startswith("Appendix")
    assert all(page.section == "References" for page in moshi[41:52])
    assert moshi[52].section.startswith("Appendix")
    assert all(page.section == "References" for page in hnet[24:30])
    assert hnet[30].section.startswith("Appendix")


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf").exists(),
    reason="downloaded paper PDF is not present",
)
def test_moshi_numbered_outline_headings_are_detected() -> None:
    from relay.ingest.extract import extract_pdf

    pages = extract_pdf(PROJECT_ROOT / "data/pdfs/2410.00037v1.pdf")
    detected = {heading.title for page in pages for heading in page.headings}
    for expected in (
        "3.2.1 Architecture",
        "3.2.2 Pre-training data filtering",
        "3.3.1 Architecture",
        "3.3.2 Learning semantic-acoustic tokens with a split RVQ",
        "3.4.1 Hierarchical autoregressive modeling with RQ-Transformer",
        "3.4.2 Audio modeling",
        "3.4.3 Multi-stream modeling",
    ):
        assert expected in detected

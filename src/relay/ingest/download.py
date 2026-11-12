"""Download and verify the fixed paper inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PAPERS: tuple[tuple[str, int], ...] = (
    ("1706.03762", 7),
    ("2410.00037", 1),
    ("2507.07955", 2),
)
ARXIV_API_URL = "https://export.arxiv.org/api/query?id_list={paper_id}v{version}"
PDF_URL = "https://export.arxiv.org/pdf/{paper_id}v{version}.pdf"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_VERSION_RE = re.compile(r"/abs/[^/]+v(?P<version>\d+)$")


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    version: int


def _entry_metadata(api_xml: bytes) -> tuple[int, str]:
    """Return the version and title in the API entry."""
    root = ET.fromstring(api_xml)
    entry = root.find("atom:entry", _ARXIV_NS)
    if entry is None:
        raise ValueError("arXiv API response contains no entry")
    identifier = entry.findtext("atom:id", namespaces=_ARXIV_NS)
    if not identifier:
        raise ValueError("arXiv API entry contains no identifier")
    match = _VERSION_RE.search(identifier.strip())
    if match is None:
        raise ValueError(f"arXiv identifier has no version: {identifier!r}")
    title = entry.findtext("atom:title", namespaces=_ARXIV_NS)
    if not title:
        raise ValueError("arXiv API entry contains no title")
    return int(match.group("version")), " ".join(title.split())


def verify_arxiv_version(
    paper_id: str,
    version: int,
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> None:
    """Verify that arXiv metadata resolves to the requested version."""
    if version < 1:
        raise ValueError("arXiv versions must be positive")
    url = ARXIV_API_URL.format(paper_id=paper_id, version=version)
    with opener(url, timeout=30) as response:
        reported_version, _ = _entry_metadata(response.read())
    if reported_version != version:
        raise ValueError(
            f"arXiv version mismatch for {paper_id}: expected v{version}, "
            f"metadata reports v{reported_version}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_papers(
    output_dir: Path | str = Path("data/pdfs"),
    *,
    papers: tuple[tuple[str, int], ...] = PAPERS,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> list[dict[str, str | int]]:
    """Download, verify, hash, and manifest the exact configured paper versions."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    previous_manifest: dict[str, dict[str, str | int]] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"manifest must contain a list: {manifest_path}")
        for item in loaded:
            if not isinstance(item, dict) or "paper_id" not in item:
                raise ValueError(f"invalid manifest entry: {item!r}")
            previous_manifest[str(item["paper_id"])] = item

    manifest: list[dict[str, str | int]] = []
    summary: list[dict[str, str | int]] = []
    for paper_id, version in papers:
        api_url = ARXIV_API_URL.format(paper_id=paper_id, version=version)
        with opener(api_url, timeout=30) as response:
            reported_version, title = _entry_metadata(response.read())
        if reported_version != version:
            raise ValueError(
                f"arXiv version mismatch for {paper_id}: expected v{version}, "
                f"metadata reports v{reported_version}"
            )

        url = PDF_URL.format(paper_id=paper_id, version=version)
        target = destination / f"{paper_id}v{version}.pdf"
        previous = previous_manifest.get(paper_id)
        if previous is not None and target.exists():
            digest = _sha256(target)
            if digest != previous.get("sha256"):
                raise ValueError(
                    f"saved PDF hash changed for {paper_id}: manifest has "
                    f"{previous.get('sha256')}, file has {digest}"
                )
            manifest.append(
                {
                    "paper_id": paper_id,
                    "url": url,
                    "version": version,
                    "api_reported_version": reported_version,
                    "title": title,
                    "sha256": digest,
                    "file_size_bytes": target.stat().st_size,
                    "retrieved_at": previous.get(
                        "retrieved_at", datetime.now(timezone.utc).isoformat()
                    ),
                }
            )
            summary.append(
                {"paper": paper_id, "version": reported_version, "sha256": digest,
                 "status": "already verified"}
            )
            continue

        temporary = target.with_suffix(".pdf.tmp")
        try:
            with opener(url, timeout=60) as response, temporary.open("wb") as stream:
                stream.write(response.read())
            digest = _sha256(temporary)
            if previous is not None and digest != previous.get("sha256"):
                raise ValueError(
                    f"downloaded PDF hash changed for {paper_id}: manifest has "
                    f"{previous.get('sha256')}, download has {digest}"
                )
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        manifest.append(
            {
                "paper_id": paper_id,
                "url": url,
                "version": version,
                "api_reported_version": reported_version,
                "title": title,
                "sha256": digest,
                "file_size_bytes": target.stat().st_size,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        summary.append(
            {"paper": paper_id, "version": reported_version, "sha256": digest,
             "status": "downloaded"}
        )

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("paper\tversion\tsha256\tstatus")
    for item in summary:
        print(
            f"{item['paper']}\t{item['version']}\t{item['sha256']}\t{item['status']}"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the exact Relay arXiv paper versions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pdfs"),
        help="directory for PDFs and manifest (default: data/pdfs)",
    )
    args = parser.parse_args()
    download_papers(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

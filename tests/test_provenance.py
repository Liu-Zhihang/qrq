"""The vendored-source manifest matches the files in the repository."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "vendored_sources.json"
BACKEND_SHA256 = "457dee1a1b312a8bbc48f2b848d00112e18d2f12324270d1536e44219e244e05"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_hashes():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["files"]
    assert entries, "manifest lists no files"
    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert sha256(path) == entry["vendored_sha256"], entry["path"]
        if not entry["edits"]:
            assert entry["vendored_sha256"] == entry["original_sha256"], entry["path"]
    vendor_dir = ROOT / "src" / "qrq" / "_vendor"
    listed = {entry["path"] for entry in entries}
    for path in vendor_dir.glob("*.py"):
        if path.name != "__init__.py":
            assert path.relative_to(ROOT).as_posix() in listed


def test_backend_source_is_byte_identical():
    assert sha256(ROOT / "cpp" / "trajectory_trees_3d.cpp") == BACKEND_SHA256

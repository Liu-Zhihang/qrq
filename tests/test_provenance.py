"""The vendored-source manifest matches the files in the repository.

Rule: every vendored file is pinned by its full SHA-256, every difference from
the original is listed as a classified edit, and the code bodies (all
function and class definitions, without docstrings and comments) equal the
original unless an edit of a body-changing class is listed.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "vendored_sources.json"
BACKEND_SHA256 = "457dee1a1b312a8bbc48f2b848d00112e18d2f12324270d1536e44219e244e05"
BODY_EDIT_CLASSES = {"hard-coded prices replaced by keyword parameters with identical defaults"}
"""Edit classes that may change a function body."""
FILE_NAME = "file name"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_fingerprint(source: str) -> str:
    """SHA-256 of the top-level function and class definitions of a module.

    Docstrings (located with ``ast``) and comments (located with
    ``tokenize``) are removed; the remaining text keeps its exact spelling,
    only trailing whitespace and the resulting blank lines are dropped. The
    fingerprint therefore ignores documentation wording and module-level
    statements such as imports, and is stable across Python versions.
    """
    tree = ast.parse(source)
    lines = [list(line) for line in source.splitlines(keepends=True)]

    def column(lineno: int, byte_offset: int) -> int:
        return len("".join(lines[lineno - 1]).encode()[:byte_offset].decode())

    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                spans.append((first.lineno, column(first.lineno, first.col_offset),
                              first.end_lineno, column(first.end_lineno, first.end_col_offset)))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.append((*token.start, *token.end))
    for start_line, start_col, end_line, end_col in spans:
        for lineno in range(start_line, end_line + 1):
            chars = lines[lineno - 1]
            lo = start_col if lineno == start_line else 0
            hi = end_col if lineno == end_line else len(chars)
            for i in range(lo, hi):
                if chars[i] != "\n":
                    chars[i] = ""
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_line = min([node.lineno] + [d.lineno for d in node.decorator_list])
            for lineno in range(first_line, node.end_lineno + 1):
                text = "".join(lines[lineno - 1]).rstrip()
                if text:
                    kept.append(text)
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()


def test_manifest_hashes_and_edits():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    allowed = set(manifest["allowed_edit_classes"])
    assert BODY_EDIT_CLASSES | {FILE_NAME} <= allowed
    entries = manifest["files"]
    assert entries, "manifest lists no files"
    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert sha256(path) == entry["vendored_sha256"], entry["path"]
        classes = {edit["class"] for edit in entry["edits"]}
        assert classes <= allowed, entry["path"]
        assert all(edit["detail"] for edit in entry["edits"]), entry["path"]
        if classes <= {FILE_NAME}:
            assert entry["vendored_sha256"] == entry["original_sha256"], entry["path"]
        if (FILE_NAME in classes) != (path.name != entry["original_file"]):
            raise AssertionError(f"{entry['path']}: file-name edit and file name disagree")
        if path.suffix == ".py":
            assert body_fingerprint(path.read_text(encoding="utf-8")) == entry["body_sha256"], entry["path"]
            if not classes & BODY_EDIT_CLASSES:
                assert entry["body_sha256"] == entry["original_body_sha256"], entry["path"]
    vendor_dir = ROOT / "src" / "qrq" / "_vendor"
    listed = {entry["path"] for entry in entries}
    for path in vendor_dir.glob("*.py"):
        if path.name != "__init__.py":
            assert path.relative_to(ROOT).as_posix() in listed


def test_body_fingerprint_ignores_documentation_only():
    base = 'import a\n\ndef f(x):\n    """Doc."""\n    return x + 1  # note\n\nclass C:\n    "Doc."\n    y = 2\n'
    reworded = 'import b\n\ndef f(x):\n    """Other words,\n    two lines."""\n    return x + 1\n\nclass C:\n    y = 2\n'
    changed = base.replace("x + 1", "x + 2")
    assert body_fingerprint(base) == body_fingerprint(reworded)
    assert body_fingerprint(base) != body_fingerprint(changed)


def test_backend_source_is_byte_identical():
    assert sha256(ROOT / "cpp" / "trajectory_trees_3d.cpp") == BACKEND_SHA256

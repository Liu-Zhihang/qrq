"""Locate sealed evidence passed through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def env_dir(name: str) -> Path:
    """Return the directory named by ``name`` or skip the calling test."""
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"{name} does not point to a directory")
    return path

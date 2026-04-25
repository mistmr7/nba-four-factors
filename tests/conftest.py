"""Shared pytest fixtures and vcrpy configuration.

Cassettes are stored at tests/fixtures/cassettes/ (locked, Section 2.4).
Usage in a test:

    def test_something(recorded):
        with recorded("my_cassette"):
            # HTTP calls here are recorded on first run and replayed after.
            ...

Record mode is "once": cassettes record on first run, replay on subsequent
runs. To re-record, delete the cassette file and rerun. Headers that could
leak session state or vary between machines are filtered.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import vcr
from vcr.record_mode import RecordMode

CASSETTE_DIR = Path(__file__).parent / "fixtures" / "cassettes"

_FILTER_HEADERS = [
    "Authorization",
    "Cookie",
    "Set-Cookie",
    "User-Agent",
    "x-nba-stats-token",
    "x-nba-stats-origin",
]

_VCR = vcr.VCR(
    cassette_library_dir=str(CASSETTE_DIR),
    record_mode=RecordMode.ONCE,
    match_on=["method", "scheme", "host", "port", "path", "query"],
    filter_headers=_FILTER_HEADERS,
    decode_compressed_response=True,
)


@pytest.fixture
def cassettes_dir() -> Path:
    """Absolute path to the cassettes directory."""
    return CASSETTE_DIR


@pytest.fixture
def recorded() -> Callable[[str], Any]:
    """Context-manager factory: `with recorded("name"): ...`.

    Records to tests/fixtures/cassettes/<name>.yaml on first run,
    replays thereafter. Name is a bare cassette stem; .yaml is appended.
    """

    def _use(name: str) -> Any:
        return _VCR.use_cassette(f"{name}.yaml")

    return _use

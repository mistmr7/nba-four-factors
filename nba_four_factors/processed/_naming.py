"""Internal naming helpers shared across the processed package.

Lives in its own module to avoid the obvious circular-import trap: if these
helpers were defined in ``__init__.py``, the submodules that need them
(``schedule.py``, ``pipeline.py``) couldn't import from the package while
``__init__.py`` was still executing its own imports.

These functions implement the Session 10 "human-readable + computer-readable"
convention.  Re-exported publicly from ``__init__.py``.

Implementation note: SeasonType.value mirrors the NBA stats API parameter
strings, which are not internally consistent ("Regular Season" has a space,
"PlayIn" does not).  A computed conversion (.lower() + replace) would give
"playin" instead of "play_in" for PLAY_IN.  Explicit mappings dodge the
quirk and document the contract in one place.
"""

from __future__ import annotations

from ..config import SeasonType

_TO_SNAKE: dict[SeasonType, str] = {
    SeasonType.REGULAR: "regular_season",
    SeasonType.PLAY_IN: "play_in",
    SeasonType.PLAYOFFS: "playoffs",
}

_TO_HUMAN: dict[str, str] = {snake: st.value for st, snake in _TO_SNAKE.items()}


def season_type_to_snake(season_type: SeasonType) -> str:
    """Convert a SeasonType enum member to the snake_case storage form.

    >>> season_type_to_snake(SeasonType.REGULAR)
    'regular_season'
    >>> season_type_to_snake(SeasonType.PLAY_IN)
    'play_in'

    The snake form is used in Parquet column values and processed-layer
    file paths.  It's intentionally consistent (always snake_case, always
    multi-word where natural) even though the underlying enum values are
    not.
    """
    return _TO_SNAKE[season_type]


def snake_to_human(snake: str) -> str:
    """Convert a snake_case season-type back to the SeasonType.value form.

    >>> snake_to_human("regular_season")
    'Regular Season'
    >>> snake_to_human("play_in")
    'PlayIn'

    Inverse of :func:`season_type_to_snake`.  Returns whatever the
    SeasonType enum stores as its value, which mirrors the NBA stats API
    parameter strings (and is therefore inconsistent across season types).
    Used by the manifest layer to round-trip back to the API form.
    """
    return _TO_HUMAN[snake]

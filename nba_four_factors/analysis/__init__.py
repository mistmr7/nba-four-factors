"""Analysis layer: regression analyses on the processed four-factors data.

Public API:

    load_processed
        Read processed Parquets across a season range and season type,
        concatenated into a single DataFrame.

Private modules use the underscore prefix. The public API expands as later
steps in Session 11 land.
"""

from ._loading import read_processed as load_processed

__all__ = ["load_processed"]

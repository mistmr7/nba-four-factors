"""Sequence-model layer (LSTM / GRU) over per-game four-factor histories.

dataset.py builds the training tensors (pure numpy/pandas, runs anywhere).
models.py and train.py use PyTorch and are meant to run where a DL stack and
more compute are available; the sandbox cannot install torch.
"""

from __future__ import annotations

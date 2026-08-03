"""Modeling layer: feature assembly, Vegas seed, and the regression ladder.

These modules are dependency-light (pandas, numpy, scikit-learn) and derive
paths from the file location, so they run under the sandbox interpreter as well
as the project's. The Session 14 plan (docs/Session14_regression_build_plan.md)
is the design of record.
"""

from __future__ import annotations

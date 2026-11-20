"""Unified pluggable evaluation interface, closed-loop runner, and official benchmark scorer."""
from __future__ import annotations

from jevnav.eval.interface import (
    BaseWSINavPolicy,
    JevNavLocalPolicy,
    NavigationAction,
    NavigationObservation,
    OpenAICompatiblePolicy,
    OverviewOnlyPolicy,
    RandomNavPolicy,
    TissuePeakNavPolicy,
)
from jevnav.eval.metrics import score_cam16_predictions, score_wsivqa_predictions

__all__ = [
    "BaseWSINavPolicy",
    "JevNavLocalPolicy",
    "NavigationAction",
    "NavigationObservation",
    "OpenAICompatiblePolicy",
    "OverviewOnlyPolicy",
    "RandomNavPolicy",
    "TissuePeakNavPolicy",
    "score_cam16_predictions",
    "score_wsivqa_predictions",
]

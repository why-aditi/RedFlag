"""
RedFlag Code Review Environment.

An RL environment where an agent reviews code diffs and posts
structured comments identifying bugs, security issues, and logic errors.
"""

from .client import RedflagEnv
from .models import (
    RedflagAction,
    RedflagObservation,
    RedflagState,
    ReviewComment,
    RequestContextAction,
    DiffHunk,
)

__all__ = [
    "RedflagEnv",
    "RedflagAction",
    "RedflagObservation",
    "RedflagState",
    "ReviewComment",
    "RequestContextAction",
    "DiffHunk",
]

"""
Episode State Management for RedFlag.

Tracks all mutable state within a single episode:
  - Step counter
  - Comments posted
  - Context requests used (per-file tracking)
  - Golden issues matched
  - Accumulated rewards
"""

from typing import Dict, List, Set
from uuid import uuid4

try:
    from ..models import GoldenIssue, ReviewComment, TaskConfig, DiffHunk
except ImportError:
    from models import GoldenIssue, ReviewComment, TaskConfig, DiffHunk


class EpisodeState:
    """Mutable state for a single episode."""

    def __init__(self, task_config: TaskConfig, diff_hunks: List[DiffHunk]):
        self.episode_id: str = str(uuid4())
        self.task_config: TaskConfig = task_config
        self.diff_hunks: List[DiffHunk] = diff_hunks

        # Step tracking
        self.step: int = 0
        self.max_steps: int = task_config.max_steps
        self.done: bool = False

        # Comments posted by agent
        self.comments: List[ReviewComment] = []

        # Context requests
        self.context_snippets: Dict[str, str] = {}
        self.context_requests_used: Set[str] = set()  # file keys already requested
        self.context_slots_remaining: int = task_config.context_slots

        # Grading state
        self.matched_issues: Set[int] = set()  # indices into golden_issues
        self.step_rewards: List[float] = []

        # Available context files (loaded at reset)
        self.available_context: Dict[str, str] = {}

    @property
    def total_reward(self) -> float:
        return sum(self.step_rewards)

    @property
    def all_issues_found(self) -> bool:
        return len(self.matched_issues) == len(self.task_config.golden_issues)

    @property
    def max_possible_reward(self) -> float:
        """Compute maximum achievable reward for this task."""
        n_issues = len(self.task_config.golden_issues)
        # Per issue: 0.40 (found) + 0.20 (line) + 0.15 (fix) + 0.05 (severity) = 0.80
        per_issue = 0.80
        # Early termination bonus
        bonus = 0.10
        return n_issues * per_issue + bonus

    @property
    def final_score(self) -> float:
        """Normalized score in [0.0, 1.0]."""
        max_r = self.max_possible_reward
        if max_r <= 0:
            return 0.0
        return max(0.0, min(1.0, self.total_reward / max_r))

    @property
    def success(self) -> bool:
        return self.final_score >= 0.6

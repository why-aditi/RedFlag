"""
Base grader interface for RedFlag.

All graders inherit from BaseGrader and implement the `grade` method.
Output rewards are always clamped to the valid range.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    from ..models import GoldenIssue, ReviewComment
except ImportError:
    from models import GoldenIssue, ReviewComment


@dataclass
class GradeResult:
    """Result of grading a single review comment."""

    reward: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    matched_issue_idx: Optional[int] = None  # index into golden_issues list
    is_false_positive: bool = False
    is_duplicate: bool = False


class BaseGrader(ABC):
    """
    Abstract base class for graders.

    Subclasses implement `grade()` which evaluates a single comment
    against the golden issue set.
    """

    @abstractmethod
    def grade(
        self,
        comment: ReviewComment,
        golden_issues: List[GoldenIssue],
        already_matched: Set[int],
        line_tolerance: int = 3,
    ) -> GradeResult:
        """
        Grade a single review comment.

        Args:
            comment: The agent's review comment.
            golden_issues: List of golden issues for the task.
            already_matched: Set of golden issue indices already matched.
            line_tolerance: Allowed line deviation for proximity matching.

        Returns:
            GradeResult with reward breakdown.
        """
        ...

    @staticmethod
    def clamp_reward(value: float, lo: float = -0.5, hi: float = 1.0) -> float:
        """Clamp a reward value to the valid range."""
        return max(lo, min(hi, value))

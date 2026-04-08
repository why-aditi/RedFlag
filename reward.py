"""
Reward computation for RedFlag.

Per-step reward table:
  +0.40  Issue found (type match, 2+ keywords)
  +0.20  Correct line cited (within tolerance)
  +0.15  Valid fix suggested (fix keyword match)
  +0.05  Severity correctly labeled
  -0.10  False positive (comment on non-issue line)
  -0.05  Duplicate issue (same line commented twice)
   0.00  Context request (neutral)
  -0.10  Duplicate context request for same file
  +0.10  Early termination bonus (all issues found)

Final score: clamp(sum(rewards) / max_possible, 0.0, 1.0)
"""

from typing import Dict, List, Set, Tuple

try:
    from .models import GoldenIssue, ReviewComment, TaskConfig
    from .graders.base import BaseGrader, GradeResult
    from .graders.keyword_grader import KeywordGrader
    from .graders.semantic_grader import SemanticGrader
    from .graders.false_positive_detector import is_false_positive, is_duplicate_comment
except ImportError:
    from models import GoldenIssue, ReviewComment, TaskConfig
    from graders.base import BaseGrader, GradeResult
    from graders.keyword_grader import KeywordGrader
    from graders.semantic_grader import SemanticGrader
    from graders.false_positive_detector import is_false_positive, is_duplicate_comment


def get_grader(grader_type: str) -> BaseGrader:
    """Get the appropriate grader for a task."""
    if grader_type == "semantic":
        return SemanticGrader()
    return KeywordGrader()


def compute_comment_reward(
    comment: ReviewComment,
    task_config: TaskConfig,
    previous_comments: List[ReviewComment],
    matched_issues: Set[int],
    grader: BaseGrader,
) -> Tuple[float, Dict[str, float], GradeResult]:
    """
    Compute reward for a single comment action.

    Returns:
        (total_reward, breakdown_dict, grade_result)
    """
    breakdown: Dict[str, float] = {}
    reward = 0.0

    # Check for duplicate first
    if is_duplicate_comment(comment, previous_comments):
        breakdown["duplicate"] = -0.05
        return -0.05, breakdown, GradeResult(reward=-0.05, is_duplicate=True)

    # Grade the comment against golden issues
    grade = grader.grade(
        comment=comment,
        golden_issues=task_config.golden_issues,
        already_matched=matched_issues,
        line_tolerance=task_config.line_tolerance,
    )

    if grade.matched_issue_idx is not None:
        # Issue was matched — use grader's breakdown
        reward = grade.reward
        breakdown = grade.breakdown.copy()
    elif grade.is_false_positive or is_false_positive(
        comment, task_config.golden_issues, task_config.line_tolerance
    ):
        # False positive
        reward = -0.10
        breakdown["false_positive"] = -0.10
        grade.is_false_positive = True
    else:
        # Near an issue but didn't match keywords — neutral
        reward = 0.0

    grade.reward = reward
    return reward, breakdown, grade


def compute_context_request_reward(
    file_key: str,
    already_requested: Set[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute reward for a context request action.

    First request for a file: 0.00 (neutral)
    Repeat request for same file: -0.10 (penalty)
    """
    if file_key in already_requested:
        return -0.10, {"duplicate_context_request": -0.10}
    return 0.0, {}


def compute_early_termination_bonus(all_found: bool) -> float:
    """Return +0.10 bonus if agent found all golden issues."""
    return 0.10 if all_found else 0.0

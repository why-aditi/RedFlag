"""
False Positive Detector — penalizes comments on non-issue lines.

Checks:
  - Comment targets a line NOT near any golden issue → -0.10
  - Comment duplicates a previous comment on the same line → -0.05
"""

from typing import List, Set

try:
    from ..models import GoldenIssue, ReviewComment
except ImportError:
    from models import GoldenIssue, ReviewComment


def is_false_positive(
    comment: ReviewComment,
    golden_issues: List[GoldenIssue],
    line_tolerance: int = 5,
) -> bool:
    """
    Check if a comment targets a line not near any golden issue.

    Returns True if the comment's line doesn't fall within tolerance
    of ANY golden issue's line/range in the same file.
    """
    for issue in golden_issues:
        if comment.file != issue.file:
            continue

        if issue.line is not None:
            if abs(comment.line - issue.line) <= line_tolerance:
                return False

        if issue.line_range is not None and len(issue.line_range) == 2:
            lo, hi = issue.line_range
            if lo - line_tolerance <= comment.line <= hi + line_tolerance:
                return False

    return True


def is_duplicate_comment(
    comment: ReviewComment,
    previous_comments: List[ReviewComment],
) -> bool:
    """
    Check if a comment duplicates a previous comment on the same line.

    Same file + same line = duplicate.
    """
    for prev in previous_comments:
        if prev.file == comment.file and prev.line == comment.line:
            return True
    return False

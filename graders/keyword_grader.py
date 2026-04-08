"""
Keyword Grader — used for Tasks 1 (null_deref_basic) and 2 (auth_bypass_web).

Matches review comments against golden issues using:
  1. Issue type match
  2. Keyword hits in comment body + suggestion (2+ required)
  3. Line proximity (within ±tolerance)
  4. Fix keyword match in suggestion
  5. Severity label match
"""

from typing import List, Set

try:
    from ..models import GoldenIssue, ReviewComment
except ImportError:
    from models import GoldenIssue, ReviewComment
from .base import BaseGrader, GradeResult


class KeywordGrader(BaseGrader):
    """Grades comments using keyword matching and line proximity."""

    def grade(
        self,
        comment: ReviewComment,
        golden_issues: List[GoldenIssue],
        already_matched: Set[int],
        line_tolerance: int = 3,
    ) -> GradeResult:
        result = GradeResult()
        comment_text = (comment.body + " " + (comment.suggestion or "")).lower()

        best_match_idx = None
        best_keyword_hits = 0

        for idx, issue in enumerate(golden_issues):
            if idx in already_matched:
                continue

            # File must match
            if comment.file != issue.file:
                continue

            # Count keyword hits
            hits = sum(1 for kw in issue.keywords if kw.lower() in comment_text)
            if hits >= 2 and hits > best_keyword_hits:
                best_keyword_hits = hits
                best_match_idx = idx

        if best_match_idx is not None:
            issue = golden_issues[best_match_idx]
            result.matched_issue_idx = best_match_idx

            # +0.40 for identifying the issue (type match + 2+ keywords)
            result.breakdown["issue_found"] = 0.40
            result.reward += 0.40

            # +0.20 for correct line citation
            if self._line_matches(comment.line, issue, line_tolerance):
                result.breakdown["line_match"] = 0.20
                result.reward += 0.20

            # +0.15 for valid fix suggestion
            if comment.suggestion:
                fix_hits = sum(
                    1
                    for kw in issue.fix_keywords
                    if kw.lower() in comment.suggestion.lower()
                )
                if fix_hits >= 1:
                    result.breakdown["fix_suggested"] = 0.15
                    result.reward += 0.15

            # +0.05 for correct severity
            if comment.severity == issue.severity:
                result.breakdown["severity_match"] = 0.05
                result.reward += 0.05
        else:
            # No match — this is a false positive
            result.is_false_positive = True

        return result

    @staticmethod
    def _line_matches(
        comment_line: int, issue: GoldenIssue, tolerance: int
    ) -> bool:
        """Check if a comment's line is within tolerance of the golden issue."""
        if issue.line is not None:
            return abs(comment_line - issue.line) <= tolerance
        if issue.line_range is not None and len(issue.line_range) == 2:
            lo, hi = issue.line_range
            return lo - tolerance <= comment_line <= hi + tolerance
        return False

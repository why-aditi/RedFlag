"""
Semantic Grader — used for Task 3 (race_condition_async).

Multi-part scoring:
  40% — Identifying race condition / concurrent access
  30% — Correctly citing lines 44–52 (any line in range)
  20% — Suggesting asyncio.Lock or equivalent fix
  10% — Identifying the busy-wait secondary issue

This grader evaluates ALL comments posted during the episode collectively,
but is called per-comment and accumulates partial credit.
"""

from typing import List, Set

try:
    from ..models import GoldenIssue, ReviewComment
except ImportError:
    from models import GoldenIssue, ReviewComment
from .base import BaseGrader, GradeResult


class SemanticGrader(BaseGrader):
    """Multi-part semantic grader for complex tasks like race conditions."""

    def grade(
        self,
        comment: ReviewComment,
        golden_issues: List[GoldenIssue],
        already_matched: Set[int],
        line_tolerance: int = 5,
    ) -> GradeResult:
        result = GradeResult()
        comment_text = (comment.body + " " + (comment.suggestion or "")).lower()

        # Try to match against each golden issue
        for idx, issue in enumerate(golden_issues):
            if idx in already_matched:
                continue

            if comment.file != issue.file:
                continue

            # For the primary race condition issue (idx 0)
            if idx == 0 and issue.line_range is not None:
                score = self._grade_race_condition(comment, issue, comment_text)
                if score > 0:
                    result.matched_issue_idx = idx
                    result.reward = score
                    result.breakdown = self._last_breakdown.copy()
                    return result

            # For the secondary busy-wait issue (idx 1)
            if idx == 1 and issue.line is not None:
                score = self._grade_busy_wait(comment, issue, comment_text, line_tolerance)
                if score > 0:
                    result.matched_issue_idx = idx
                    result.reward = score
                    result.breakdown = self._last_breakdown.copy()
                    return result

        # No match — false positive
        result.is_false_positive = True
        return result

    def __init__(self):
        self._last_breakdown = {}

    def _grade_race_condition(
        self, comment: ReviewComment, issue: GoldenIssue, text: str
    ) -> float:
        """Grade a comment for the race condition issue (40% + 30% + 20%)."""
        self._last_breakdown = {}
        score = 0.0

        # 40%: Identify race condition / concurrent access
        race_keywords = [
            kw.lower() for kw in issue.keywords
        ]
        race_hits = sum(1 for kw in race_keywords if kw in text)
        if race_hits >= 2:
            self._last_breakdown["race_condition_identified"] = 0.40
            score += 0.40

            # 30%: Correct lines cited (any line in range counts)
            if issue.line_range and len(issue.line_range) == 2:
                lo, hi = issue.line_range
                if lo <= comment.line <= hi:
                    self._last_breakdown["correct_lines_cited"] = 0.30
                    score += 0.30

            # 20%: Fix suggestion AST check + Keyword fallback
            full_raw_text = comment.body + "\n" + (comment.suggestion or "")
            ast_passed = False
            try:
                from .ast_grader import grade_suggestion_with_ast
                ast_passed = grade_suggestion_with_ast(full_raw_text, ["AsyncWith", "Lock", "Mutex", "Semaphore"])
            except ImportError:
                pass
            
            if ast_passed:
                self._last_breakdown["lock_fix_suggested"] = 0.20
                score += 0.20
            else:
                full_text = text  # lowered body + suggestion combined
                fix_hits = sum(
                    1 for kw in issue.fix_keywords if kw.lower() in full_text
                )
                if fix_hits >= 1:
                    self._last_breakdown["lock_fix_suggested"] = 0.20
                    score += 0.20

        return score

    def _grade_busy_wait(
        self,
        comment: ReviewComment,
        issue: GoldenIssue,
        text: str,
        line_tolerance: int,
    ) -> float:
        """Grade a comment for the busy-wait issue (10%)."""
        self._last_breakdown = {}
        score = 0.0

        keyword_hits = sum(1 for kw in issue.keywords if kw.lower() in text)
        if keyword_hits >= 2:
            # Check line proximity
            if issue.line is not None:
                if abs(comment.line - issue.line) <= line_tolerance:
                    self._last_breakdown["busy_wait_identified"] = 0.10
                    score += 0.10

        return score

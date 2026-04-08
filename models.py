# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the RedFlag Code Review Environment.

RedFlag is an RL environment where an agent reviews code diffs and posts
structured comments identifying bugs, security issues, and logic errors.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Diff & Context Models
# ──────────────────────────────────────────────────────────────────────────────

class DiffHunk(BaseModel):
    """A single hunk from a unified diff."""
    file: str = Field(..., description="File path the hunk belongs to")
    start_line: int = Field(..., description="Starting line number of the hunk")
    end_line: int = Field(..., description="Ending line number of the hunk")
    content: str = Field(..., description="Unified diff content")


# ──────────────────────────────────────────────────────────────────────────────
# Review Comment & Context Request
# ──────────────────────────────────────────────────────────────────────────────

class ReviewComment(BaseModel):
    """A structured review comment targeting a specific line."""
    file: str = Field(..., description="File path the comment targets")
    line: int = Field(..., description="Line number the comment targets")
    issue_type: str = Field(
        ...,
        description='Type of issue: "bug", "security", "logic", or "style"',
    )
    severity: str = Field(
        ...,
        description='Severity level: "critical", "major", or "minor"',
    )
    body: str = Field(..., description="Natural language explanation of the issue")
    suggestion: Optional[str] = Field(
        None, description="Optional fix suggestion"
    )


class RequestContextAction(BaseModel):
    """Agent requests more context about the codebase."""
    request_type: str = Field(
        ...,
        description='Type of context request: "imports", "function_def", or "test_file"',
    )
    file: str = Field(..., description="File to get context for")


# ──────────────────────────────────────────────────────────────────────────────
# Action (Agent → Environment)
# ──────────────────────────────────────────────────────────────────────────────

class RedflagAction(Action):
    """
    Action the agent can take in the RedFlag environment.

    action_type determines what the agent is doing:
    - "comment"          → post a review comment (requires `comment` field)
    - "request_context"  → request additional code context (requires `context_request` field)
    - "approve"          → approve the PR (ends episode)
    - "request_changes"  → request changes on the PR (ends episode)
    """
    action_type: str = Field(
        ...,
        description='One of: "comment", "request_context", "approve", "request_changes"',
    )
    reasoning: Optional[str] = Field(
        None, description="Optional chain-of-thought reasoning from the model"
    )
    comment: Optional[ReviewComment] = Field(
        None, description="Review comment (required when action_type='comment')"
    )
    context_request: Optional[RequestContextAction] = Field(
        None,
        description="Context request (required when action_type='request_context')",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Observation (Environment → Agent)
# ──────────────────────────────────────────────────────────────────────────────

class RedflagObservation(Observation):
    """
    Observation returned to the agent after each step.

    Contains the full state of the code review: PR metadata, diff hunks,
    any context snippets fetched, comments posted so far, and step info.
    """
    task_id: str = Field(default="", description="Current task identifier")
    pr_title: str = Field(default="", description="Pull request title")
    pr_description: str = Field(default="", description="Pull request description")
    diff_hunks: List[DiffHunk] = Field(
        default_factory=list, description="Code diff hunks to review"
    )
    context_snippets: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional context keyed by 'file:function'",
    )
    comments_so_far: List[ReviewComment] = Field(
        default_factory=list, description="Comments posted in this episode"
    )
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=8, description="Maximum steps per episode")
    context_requests_remaining: int = Field(
        default=3, description="Remaining context requests before penalty"
    )
    last_action_error: Optional[str] = Field(
        None, description="Error message from last action, if any"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reward Breakdown
# ──────────────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed breakdown of how a reward was computed."""
    value: float = Field(0.0, description="Total reward for this step")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description='Component scores, e.g. {"issue_found": 0.4, "line_match": 0.2}',
    )


# ──────────────────────────────────────────────────────────────────────────────
# Golden Issue (used internally by graders, not sent to agent)
# ──────────────────────────────────────────────────────────────────────────────

class GoldenIssue(BaseModel):
    """A known issue in the diff that the agent should find."""
    issue_type: str
    severity: str
    file: str
    line: Optional[int] = None
    line_range: Optional[List[int]] = None  # [start, end] for range-based matching
    keywords: List[str] = Field(default_factory=list)
    fix_keywords: List[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    """Configuration loaded from a task's golden.json."""
    task_id: str
    pr_title: str
    pr_description: str
    golden_issues: List[GoldenIssue]
    line_tolerance: int = 3
    grader_type: str = "keyword"  # "keyword" or "semantic"
    max_steps: int = 8
    context_slots: int = 3


# ──────────────────────────────────────────────────────────────────────────────
# Extended State (for /state endpoint)
# ──────────────────────────────────────────────────────────────────────────────

class RedflagState(State):
    """Extended state for the RedFlag environment."""
    task_id: str = Field(default="", description="Current task identifier")
    done: bool = Field(default=False, description="Whether the episode is finished")

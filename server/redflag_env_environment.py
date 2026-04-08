"""
RedFlag Environment Implementation.

This is the core environment that processes code review actions:
- comment: Agent posts a review comment → graded against golden issues
- request_context: Agent requests additional code context
- approve / request_changes: Agent ends the episode

Implements the OpenEnv Environment interface with reset(), step(), state().
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        RedflagAction,
        RedflagObservation,
        RedflagState,
        ReviewComment,
        DiffHunk,
    )
    from ..env import EpisodeState
    from ..tasks.registry import (
        list_task_ids,
        load_task,
        load_diff_hunks,
        load_context_files,
    )
    from ..reward import (
        get_grader,
        compute_comment_reward,
        compute_context_request_reward,
        compute_early_termination_bonus,
    )
except ImportError:
    from models import (
        RedflagAction,
        RedflagObservation,
        RedflagState,
        ReviewComment,
        DiffHunk,
    )
    from env import EpisodeState
    from tasks.registry import (
        list_task_ids,
        load_task,
        load_diff_hunks,
        load_context_files,
    )
    from reward import (
        get_grader,
        compute_comment_reward,
        compute_context_request_reward,
        compute_early_termination_bonus,
    )


class RedflagEnvironment(Environment):
    """
    RedFlag Code Review RL Environment.

    An RL environment where an agent reviews code diffs and posts
    structured comments identifying bugs, security issues, and logic errors.

    Episode flow:
        1. reset(task_id) → load diff + golden issues → initial observation
        2. step(comment)  → grade comment → partial reward → updated observation
        3. step(request_context) → return code snippet → 0 reward
        4. step(approve/request_changes) → end episode → final score
        5. max_steps=8 → auto-terminate
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the RedFlag environment."""
        self._episode: Optional[EpisodeState] = None
        self._grader = None
        self._task_ids = list_task_ids()
        self._current_task_idx = 0

    def reset(self, **kwargs: Any) -> RedflagObservation:
        """
        Reset the environment for a new episode.

        Accepts optional task_id in kwargs. If not provided, cycles through tasks.

        Returns:
            RedflagObservation with the diff and PR metadata.
        """
        task_id = kwargs.get("task_id")

        if not task_id:
            # Cycle through tasks
            if self._task_ids:
                task_id = self._task_ids[self._current_task_idx % len(self._task_ids)]
                self._current_task_idx += 1
            else:
                task_id = "null_deref_basic"

        # Load task data
        task_config = load_task(task_id)
        diff_hunks = load_diff_hunks(task_id)
        context_files = load_context_files(task_id)

        # Create grader for this task
        self._grader = get_grader(task_config.grader_type)

        # Create episode state
        self._episode = EpisodeState(task_config, diff_hunks)
        self._episode.available_context = context_files

        return self._build_observation(reward=0.0)

    def step(self, action: RedflagAction) -> Tuple[RedflagObservation, float, bool, Dict[str, Any]]:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: RedflagAction with action_type and optional comment/context_request.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._episode is None:
            obs = RedflagObservation(
                done=True,
                reward=0.0,
                metadata={"error": "Environment not reset. Call reset() first."},
                last_action_error="Environment not reset. Call reset() first.",
            )
            return obs, 0.0, True, obs.metadata

        if self._episode.done:
            obs = self._build_observation(
                reward=0.0,
                error="Episode already finished.",
            )
            return obs, 0.0, True, obs.metadata

        # Increment step
        self._episode.step += 1
        action_type = action.action_type

        reward = 0.0
        error = None
        info = {}

        try:
            if action_type == "comment":
                reward, info, error = self._handle_comment(action)
            elif action_type == "request_context":
                reward, info, error = self._handle_context_request(action)
            elif action_type in ("approve", "request_changes"):
                reward, info = self._handle_terminal(action_type)
            else:
                error = f"Unknown action_type: '{action_type}'. Use 'comment', 'request_context', 'approve', or 'request_changes'."
                reward = 0.0
        except Exception as e:
            error = f"Error processing action: {str(e)}"
            reward = 0.0

        # Record step reward
        self._episode.step_rewards.append(reward)

        # Check max steps
        if self._episode.step >= self._episode.max_steps and not self._episode.done:
            self._episode.done = True

        obs = self._build_observation(reward=reward, error=error)
        
        # Prepare info dict
        info_dict = {
            **(obs.metadata or {}),
            "reward_breakdown": info.get("breakdown", {}),
            "final_score": self._episode.final_score if self._episode.done else None,
            "success": self._episode.success if self._episode.done else None,
        }
        obs.metadata = info_dict
        
        return obs, reward, obs.done, info_dict

    def state(self) -> State:
        """Explicit state() method for OpenEnv interface."""
        return self._episode_state_property

    @property
    def _episode_state_property(self) -> State:
        """Internal property for getting the state."""
        if self._episode is None:
            return RedflagState(episode_id="", step_count=0)

        return RedflagState(
            episode_id=self._episode.episode_id,
            step_count=self._episode.step,
            task_id=self._episode.task_config.task_id,
            done=self._episode.done,
        )

    # ── Action Handlers ──────────────────────────────────────────────────

    def _handle_comment(self, action: RedflagAction):
        """Handle a comment action."""
        if action.comment is None:
            return 0.0, {}, "action_type is 'comment' but no comment provided."

        comment = action.comment

        reward, breakdown, grade = compute_comment_reward(
            comment=comment,
            task_config=self._episode.task_config,
            previous_comments=self._episode.comments,
            matched_issues=self._episode.matched_issues,
            grader=self._grader,
        )

        # Update state
        self._episode.comments.append(comment)
        if grade.matched_issue_idx is not None:
            self._episode.matched_issues.add(grade.matched_issue_idx)

        # Check if all issues found → early termination
        if self._episode.all_issues_found:
            bonus = compute_early_termination_bonus(True)
            reward += bonus
            breakdown["early_termination_bonus"] = bonus
            self._episode.done = True

        return reward, {"breakdown": breakdown}, None

    def _handle_context_request(self, action: RedflagAction):
        """Handle a context request action."""
        if action.context_request is None:
            return 0.0, {}, "action_type is 'request_context' but no context_request provided."

        req = action.context_request
        file_key = f"{req.file}:{req.request_type}"

        # Compute reward (penalty for duplicate)
        reward, breakdown = compute_context_request_reward(
            file_key, self._episode.context_requests_used
        )

        # Track the request
        self._episode.context_requests_used.add(file_key)

        # Decrement context slots
        if reward == 0.0:
            self._episode.context_slots_remaining = max(
                0, self._episode.context_slots_remaining - 1
            )

        # Look up the context file
        context_content = None
        for fname, content in self._episode.available_context.items():
            if req.file.replace("/", "_").replace(".", "_") in fname.replace(".", "_"):
                context_content = content
                break
            if req.file.split("/")[-1].replace(".", "_") in fname.replace(".", "_"):
                context_content = content
                break

        if context_content:
            snippet_key = f"{req.file}:{req.request_type}"
            self._episode.context_snippets[snippet_key] = context_content
        else:
            return reward, {"breakdown": breakdown}, f"No context found for file '{req.file}'"

        return reward, {"breakdown": breakdown}, None

    def _handle_terminal(self, action_type: str):
        """Handle approve or request_changes (terminal actions)."""
        self._episode.done = True

        # Check for early termination bonus
        bonus = compute_early_termination_bonus(self._episode.all_issues_found)
        breakdown = {}
        if bonus > 0:
            breakdown["early_termination_bonus"] = bonus

        return bonus, {"breakdown": breakdown}

    # ── Observation Builder ──────────────────────────────────────────────

    def _build_observation(
        self,
        reward: float = 0.0,
        error: Optional[str] = None,
    ) -> RedflagObservation:
        """Build a RedflagObservation from current episode state."""
        ep = self._episode

        return RedflagObservation(
            task_id=ep.task_config.task_id,
            pr_title=ep.task_config.pr_title,
            pr_description=ep.task_config.pr_description,
            diff_hunks=ep.diff_hunks,
            context_snippets=ep.context_snippets,
            comments_so_far=ep.comments,
            step=ep.step,
            max_steps=ep.max_steps,
            context_requests_remaining=ep.context_slots_remaining,
            last_action_error=error,
            done=ep.done,
            reward=reward,
            metadata={
                "episode_id": ep.episode_id,
                "total_reward": ep.total_reward + reward,
                "issues_found": len(ep.matched_issues),
                "total_issues": len(ep.task_config.golden_issues),
            },
        )

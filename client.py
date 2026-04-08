"""RedFlag Code Review Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import RedflagAction, RedflagObservation, RedflagState
except ImportError:
    from models import RedflagAction, RedflagObservation, RedflagState


class RedflagEnv(
    EnvClient[RedflagAction, RedflagObservation, RedflagState]
):
    """
    Client for the RedFlag Code Review Environment.

    Example:
        >>> with RedflagEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.pr_title)
        ...
        ...     action = RedflagAction(
        ...         action_type="comment",
        ...         comment=ReviewComment(
        ...             file="api/stats.py",
        ...             line=23,
        ...             issue_type="bug",
        ...             severity="critical",
        ...             body="This will crash on empty input",
        ...         ),
        ...     )
        ...     result = client.step(action)
        ...     print(result.reward)

    Example with Docker:
        >>> client = await RedflagEnv.from_docker_image("redflag-env:latest")
    """

    def _step_payload(self, action: RedflagAction) -> Dict:
        """Convert RedflagAction to JSON payload."""
        payload = {"action_type": action.action_type}

        if action.comment is not None:
            payload["comment"] = action.comment.model_dump()

        if action.context_request is not None:
            payload["context_request"] = action.context_request.model_dump()

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[RedflagObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})

        observation = RedflagObservation(
            task_id=obs_data.get("task_id", ""),
            pr_title=obs_data.get("pr_title", ""),
            pr_description=obs_data.get("pr_description", ""),
            diff_hunks=obs_data.get("diff_hunks", []),
            context_snippets=obs_data.get("context_snippets", {}),
            comments_so_far=obs_data.get("comments_so_far", []),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 8),
            context_requests_remaining=obs_data.get("context_requests_remaining", 3),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> RedflagState:
        """Parse server response into RedflagState."""
        return RedflagState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            done=payload.get("done", False),
        )

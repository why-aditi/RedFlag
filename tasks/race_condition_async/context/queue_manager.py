"""
Queue manager module — available as context for the agent.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class QueueManager:
    """Manages multiple job queues and their dispatchers."""

    def __init__(self):
        self._queues: Dict[str, Any] = {}
        self._dispatchers: Dict[str, Any] = {}
        self._hooks: Dict[str, Callable[..., Awaitable]] = {}

    def register_queue(self, name: str, queue: Any) -> None:
        """Register a named queue."""
        self._queues[name] = queue

    def register_dispatcher(self, name: str, dispatcher: Any) -> None:
        """Register a dispatcher for a named queue."""
        self._dispatchers[name] = dispatcher

    def on_job_complete(self, hook: Callable[..., Awaitable]) -> None:
        """Register a completion hook."""
        self._hooks["on_complete"] = hook

    async def run_all(self) -> Dict[str, Any]:
        """Run all registered dispatchers."""
        results = {}
        for name, dispatcher in self._dispatchers.items():
            try:
                result = await dispatcher.dispatch_all()
                results[name] = {"status": "ok", "results": result}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        return results

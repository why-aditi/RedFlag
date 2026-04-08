"""
FastAPI application for the RedFlag Code Review Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with 'pip install openenv-core[core]'"
    ) from e

try:
    from ..models import RedflagAction, RedflagObservation
    from .redflag_env_environment import RedflagEnvironment
except (ImportError, ModuleNotFoundError):
    from models import RedflagAction, RedflagObservation
    from server.redflag_env_environment import RedflagEnvironment


# Create the app using OpenEnv's create_app factory
app = create_app(
    RedflagEnvironment,
    RedflagAction,
    RedflagObservation,
    env_name="redflag_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution.

    Usage:
        uv run --project . server
        python -m redflag_env.server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)

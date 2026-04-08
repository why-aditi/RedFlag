"""
FastAPI application for the RedFlag Code Review Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /: Root status endpoint
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'pip install openenv-core[core]'"
    ) from e

try:
    from models import RedflagAction, RedflagObservation
    from server.redflag_env_environment import RedflagEnvironment
except (ImportError, ModuleNotFoundError):
    from .models import RedflagAction, RedflagObservation
    from .server.redflag_env_environment import RedflagEnvironment

# Create the app using OpenEnv's create_app factory
app = create_app(
    RedflagEnvironment,
    RedflagAction,
    RedflagObservation,
    env_name="redflag_env",
    max_concurrent_envs=1,
)

@app.get("/")
def root():
    """Friendly root endpoint for UI/Hugging Face health checks."""
    return {
        "status": "Running",
        "message": "RedFlag RL Code Review Environment is active.",
        "documentation": "https://huggingface.co/spaces/lostinthesky/redflag-env",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

def main():
    """Entry point for standalone execution."""
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

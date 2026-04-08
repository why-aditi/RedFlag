# Use a stable, lightweight Python base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    PORT=7860

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Final installation of the project itself
RUN uv sync --frozen --no-dev

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Metadata tags
LABEL maintainer="Antigravity AI"
LABEL description="RedFlag Automated Code Review RL Environment"

# Run the FastAPI server
# We use uvicorn directly from the uv-managed environment
CMD ["uv", "run", "python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

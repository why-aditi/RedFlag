---
title: RedFlag
emoji: 🚩
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
hf_oauth: true
---

# RedFlag — Code Review RL Environment

RedFlag is an OpenEnv-compliant reinforcement learning environment designed to simulate a senior software engineer performing automated code review. An AI agent is presented with a realistic code diff (pull request) and must post structured review comments to identify bugs, security vulnerabilities, and logic errors, all while minimizing false positives.

## 1. Environment Overview & Motivation

Code review is a high-stakes, cognitively demanding task where missing a subtle bug can lead to production outages or security breaches. Traditional static analysis lacks the contextual understanding necessary to catch complex logic flaws or concurrency issues. RedFlag simulates a realistic pull-request review workflow by giving an RL agent a unified patch diff and the ability to selectively request surrounding context files.

The primary motivation is to train AI agents that can accurately flag issues with the correct severity, pinpoint the actual faulty line, suggest a valid fix, and avoid noisy "nitpick" comments or false positives, replicating human expert behavior. The state space is vast and unstructured since the agent operates over natural language and code bases.

## 2. Observation & Action Spaces

The agent communicates with the environment using a standardized JSON schema.

### Observation Space

The observation provides the agent with all context about the PR and the history of the current review session.

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | The identifier for the current PR task. |
| `pr_title` | `string` | Title of the Pull Request. |
| `pr_description` | `string` | Description of the changes made in the PR. |
| `diff_hunks` | `List[DiffHunk]` | Array of modified code hunks `(file, start_line, end_line, content)`. |
| `context_snippets` | `Dict[str, str]` | Additional code context requested by the agent during previous steps. |
| `comments_so_far` | `List[ReviewComment]` | Any review comments the agent has successfully posted in this episode. |
| `step` | `int` | The current action step the agent is on. |
| `max_steps` | `int` | Typically 8 or 12. Auto-terminates when reached. |
| `context_requests_remaining` | `int` | Number of remaining allowed context file pulls before penalty. |
| `last_action_error` | `string\|null` | Error message if the previous action was malformed. |

### Action Space

The agent can output exactly one actionable JSON block per step. The top-level `action_type` dictates what action is taken.

| Action Type | Required Fields | Description |
|---|---|---|
| `comment` | `comment: ReviewComment` | Posts a review comment targeting a specific file and line. Requires `issue_type`, `severity`, `body`, and an optional `suggestion`. |
| `request_context` | `context_request: RequestContextAction` | Requests another file from the codebase (`imports`, `function_def`, `test_file`). Fills an observation snippet in the next step. |
| `approve` | *(None)* | Immediately ends the review episode, signaling the PR looks good. |
| `request_changes` | *(None)* | Immediately ends the review episode, signaling the agent has finished reporting all issues. |

## 3. Task Descriptions

RedFlag currently supports 4 meticulously crafted code review tasks built into the environment directory framework.

| Task ID | Level | Description |
|---|---|---|
| **`null_deref_basic`** | **Easy** | A FastAPI endpoint introducing a missing length validation constraint leading to an empty list `IndexError`, and a separate off-by-one loop boundary bug. |
| **`auth_bypass_web`** | **Medium** | A Flask admin-panel blueprint missing a crucial `@login_required` decorator, alongside a string-interpolated SQL injection vulnerability in a DB search query. |
| **`race_condition_async`** | **Hard** | An `asyncio` job dispatcher demonstrating a Time-Of-Check/Time-Of-Use (TOCTOU) race condition during concurrent list modifications, accompanied by a CPU busy-spin anti-pattern. |
| **`django_django_16588`** | **Real-world** | An actual security patch pulled from Django core to address precision vulnerabilities in a floatformat filter template when handling zero values. |

## 4. Reward Shaping & Rules

The maximum possible reward is normalized to `1.0`.

- **Positive Rewards**:
  - `+0.40`: Identifying an issue (matching issue_type and 2+ keywords)
  - `+0.20`: Pinpointing the correct line (within tolerance)
  - `+0.15`: Suggesting a valid fix 
  - `+0.05`: Correctly labeling the severity
  - `+0.10`: Early termination bonus (finding all issues and ending the episode)
- **Penalties** (False Positives):
  - `-0.10`: Commenting on a non-issue line (strict false positive punishment)
  - `-0.05`: Submitting a duplicate comment on the same line
  - `-0.10`: Requesting the same context file multiple times

## 5. Setup & Usage

You can run RedFlag natively using standard python tooling or package it via Docker.

**Using uv / Python:**
```bash
python -m venv .venv
source .venv/bin/activate  # Or `.venv\Scripts\activate` on Windows
pip install -e .
uv run server
# alternatively: python -m uvicorn server.app:app --port 7860
```

**Using Docker:**
```bash
docker build -t redflag-env .
docker run -p 7860:7860 redflag-env
```

**Inference Execution:**
Once the server is running, you can run the benchmark inference script:
```bash
cp .env.example .env
# Edit .env and supply your HF_TOKEN
export MODEL_NAME="zai-org/GLM-5"
python inference.py
```

## 6. Baseline Performance

The table below documents the baseline performance running the `zai-org/GLM-5` zero-shot inference pipeline. Performance was consolidated from multiple runs to account for transient provider rate limits.

| Model | Task | Score (0.0 to 1.0) | Result |
|---|---|---|---|
| `zai-org/GLM-5` | `auth_bypass_web` | **1.000** | `[PASS]` |
| `zai-org/GLM-5` | `null_deref_basic` | **0.765** | `[PASS]` |
| `zai-org/GLM-5` | `race_condition_async` | **0.647** | `[PASS]` |
| `zai-org/GLM-5` | `django_django_16588` | **0.529** | `[PASS]` |



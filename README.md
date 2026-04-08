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

RedFlag is a **fully compliant OpenEnv reinforcement learning environment** designed to simulate a senior software engineer performing automated code review. An AI agent is presented with a realistic code diff (pull request) and must post structured review comments to identify bugs, security vulnerabilities, and logic errors, all while minimizing false positives.

## 1. Environment Overview & Motivation

Code review is a high-stakes, cognitively demanding task where missing a subtle bug can lead to production outages or security breaches. RedFlag simulates a realistic pull-request review workflow by giving an RL agent a unified patch diff and the ability to selectively request surrounding context files.

As an OpenEnv-native space, RedFlag implements the standard `Environment` interface, making it ready for automated evaluation, validation, and training pipelines.

## 2. OpenEnv Interface

RedFlag implements the full OpenEnv specification (v1) with typed Pydantic models for actions, observations, and rewards.

### Observation Space

The observation provides the agent with all context about the PR and the history of the current review session.

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | The identifier for the current PR task. |
| `pr_title` | `string` | Title of the Pull Request. |
| `diff_hunks` | `List[DiffHunk]` | Array of modified code hunks. |
| `comments_so_far` | `List[ReviewComment]` | Review comments posted in this episode. |
| `reward` | `float` | Scalar reward from the last action. |
| `done` | `bool` | Whether the episode has terminated. |

### Action Space

| Action Type | Required Fields | Description |
|---|---|---|
| `comment` | `comment: ReviewComment` | Posts a review comment targeting a specific file and line. |
| `request_context` | `context_request` | Requests another file from the codebase. |
| `approve` | *(None)* | Immediately ends the review episode (Success). |
| `request_changes` | *(None)* | Immediately ends the review episode (Failure). |

### Reward Shaping & Rules

RedFlag implements a sophisticated reward function to incentivize precise code review. The maximum possible reward is normalized to **1.0**.

- **Positive Incentives**:
  - `+0.40`: Identifying an issue (matching issue_type and 2+ keywords)
  - `+0.20`: Pinpointing the correct line (within tolerance)
  - `+0.15`: Suggesting a valid fix 
  - `+0.05`: Correctly labeling the severity
  - `+0.10`: Early termination bonus (all issues found)

- **Penalties (False Positives)**:
  - `-0.10`: Commenting on a non-issue line (strict noise penalty)
  - `-0.05`: Submitting a duplicate comment on the same line
  - `-0.10`: Requesting the same context file multiple times

## 3. Usage & Setup

### Installation

```bash
pip install openenv-core
pip install -e .
```

### Validation & Server

Ensure the environment is compliant and start the server:
```bash
openenv validate
openenv up server.app:app --port 7860
```

## 4. Task Descriptions

| Task ID | Level | Description |
|---|---|---|
| **`null_deref_basic`** | **Easy** | Missing length validation and off-by-one boundary bug. |
| **`auth_bypass_web`** | **Medium** | Missing authorization and SQL Injection. |
| **`race_condition_async`** | **Hard** | TOCTOU race condition in concurrent modifications. |
| **`django_django_16588`** | **Real-world** | Precision vulnerability in Django template filter. |

## 5. Deployment

RedFlag is configured for multi-mode deployment (Hugging Face Spaces, Docker, Local).

```bash
docker build -t redflag-env .
docker run -p 7860:7860 redflag-env
```

**Inference Execution:**
```bash
export MODEL_NAME="zai-org/GLM-5"
python inference.py
```

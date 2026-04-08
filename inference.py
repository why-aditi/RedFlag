"""
RedFlag Inference Script
========================

MANDATORY root-level inference script for OpenEnv submission.

Environment Variables:
    API_BASE_URL   The API endpoint for the LLM (default: HF Router)
    MODEL_NAME     The model identifier (default: zai-org/GLM-5)
    HF_TOKEN       Your Hugging Face / API key
    IMAGE_NAME     Docker image name for the environment

STDOUT FORMAT:
    [START] task=<task_name> env=redflag_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────

def load_env():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
load_env()

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "zai-org/GLM-5"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if not API_KEY:
    print("\n[ERROR] Missing Hugging Face Token!", file=sys.stderr)
    print("This script uses the free Hugging Face API router by default (via the OpenAI python package).", file=sys.stderr)
    print("Please set your HF_TOKEN in the .env file or run `export HF_TOKEN=...`\n", file=sys.stderr)
    exit(1)

import os
from pathlib import Path

BENCHMARK = "redflag_env"
TASKS_DIR = Path(__file__).parent / "tasks"
TASKS = [
    d.name for d in TASKS_DIR.iterdir()
    if d.is_dir() and (d / "golden.json").exists()
]
MAX_STEPS = 12
TEMPERATURE = 0.3
MAX_TOKENS = 2048
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior software engineer doing code review.
You will receive a code diff (pull request). Your job is to identify bugs,
security issues, and logic errors.

For each issue you find, respond with EXACTLY one JSON object per message.

### Required Structure:
{
  "reasoning": "<step-by-step logic>",
  "action_type": "comment",
  "comment": {
    "file": "<filename>",
    "line": <line_number>,
    "issue_type": "bug|security|logic",
    "severity": "critical|major|minor",
    "body": "<explanation>",
    "suggestion": "<fix>"
  }
}

### Rules:
- Be precise. Only comment on lines that have real issues.
- Keep the 'reasoning' and 'body' fields CONCISE (under 50 words each).
- When finished, use {"action_type": "request_changes"}.
""").strip()


import re

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract the first JSON object from text."""
    # Try literal match first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { } block
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_candidates = []
        # Try best effort from start to end
        json_candidates.append(text[start:end+1])
        
        for candidate in json_candidates:
            try:
                # Basic cleanup: remove common markdown markers if they were inside found block
                cleaned = candidate.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    return None


CRITIC_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Review Supervisor.
Your job is to double-check a junior reviewer's proposed comment.

You will be given the original PR diff and Context, followed by the junior's Proposed Action.

Your tasks:
1. Verify the file and line number cited truly exist and have the problem indicated.
2. Ensure the vulnerability is a genuine flaw, not hallucinated or over-exaggerated.

Respond with EXACTLY ONE JSON object:
{
  "approved": true|false,
  "feedback": "<If approved, leave empty. If false, strictly explain why so the reviewer can correct it>"
}
""").strip()


import sys

# ── Logging ──────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ── LLM Interface ───────────────────────────────────────────────────────

def call_llm_with_fallback(client: OpenAI, messages: list, temperature: float, max_tokens: int) -> str:
    """Invokes LLM and automatically falls back to secondary models if Pro tier quota is depleted."""
    models = [
        MODEL_NAME,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "XiaomiMiMo/MiMo-V2-Flash",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    for i, m in enumerate(models):
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            err_msg = str(exc).lower()
            if "402" in err_msg or "429" in err_msg or "depleted" in err_msg:
                print(f"[DEBUG] Model {m} hit quota. Falling back to alternative...", file=sys.stderr, flush=True)
                continue
            # If it's a different error but we have more models, also try to fallback
            if i < len(models) - 1:
                print(f"[DEBUG] Model {m} failed: {exc}. Trying next...", file=sys.stderr, flush=True)
                continue
            raise exc
    raise Exception("All fallback models failed due to quota limits or errors.")

def build_user_prompt(
    observation: Dict[str, Any], last_reward: float = 0.0, last_error: Optional[str] = None, critic_feedback: Optional[str] = None
) -> str:
    """Build the user prompt from the environment observation."""
    diff_text = ""
    for hunk in observation.get("diff_hunks", []):
        if isinstance(hunk, dict):
            diff_text += f"\n--- {hunk.get('file', '')} ---\n{hunk.get('content', '')}\n"
        else:
            diff_text += f"\n--- {hunk.file} ---\n{hunk.content}\n"

    context_text = ""
    context_snippets = observation.get("context_snippets", {})
    if context_snippets:
        context_text = "\n\nAdditional context:\n"
        for key, snippet in context_snippets.items():
            context_text += f"\n--- {key} ---\n{snippet}\n"

    comments_text = ""
    comments = observation.get("comments_so_far", [])
    if comments:
        comments_text = "\n\nComments already posted:\n"
        for c in comments:
            if isinstance(c, dict):
                comments_text += f"- {c.get('file', '')}:{c.get('line', '')} [{c.get('issue_type', '')}] {c.get('body', '')}\n"
            else:
                comments_text += f"- {c.file}:{c.line} [{c.issue_type}] {c.body}\n"

    pr_title = observation.get("pr_title", "")
    pr_desc = observation.get("pr_description", "")
    step = observation.get("step", 0)
    max_steps = observation.get("max_steps", 8)

    feedback_text = ""
    if critic_feedback:
        feedback_text += f"\n\n[CRITIC SUPERVISOR REJECTED LAST PROPOSAL]\n{critic_feedback}\nPlease try again and fix the issue.\n"
    elif step > 0:
        feedback_text += f"\n\n[ENVIRONMENT FEEDBACK]\nYour last action gave a reward of {last_reward:.2f}.\n"
        if last_error:
            feedback_text += f"Error from last action: {last_error}\n"
        feedback_text += "(If reward <= 0, your comment was a false positive, a duplicate, or invalid. Adjust your strategy!)\n"

    return textwrap.dedent(f"""
PR Title: {pr_title}
PR Description: {pr_desc}
Step: {step}/{max_steps}

Code Diff:
{diff_text}
{context_text}
{comments_text}{feedback_text}

Identify the next issue in this diff, request context, or if you've found all issues, respond with {{"action_type": "request_changes"}}.
""").strip()


def get_model_action(
    client: OpenAI,
    observation: Dict[str, Any],
    history: List[Dict],
    last_reward: float = 0.0,
    last_error: Optional[str] = None,
    critic_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LLM and parse its response into an action dict."""
    user_prompt = build_user_prompt(observation, last_reward, last_error, critic_feedback)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last 4 exchanges)
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["prompt"]})
        messages.append({"role": "assistant", "content": h["response"]})

    messages.append({"role": "user", "content": user_prompt})

    try:
        text = call_llm_with_fallback(client, messages, TEMPERATURE, MAX_TOKENS)

        action = extract_json(text)
        if action:
            # Healing: If model flattened the comment into top level
            if "file" in action and "line" in action and "action_type" not in action:
                action = {
                    "reasoning": action.get("reasoning", ""),
                    "action_type": "comment",
                    "comment": {
                        "file": action.get("file"),
                        "line": action.get("line"),
                        "issue_type": action.get("issue_type", "bug"),
                        "severity": action.get("severity", "major"),
                        "body": action.get("body", action.get("reasoning", "Issue found.")),
                        "suggestion": action.get("suggestion")
                    }
                }
            return action, text, user_prompt
        
        raise json.JSONDecodeError("No valid JSON found", text, 0)

    except json.JSONDecodeError:
        # If LLM returns non-JSON, try to end the episode
        print(f"[DEBUG] Failed to parse LLM response as JSON: {text[:200]}", file=sys.stderr, flush=True)
        return {"action_type": "request_changes"}, text, user_prompt
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return {"action_type": "request_changes"}, str(exc), user_prompt


def get_critic_approval(client: OpenAI, user_prompt: str, proposed_action: str) -> tuple[bool, str]:
    """Ask the critic model to verify the proposed action."""
    content = user_prompt + "\n\n--- JUNIOR PROPOSED ACTION ---\n" + proposed_action
    try:
        text = call_llm_with_fallback(
            client,
            [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            0.1,
            500
        )
        data = extract_json(text)
        if data:
            return data.get("approved", False), data.get("feedback", "")
        
        return True, ""  # Default to allow if critic returns non-JSON
    except Exception as e:
        print(f"[DEBUG CRITIC] Failed to parse critic: {e}", file=sys.stderr, flush=True)
        return True, ""  # Default to allow if critic crashes


# ── Environment Interaction ───────────────────────────────────────────────

from client import RedflagEnv
from models import RedflagAction

def run_task(client: OpenAI, base_url: str, task_id: str) -> Dict[str, Any]:
    """Run a single task episode and return results."""
    rewards: List[float] = []
    steps_taken = 0
    history: List[Dict] = []
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with RedflagEnv(base_url=base_url).sync() as env:
            # Reset environment
            reset_result = env.reset(task_id=task_id)
            observation = reset_result.observation.model_dump()
            done = reset_result.done
            reward = 0.0
            error = None

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                critic_feedback = None
                max_critic_retries = 2
                
                for attempt in range(max_critic_retries):
                    # Get action from LLM
                    action_dict, raw_response, prompt = get_model_action(
                        client, observation, history, reward, error, critic_feedback
                    )
                    
                    if action_dict.get("action_type") != "comment":
                        break # Only verify comments
                        
                    is_approved, critique = get_critic_approval(client, prompt, raw_response)
                    if is_approved:
                        break
                        
                    print(f"[CRITIC REJECTED] {critique}", file=sys.stderr, flush=True)
                    critic_feedback = critique

                history.append({"prompt": prompt, "response": raw_response})

                # Convert dict to Pydantic model
                try:
                    action_obj = RedflagAction.model_validate(action_dict)
                except Exception as e:
                    print(f"[DEBUG] Validation error: {e}", file=sys.stderr)
                    action_obj = RedflagAction(action_type="request_changes")
                    action_dict = {"action_type": "request_changes"}

                # Send action to environment
                step_result = env.step(action_obj)

                observation = step_result.observation.model_dump()
                reward = step_result.reward or 0.0
                done = step_result.done
                error = observation.get("last_action_error")

                rewards.append(reward)
                steps_taken = step

                # Format action string for logging
                action_str = action_dict.get("action_type", "unknown")
                if action_dict.get("comment"):
                    c = action_dict["comment"]
                    action_str = f"comment({c.get('file', '')}:{c.get('line', '')})"

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error,
                )

                if done:
                    break
        
        # Compute final success
        max_possible = 1.7
        score = sum(rewards) / max_possible if max_possible > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"Task {task_id} error: {e}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def main() -> None:
    """Run inference on all tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    base_url = ENV_BASE_URL

    print(f"Using model: {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"Using env: {base_url}", file=sys.stderr, flush=True)

    for task_id in TASKS:
        run_task(client, base_url, task_id)


if __name__ == "__main__":
    main()


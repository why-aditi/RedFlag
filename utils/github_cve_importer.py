"""
Automated Task Ingestion Pipeline for RedFlag Environment.
Downloads security patches directly from GitHub and auto-generates golden.json tasks
using an LLM to identify the critical vulnerability lines.
"""

import os
import re
import json
import requests
import argparse
from pathlib import Path
from openai import OpenAI

# Support reading .env locally if run manually
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

PROMPT_TEMPLATE = """
You are an expert security engineer creating a benchmark task for an AI Code Review Environment.
Below is a raw diff from a GitHub Pull Request that patches a vulnerability.

Your job is to identify the EXACT line number(s) where the vulnerability was present (or where a check should be added) in the modified files.
Also, provide keywords that an AI agent might use when pointing out the vulnerability, and keywords for fixing it.

Respond ONLY with a valid JSON document matching this schema:
{
    "task_id": "<string identifier>",
    "pr_title": "<short title>",
    "pr_description": "<1 sentence context>",
    "grader_type": "semantic",
    "golden_issues": [
        {
            "issue_type": "bug|security",
            "severity": "critical|major",
            "file": "<filename>",
            "line": <integer>,
            "keywords": ["<keyword1>", "<keyword2>"],
            "fix_keywords": ["<fix_keyword1>", "<fix_keyword2>"]
        }
    ]
}

DIFF:
{diff_content}
"""

def generate_golden_json(diff_content: str) -> dict:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.replace("{diff_content}", diff_content[:15000])}],
        temperature=0.1,
    )
    text = response.choices[0].message.content or ""
    # Strip markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        text = match.group(1) if match else "{}"
    return json.loads(text)

def ingest_github_pr(owner: str, repo: str, pr_number: int):
    print(f"[*] Fetching PR #{pr_number} from {owner}/{repo}...")
    patch_url = f"https://patch-diff.githubusercontent.com/raw/{owner}/{repo}/pull/{pr_number}.patch"
    resp = requests.get(patch_url)
    resp.raise_for_status()
    diff_content = resp.text
    
    print("[*] Generating semantic golden issue using LLM...")
    golden_data = generate_golden_json(diff_content)
    
    task_id = f"{owner}_{repo}_{pr_number}"
    golden_data["task_id"] = task_id
    
    target_dir = Path(__file__).parent.parent / "tasks" / task_id
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with open(target_dir / "diff.patch", "w", encoding="utf-8") as f:
        f.write(diff_content)
        
    with open(target_dir / "golden.json", "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=4)
        
    print(f"[+] Task '{task_id}' fully generated at {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("owner", help="GitHub repo owner")
    parser.add_argument("repo", help="GitHub repository name")
    parser.add_argument("pr", type=int, help="PR number")
    args = parser.parse_args()
    
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not set in .env file.")
        exit(1)
        
    ingest_github_pr(args.owner, args.repo, args.pr)

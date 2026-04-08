"""
Task Registry — loads tasks from the tasks/ directory structure.

Each task directory contains:
  - diff.patch    — unified diff the agent reviews
  - golden.json   — golden issues + grader config
  - context/      — additional source files the agent can request
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ..models import DiffHunk, GoldenIssue, TaskConfig
except ImportError:
    from models import DiffHunk, GoldenIssue, TaskConfig


_TASKS_DIR = Path(__file__).parent


def list_task_ids() -> List[str]:
    """Return all available task IDs."""
    return [
        d.name
        for d in _TASKS_DIR.iterdir()
        if d.is_dir() and (d / "golden.json").exists()
    ]


def load_task(task_id: str) -> TaskConfig:
    """Load a task configuration from its golden.json."""
    task_dir = _TASKS_DIR / task_id
    golden_path = task_dir / "golden.json"

    if not golden_path.exists():
        raise ValueError(f"Task '{task_id}' not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    golden_issues = [GoldenIssue(**issue) for issue in data["golden_issues"]]

    return TaskConfig(
        task_id=data["task_id"],
        pr_title=data["pr_title"],
        pr_description=data["pr_description"],
        golden_issues=golden_issues,
        line_tolerance=data.get("line_tolerance", 3),
        grader_type=data.get("grader_type", "keyword"),
        max_steps=data.get("max_steps", 8),
        context_slots=data.get("context_slots", 3),
    )


def load_diff_hunks(task_id: str) -> List[DiffHunk]:
    """Parse the diff.patch file into DiffHunk objects."""
    task_dir = _TASKS_DIR / task_id
    diff_path = task_dir / "diff.patch"

    if not diff_path.exists():
        raise ValueError(f"No diff.patch found for task '{task_id}'")

    with open(diff_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the unified diff into hunks
    hunks: List[DiffHunk] = []
    current_file = ""
    current_hunk_lines: List[str] = []
    current_start = 0
    current_end = 0

    for line in content.split("\n"):
        if line.startswith("+++ b/"):
            # Flush previous hunk
            if current_file and current_hunk_lines:
                hunks.append(DiffHunk(
                    file=current_file,
                    start_line=current_start,
                    end_line=current_end,
                    content="\n".join(current_hunk_lines),
                ))
                current_hunk_lines = []
            current_file = line[6:]  # strip "+++ b/"
        elif line.startswith("@@ "):
            # Flush previous hunk if same file
            if current_file and current_hunk_lines:
                hunks.append(DiffHunk(
                    file=current_file,
                    start_line=current_start,
                    end_line=current_end,
                    content="\n".join(current_hunk_lines),
                ))
                current_hunk_lines = []

            # Parse @@ -old,count +new,count @@
            parts = line.split(" ")
            if len(parts) >= 3:
                new_range = parts[2]  # e.g. "+1,120"
                if "," in new_range:
                    start_str, count_str = new_range.lstrip("+").split(",")
                    current_start = int(start_str)
                    current_end = current_start + int(count_str) - 1
                else:
                    current_start = int(new_range.lstrip("+"))
                    current_end = current_start
            current_hunk_lines = [line]
        elif line.startswith("--- "):
            continue  # skip old file header
        else:
            current_hunk_lines.append(line)

    # Flush last hunk
    if current_file and current_hunk_lines:
        hunks.append(DiffHunk(
            file=current_file,
            start_line=current_start,
            end_line=current_end,
            content="\n".join(current_hunk_lines),
        ))

    return hunks


def load_context_files(task_id: str) -> Dict[str, str]:
    """Load all context files for a task, keyed by filename."""
    task_dir = _TASKS_DIR / task_id / "context"
    context: Dict[str, str] = {}

    if not task_dir.exists():
        return context

    for fpath in task_dir.iterdir():
        if fpath.is_file():
            with open(fpath, "r", encoding="utf-8") as f:
                context[fpath.name] = f.read()

    return context

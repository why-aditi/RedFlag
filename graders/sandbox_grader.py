"""
Experimental Sandbox Grader for Execution-Based Evaluation.
Applies the AI's code suggestions directly to a temporary copy of the codebase
and runs the task's test suite to mathematically verify if the bug is fixed.
"""
import os
import subprocess
import tempfile
import shutil
from typing import List, Set

try:
    from ..models import GoldenIssue, ReviewComment
except ImportError:
    from models import GoldenIssue, ReviewComment
from .base import BaseGrader, GradeResult


class SandboxGrader(BaseGrader):
    """Execution-based grader running code in a temporary sandbox."""

    def __init__(self, task_dir: str):
        self.task_dir = task_dir

    def grade(
        self,
        comment: ReviewComment,
        golden_issues: List[GoldenIssue],
        already_matched: Set[int],
        line_tolerance: int = 5,
    ) -> GradeResult:
        result = GradeResult()
        
        # If there's no code suggestion, we can't execute it
        if not comment.suggestion:
            result.is_false_positive = True
            return result

        # We assume there is a `project/` directory inside `task_dir` that contains the runnable code
        project_dir = os.path.join(self.task_dir, "project")
        if not os.path.exists(project_dir):
            # Fallback for environments lacking an executable project
            result.is_false_positive = True
            return result

        # Create a temporary sandbox copy of the project
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox_project = os.path.join(temp_dir, "project")
            shutil.copytree(project_dir, sandbox_project)

            # Apply the AI's suggested fix to the target file
            target_file_path = os.path.join(sandbox_project, comment.file)
            if os.path.exists(target_file_path):
                # Simple replacement strategy (assuming the agent provided a full replacement block)
                with open(target_file_path, "r", encoding="utf-8") as f:
                    original_content = f.readlines()
                
                # We replace the specific line the AI commented on
                # Note: A real LLM patch applier would use `unidiff` or `patch` here.
                # For this proof-of-concept Sandbox, we overwrite the target line with the suggestion
                if 0 <= comment.line - 1 < len(original_content):
                    original_content[comment.line - 1] = comment.suggestion + "\n"
                
                with open(target_file_path, "w", encoding="utf-8") as f:
                    f.writelines(original_content)

            # Execute the test suite
            try:
                # We look for a test script, falling back to basic pytest
                test_cmd = ["python", "-m", "pytest", "tests/"]
                test_script = os.path.join(sandbox_project, "run_tests.sh")
                if os.path.exists(test_script):
                    test_cmd = ["bash", "run_tests.sh"]

                out = subprocess.run(
                    test_cmd,
                    cwd=sandbox_project,
                    capture_output=True,
                    timeout=10,
                    text=True
                )
                
                # If pytest returns 0, the test passed! The AI's fix worked.
                if out.returncode == 0:
                    result.reward = 1.0
                    result.breakdown = {"sandbox_test_passed": 1.0}
                    # We match it to the first unresolved golden issue since it fixed the suite
                    result.matched_issue_idx = next(
                        (i for i in range(len(golden_issues)) if i not in already_matched), 0
                    )
                    return result
            except subprocess.TimeoutExpired:
                # If the AI introduced an infinite loop (e.g. while True without an exit)
                pass

        # Execution evaluation failed
        result.is_false_positive = True
        return result

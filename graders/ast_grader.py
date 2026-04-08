"""
AST Grader toolkit for Python Code Review parsing.
Extracts code blocks from LLM comments and validates their Python AST mathematically.
"""
import ast
import re
from typing import List

def extract_code_blocks(text: str) -> List[str]:
    """Extract code chunks formatted in standard markdown blocks."""
    if not text:
        return []
    # Block matches
    blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    # Inline matches (capture slightly longer snippets to avoid trivial things)
    inlines = re.findall(r"`([^`\n]{3,})`", text)
    return blocks + inlines

def validate_ast_pattern(code: str, target_nodes: List[str]) -> bool:
    """
    Given a snippet of code, parses it as an AST and checks if it contains
    the specific required Python AST nodes or function calls.
    """
    # Wrap snippet in an async function context to allow 'await' and 'async with' parsing
    wrapped_code = "async def __dummy_wrapper__():\n"
    for line in code.split("\n"):
        if line.strip():
            wrapped_code += f"    {line}\n"

    try:
        tree = ast.parse(wrapped_code)
    except SyntaxError:
        return False
        
    found_nodes = set()
    for node in ast.walk(tree):
        node_type = type(node).__name__
        found_nodes.add(node_type)
        
        # Capture specific identifiers (like `Lock`)
        if isinstance(node, ast.Name):
            found_nodes.add(node.id)
        elif isinstance(node, ast.Attribute):
            found_nodes.add(node.attr)

    return any(target in found_nodes for target in target_nodes)

def grade_suggestion_with_ast(suggestion_text: str, required_ast_traits: List[str]) -> bool:
    """Extracts code blocks from the suggestion and runs robust AST validation."""
    if not suggestion_text:
        return False
        
    code_snippets = extract_code_blocks(suggestion_text)
    
    # Fallback to evaluating the raw text directly if they didn't use markdown ticks
    if not code_snippets:
        code_snippets = [suggestion_text]
        
    for snippet in code_snippets:
        if validate_ast_pattern(snippet, required_ast_traits):
            return True
            
    return False

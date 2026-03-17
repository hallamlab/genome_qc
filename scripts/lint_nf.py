#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys


def fail(msg: str) -> None:
    print(f"LINT_FAIL: {msg}")
    sys.exit(1)


def lint_workflow_structure(text: str) -> None:
    lines = text.splitlines()
    workflow_idx = None
    main_idx = None
    emit_idx = None

    for i, line in enumerate(lines, 1):
        if line.strip() == "workflow {":
            workflow_idx = i
        if line.strip() == "main:":
            main_idx = i
        if line.strip() == "emit:":
            emit_idx = i

    if workflow_idx is None:
        fail("Missing workflow block")
    if main_idx is None:
        fail("Missing main: block")
    if emit_idx is None:
        fail("Missing emit: block")
    if not lines[main_idx - 1].startswith("    "):
        fail("main: must be indented 4 spaces")
    if not lines[emit_idx - 1].startswith("    "):
        fail("emit: must be indented 4 spaces")
    if emit_idx <= main_idx:
        fail("emit: must come after main:")

    # Ensure all non-empty lines between main: and emit: are indented >= 8 spaces
    for line in lines[main_idx:emit_idx - 1]:
        if not line.strip() or line.strip().startswith("//"):
            continue
        if not line.startswith("        "):
            fail(f"Line inside main not indented >= 8 spaces: {line}")


def lint_braces(text: str) -> None:
    # Very strict brace balance check (does not parse strings)
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                fail("Unbalanced braces: closing brace before opening")
    if depth != 0:
        fail("Unbalanced braces: missing closing brace(s)")


def main() -> None:
    if len(sys.argv) < 2:
        fail("Usage: lint_nf.py <path_to_nf>")
    path = Path(sys.argv[1])
    if not path.exists():
        fail(f"File not found: {path}")
    text = path.read_text()
    lint_braces(text)
    lint_workflow_structure(text)
    print("LINT_OK")


if __name__ == "__main__":
    main()

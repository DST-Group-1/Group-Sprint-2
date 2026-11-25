"""Copy a notebook with some top-level settings changed, so a batch job can run it unedited.

usage: python nbparam.py IN.ipynb OUT.ipynb [NAME=EXPRESSION ...] [--keep-pip]

Each NAME=EXPRESSION replaces the first top-level line `NAME = ...` of a code cell with
`NAME = EXPRESSION`. It stops with an error if NAME is not assigned on a line of its own, so a
renamed or multi-line setting is noticed rather than silently ignored.

Lines starting with %pip or !pip are commented out (unless --keep-pip): the environments are built
on the login node beforehand, and compute nodes may have no internet. Outputs of the original run
are cleared, so the executed copy shows only this run's results.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


def lines_of(cell: dict) -> list[str]:
    source = cell["source"]
    return ("".join(source) if isinstance(source, list) else source).splitlines(keepends=True)


def set_value(cells: list[dict], name: str, expression: str) -> None:
    try:                                          # a bad expression fails here, not halfway through the job
        ast.parse(f"{name} = {expression}")
    except SyntaxError as error:
        sys.exit(f"{name}={expression} is not valid Python: {error.msg}")
    pattern = re.compile(rf"^{re.escape(name)}\s*(:[^=]+)?=(?!=)")
    for cell in cells:
        lines = lines_of(cell)
        for i, line in enumerate(lines):
            if not pattern.match(line):
                continue
            try:
                ast.parse(line)
            except SyntaxError:
                sys.exit(f"{name} is assigned over several lines; set it in the notebook instead: {line.strip()}")
            end = "\n" if line.endswith("\n") else ""
            lines[i] = f"{name} = {expression}  # set by nbparam.py for the BluePebble run{end}"
            cell["source"] = lines
            print(f"  {name} = {expression}")
            return
    sys.exit(f"{name} is not assigned at the top level of any code cell")


def main(argv: list[str]) -> None:
    keep_pip = "--keep-pip" in argv
    args = [a for a in argv if a != "--keep-pip"]
    if len(args) < 2:
        sys.exit(__doc__)
    source, target, *assignments = args
    notebook = json.loads(Path(source).read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    print(f"{source} -> {target}")
    for assignment in assignments:
        name, sep, expression = assignment.partition("=")
        if not sep or not name.strip().isidentifier():
            sys.exit(f"expected NAME=EXPRESSION, got {assignment!r}")
        set_value(cells, name.strip(), expression.strip())
    for cell in cells:
        if not keep_pip:
            cell["source"] = [f"# (installed beforehand on BluePebble) {line}"
                              if line.lstrip().startswith(("%pip", "!pip")) else line for line in lines_of(cell)]
        cell["outputs"], cell["execution_count"] = [], None
    Path(target).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

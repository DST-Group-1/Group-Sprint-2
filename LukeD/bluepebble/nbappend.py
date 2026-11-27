"""Append cells to a notebook copy, so a batch job can add a step without editing the original.

usage: python nbappend.py NOTEBOOK.ipynb CELL.md|CELL.py [...]

Each file becomes one cell at the end of NOTEBOOK, in the order given: a .md file a markdown cell,
anything else a code cell. Use it on the copy nbparam.py writes, never on a teammate's original.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit(__doc__)
    path, cells = Path(argv[0]), argv[1:]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for name in cells:
        file = Path(name)
        kind = "markdown" if file.suffix == ".md" else "code"
        cell = {"cell_type": kind, "id": re.sub(r"[^A-Za-z0-9_-]", "-", f"appended-{file.name}")[:64],
                "metadata": {}, "source": file.read_text(encoding="utf-8").splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)
        print(f"  appended {kind} cell {file.name}")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Convierte un script en formato 'percent' (# %% / # %% [markdown]) a .ipynb y lo ejecuta.

    python tools/py2nb.py notebooks/iteracion_2/_py/01_exploracion.py notebooks/iteracion_2/01_exploracion.ipynb [--no-exec]
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def convert(src: Path) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "hms", "display_name": "Python (hms)", "language": "python"}
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    nb = convert(src)
    if "--no-exec" not in sys.argv:
        NotebookClient(nb, timeout=3600, kernel_name="hms",
                       resources={"metadata": {"path": str(dst.parent)}}).execute()
    nbformat.write(nb, dst)
    print("escrito", dst)

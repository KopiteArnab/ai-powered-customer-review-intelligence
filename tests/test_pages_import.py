"""
Verifies that all page scripts can be compiled and parsed without syntax errors.
"""

import ast
import glob
from pathlib import Path


def test_all_pages_valid_python_syntax():
    pages = glob.glob("pages/*.py")
    assert len(pages) == 7, f"Expected 7 pages, found {len(pages)}: {pages}"

    for page in pages:
        with open(page, "r", encoding="utf-8") as f:
            code = f.read()
        # Compile to AST to verify valid Python syntax
        tree = ast.parse(code, filename=page)
        assert tree is not None, f"Failed to parse {page}"


def test_app_py_syntax():
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code, filename="app.py")
    assert tree is not None

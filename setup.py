"""Compatibility shim for setuptools < 61.

Modern setuptools reads project metadata from ``pyproject.toml``.  The
fallback keeps ``pip install -e .`` working on older Python 3.10 toolchains
that predate PEP 621.
"""

from __future__ import annotations

from pathlib import Path

import setuptools


def _major_minor() -> tuple[int, int]:
    parts = setuptools.__version__.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


if _major_minor() < (61, 0):
    setuptools.setup(
        name="cetra",
        version="0.1.0.dev0",
        description="CETRA: learning to decompose, abstract, and recombine agent experience",
        long_description=Path("README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        package_dir={"": "src"},
        packages=setuptools.find_packages("src"),
        python_requires=">=3.10",
        install_requires=["requests>=2.31"],
    )
else:
    setuptools.setup()

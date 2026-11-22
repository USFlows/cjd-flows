"""Sphinx configuration for the CJD-Flows documentation.

Build locally with::

    pip install -e ".[docs,eval,onnx]"
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import cjd_flows  # noqa: E402

# -- Project information -----------------------------------------------------

project = "CJD-Flows"
copyright = "2023, Transferlab, appliedAI Institute for Europe"
author = "Faried Abu Zaid"
release = cjd_flows.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # before sphinx_autodoc_typehints
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / napoleon / typehints ------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
# optional dependencies that may be missing when the docs are built
autodoc_mock_imports = ["pyro"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_attr_annotations = True

typehints_use_signature = False
typehints_use_signature_return = False
always_document_param_types = False
typehints_defaults = None

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
intersphinx_timeout = 30

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# -- HTML --------------------------------------------------------------------

html_theme = "furo"
html_title = f"CJD-Flows {release}"
html_static_path: list = []
html_theme_options = {
    "source_repository": "https://github.com/USFlows/cjd-flows",
    "source_branch": "main",
    "source_directory": "docs/",
}

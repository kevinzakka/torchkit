"""Configuration file for the Sphinx documentation builder."""

import m2r
import torchkit

# -- Project information -----------------------------------------------------

project = "torchkit"
copyright = "2021, Kevin Zakka"
author = "Kevin Zakka"
release = torchkit.__version__

# -- General configuration ---------------------------------------------------

master_doc = "index"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "style_nav_header_background": "#06203A",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# ---------------------------------------------------------------------------


def docstring(app, what, name, obj, options, lines):
    md = "\n".join(lines)
    rst = m2r.convert(md)
    lines.clear()
    lines += rst.splitlines()


def setup(app):
    app.connect("autodoc-process-docstring", docstring)

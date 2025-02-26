"""docs."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.

from pathlib import Path
import shlex
import subprocess


package_dir = Path(__file__).resolve().parents[2] / "src/mmag"
# print(f"{package_dir=}")
# sys.path.insert(0, package_dir.as_posix())


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mmag"
copyright = "2025, Thomas Schrefl, Swapneel Amit Pathak, Andrea Petrocchi, Samuel Holt, Martin Lang, Hans Fangohr"
author = "Thomas Schrefl, Swapneel Amit Pathak, Andrea Petrocchi, Samuel Holt, Martin Lang, Hans Fangohr"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = []
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
autodoc_mock_imports = ["esys"]

# -- Extension config --------------------------------------------------------

# Autodoc
autodoc_default_options = {
    # Autodoc members
    "members": True,
    # Autodoc undocumented memebers
    "undoc-members": True,
    # Autodoc private memebers
    "private-members": True,
    # Autodoc special members (for the moment only __init__)
    "special-members": "__init__",
}

# No document TypeHints
autodoc_typehints = "none"

# Autosummary
autosummary_generate = True
autosummary_generate_overwrite = True

# MyST
myst_heading_anchors = 4


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Theme configuration -----------------------------------------------------

# Sidebar configuration
html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs.html"], "index": []}


# -- Run apidoc ------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html

source_dir = Path(".").resolve()
apidoc_dir = Path("mmag_autodoc")


cmd = shlex.split(
    (
        "sphinx-apidoc "
        f"-o {apidoc_dir} "  # apidoc directory
        f"{package_dir} "  # Package directory
        "-f"  # We force it so files are overwritten
    )
)
# print(f"Command: {' '.join(cmd)}")
res = subprocess.run(cmd, stderr=subprocess.PIPE)

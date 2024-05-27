__version__ = None
exec(open("../../gsplat/version.py", "r").read())

# -- Project information

project = "gsplat"
copyright = "2023, nerfstudio team"
author = "nerfstudio"

# Formatting!
#     0.1.30 => v0.1.30
#     dev => dev
if not __version__.isalpha():
    __version__ = "v" + __version__

# The full version, including alpha/beta/rc tags
release = ""

# -- General configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "furo"

# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Options for EPUB output
epub_show_urls = "footnote"

# typehints
autodoc_typehints = "description"

# citations
bibtex_bibfiles = ["references.bib"]

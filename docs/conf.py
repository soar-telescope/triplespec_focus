# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from importlib.metadata import version as get_version
from importlib.metadata import PackageNotFoundError
try:
    __version__ = get_version('triplespec_focus')
except PackageNotFoundError:
    __version__ = '0.0.0'

# -- Project information -----------------------------------------------------

project = 'TripleSpec Focus Calculator'
copyright = '2022, Simón Torres'
author = 'Simón Torres'
license = 'bsd3'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
]

autoclass_content = 'both'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


source_suffix = ['.rst']


master_doc = 'index'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'test_', 'tests']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "navbar_end": ["navbar-icon-links"],
    "icon_links": [
        {
            "name": "SOAR Docs Index",
            "url": "https://soardocs.readthedocs.io/",  # or relative path "../index"
            "icon": "fa fa-home",
        }
    ],
}

html_logo = '_static/soar_logo.png'
html_context = {'license': 'BSD 3-Clause License'}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

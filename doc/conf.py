# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
import toml

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tinycio'
copyright = '2023, Sam Izdat'
author = 'Sam Izdat'

data = None
version = '???'
release = '???'
with open('../pyproject.toml', 'r') as f:
    data = toml.load(f)
    version = data['project']['version']
    release = data['tool']['tinycio_about']['release']



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]

autosummary_generate = True

autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_logo = "images/tinycio_sm.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
html_css_files = [
    'css/patch.css',
]

autodoc_mock_imports = ["torch", "torchvision", "numpy", "scipy", "taichi"]

html_show_sphinx = False
html_show_copyright = False
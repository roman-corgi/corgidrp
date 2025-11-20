# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'corgidrp'
copyright = '2025, corgidrp Developers'
author = 'corgidrp Developers'
release = '3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'autoapi.extension',
]

# AutoAPI settings
autoapi_type = 'python'
autoapi_dirs = ['../../corgidrp']  # Path to your package relative to conf.py
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "link": "index",
        "image_light": "corgi.png",
        "image_dark": "corgi.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/roman-corgi/corgidrp.git",
            "icon": "fab fa-github-square",
        },
    ],
    "footer_start": ["copyright", "last-updated"],
    "secondary_sidebar_items": [],
    "header_links_before_dropdown": 8,
    "pygment_light_style": "tango",
    "pygment_dark_style": "monokai",
}

html_last_updated_fmt = "%Y %b %d at %H:%M:%S UTC"
html_show_sourcelink = False

html_sidebars = {
    "*": ["sidebar-nav-bs.html"],
    "index": [],
    "pages/install": ["page-toc"],
    "pages/tutorials": [],
    "pages/data_formats/index": [],
    "tutorials/**": ["page-toc", "sidebar-nav-bs.html"],
    "case_studies/**": ["page-toc", "sidebar-nav-bs.html"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_js_files = ['custom.js']

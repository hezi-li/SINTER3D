import os
import sys

sys.path.insert(0, os.path.abspath("../SINTER3D-master"))

project = "SINTER3D"
author = "hezi-li"

extensions = [
    "nbsphinx",
]

html_theme = "sphinx_rtd_theme"

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

nbsphinx_execute = "never"

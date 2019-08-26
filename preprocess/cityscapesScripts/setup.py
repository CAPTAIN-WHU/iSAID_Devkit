#!/usr/bin/env python
#
# Enable cython support for slightly faster eval scripts:
# python -m pip install cython numpy
# CYTHONIZE_EVAL= python setup.py build_ext --inplace
#
# For MacOS X you may have to export the numpy headers in CFLAGS
# export CFLAGS="-I /usr/local/lib/python3.6/site-packages/numpy/core/include $CFLAGS"

import os
from setuptools import setup, find_packages

include_dirs = []
ext_modules = []
if 'CYTHONIZE_EVAL' in os.environ:
    from Cython.Build import cythonize
    import numpy as np
    include_dirs = [np.get_include()]

    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"

    pyxFile = os.path.join("cityscapesscripts", "evaluation", "addToConfusionMatrix.pyx")
    ext_modules = cythonize(pyxFile)

with open("README.md") as f:
    readme = f.read()

config = {
    'name': 'cityscapesScripts',
    'description': 'Scripts for the Cityscapes Dataset',
    'long_description': readme,
    'long_description_content_type': "text/markdown",
    'author': 'Marius Cordts',
    'url': 'https://github.com/mcordts/cityscapesScripts',
    'author_email': 'mail@cityscapes-dataset.net',
    'license': 'https://github.com/mcordts/cityscapesScripts/blob/master/license.txt',
    'version': '1.0.8',
    'install_requires': ['numpy', 'matplotlib', 'pillow'],
    'setup_requires': ['setuptools>=18.0'],
    'packages': find_packages(),
    'scripts': [],
    'entry_points': {'gui_scripts': ['csViewer = cityscapesscripts.viewer.cityscapesViewer:main',
                                     'csLabelTool = cityscapesscripts.annotation.cityscapesLabelTool:main'],
                     'console_scripts': ['csEvalPixelLevelSemanticLabeling = cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling:main',
                                         'csEvalInstanceLevelSemanticLabeling = cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling:main',
                                         'csCreateTrainIdLabelImgs = cityscapesscripts.preparation.createTrainIdLabelImgs:main',
                                         'csCreateTrainIdInstanceImgs = cityscapesscripts.preparation.createTrainIdInstanceImgs:main']},
    'package_data': {'': ['icons/*.png']},
    'ext_modules': ext_modules,
    'include_dirs': include_dirs
}

setup(**config)

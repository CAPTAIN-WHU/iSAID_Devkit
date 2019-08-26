"""Environment helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

# Default value of the CMake install prefix
_CMAKE_INSTALL_PREFIX = '/usr/local'


def get_runtime_dir():
    """Retrieve the path to the runtime directory."""
    return os.getcwd()


def get_py_bin_ext():
    """Retrieve python binary extension."""
    return '.py'


def set_up_matplotlib():
    """Set matplotlib up."""
    import matplotlib
    # Use a non-interactive backend
    matplotlib.use('Agg')


def exit_on_error():
    """Exit from a detectron tool when there's an error."""
    sys.exit(1)

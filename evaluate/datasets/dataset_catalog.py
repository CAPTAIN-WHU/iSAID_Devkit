
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
   
    'isaid_patch_train': {
        IM_DIR:
            _DATA_DIR + '/train/images',
        ANN_FN:
            _DATA_DIR + '/annotation/instancesonly_filtered_train.json'
    },
    'isaid_patch_val': {
        IM_DIR:
            _DATA_DIR + '/val/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/annotation/instancesonly_filtered_val.json'
    },
    'isaid_patch_test': {
        IM_DIR:
            _DATA_DIR + '/test/images',
        ANN_FN:
            _DATA_DIR + '/test/instancesonly_filtered_test_paper.json'
    }
}

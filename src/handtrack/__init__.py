"""
Hand Tracker package for tracking hand landmarks and computing joint angles
"""

__version__ = "0.0.3.10"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/jshulgach/Hand-Landmark-Tracker"
__description__ = "Python package for hand landmark tracking and kinematics suite"

import importlib as _importlib

submodules = [
    # 'decomposition',
    'applications',
    'io',
    #'plotting',
    # 'control',
    'tracker',
    'processing',
    #'rhx_interface',
    #'samples',
    # 'stream',
]

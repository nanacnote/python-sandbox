"""
Simple search engine with a data cleaner functionality
"""
import os

# set base ref path and load env
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CORPUS_DIR = f'{BASE_DIR}/corpus'
INDEX_DIR = f'{BASE_DIR}/index'

STYLE_UNDERLINE = '\033[4m'
STYLE_BOLD = '\033[1m'
STYLE_HEADER = '\033[95m'
STYLE_BLUE = '\033[94m'
STYLE_CYAN = '\033[96m'
STYLE_GREEN = '\033[92m'
STYLE_WARNING = '\033[93m'
STYLE_FAIL = '\033[91m'
STYLE_CLEAR = '\033[0m'


__author__ = "Owusu K"
__version__ = "0.1.0"

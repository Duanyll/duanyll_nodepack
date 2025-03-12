"""Top-level package for duanyll_nodepack."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """duanyll"""
__email__ = "duanyll@outlook.com"
__version__ = "0.0.1"

from .src.duanyll_nodepack.nodes import NODE_CLASS_MAPPINGS
from .src.duanyll_nodepack.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

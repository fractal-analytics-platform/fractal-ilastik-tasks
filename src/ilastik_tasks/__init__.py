"""
Package description.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ilastik-tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"
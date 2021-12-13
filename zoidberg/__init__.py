try:
    from importlib.metadata import version, PackageNotFoundError
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version("zoidberg")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

import zoidberg

from . import field, fieldtracer, grid, plot
from .zoidberg import make_maps, write_maps

__all__ = [
    "__version__",
    "field",
    "fieldtracer",
    "grid",
    "make_maps",
    "plot",
    "write_maps",
    "zoidberg",
]

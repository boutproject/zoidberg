try:
    from importlib.metadata import PackageNotFoundError, version
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import PackageNotFoundError, version
try:
    __version__ = version("zoidberg")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

from . import field, fieldtracer, grid, plot, zoidberg
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

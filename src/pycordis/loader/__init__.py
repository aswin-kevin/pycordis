"""Built-in config loader: config files, entries, interpolation, plugin resolution.

See spec §6.6. :class:`Loader` (the ``loader`` service) reconciles the entry tree
into fibers; the other modules are the pure data layer and never create fibers.
"""

from pycordis.loader.entry import Entry, LoaderOptions, Overlay
from pycordis.loader.files import LoaderConfigError, load_tree, read_config
from pycordis.loader.interpolate import expand
from pycordis.loader.resolve import import_plugin
from pycordis.loader.service import LoadedEntry, Loader, LoaderConfig

__all__ = [
    "Entry",
    "LoadedEntry",
    "Loader",
    "LoaderConfig",
    "LoaderConfigError",
    "LoaderOptions",
    "Overlay",
    "expand",
    "import_plugin",
    "load_tree",
    "read_config",
]

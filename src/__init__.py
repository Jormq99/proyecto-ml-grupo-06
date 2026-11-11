"""Módulos del proyecto HMS."""

from importlib import import_module
from types import ModuleType

__version__ = "1.0.0"
__author__ = "Jordani Mejía Quirós, Luis Fernando Solano Coto"

_MODULES = {"data_preparation", "features", "train", "evaluate"}

__all__ = sorted(_MODULES)


def __getattr__(name: str) -> ModuleType:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module

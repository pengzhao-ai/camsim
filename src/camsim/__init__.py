"""camsim: deterministic camera sensor and ISP simulation with IQ metrics.

The public namespace is deliberately lazy (PEP 562). Importing ``camsim`` loads
only version metadata; public types are resolved on first attribute access.
This keeps ``import camsim.metrics`` free of the sensor, ISP, scene, and optics
packages, which the metrics harness is required not to depend on.

Later phases register public names in ``_LAZY_EXPORTS`` as
``name -> (module path, attribute name)`` pairs.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

from ._version import __version__

if TYPE_CHECKING:
    # Static analysers and the documentation builder cannot execute the lazy
    # ``__getattr__`` below, so mirror the public map with type-checking-only
    # aliases. This block is never imported at runtime and therefore adds no
    # cost to ``import camsim``.
    from camsim.camera.camera import Camera as Camera
    from camsim.config.schema import LensConfig as LensConfig
    from camsim.core.spectral import SpectralIrradiance as SpectralIrradiance
    from camsim.core.spectral import WavelengthGrid as WavelengthGrid
    from camsim.isp.pipeline import ISP as ISP
    from camsim.sensor import Sensor as Sensor

__all__: Final[list[str]] = [
    "ISP",
    "Camera",
    "LensConfig",
    "Sensor",
    "SpectralIrradiance",
    "WavelengthGrid",
    "__version__",
]

# Public name -> (module path, attribute name). Populated as public types land.
_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "Camera": ("camsim.camera.camera", "Camera"),
    "ISP": ("camsim.isp.pipeline", "ISP"),
    "LensConfig": ("camsim.config.schema", "LensConfig"),
    "Sensor": ("camsim.sensor", "Sensor"),
    "SpectralIrradiance": ("camsim.core.spectral", "SpectralIrradiance"),
    "WavelengthGrid": ("camsim.core.spectral", "WavelengthGrid"),
}


def __getattr__(name: str) -> Any:
    """Resolve a lazily exported public name on first access."""
    try:
        module_path, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_path), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in ``dir(camsim)`` and tab completion."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))

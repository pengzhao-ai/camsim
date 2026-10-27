"""Bounded-memory guards for spectral simulation and sensor integration.

A spectral cube is deceptively large. A 512 x 512 sensor image rendered at the
conventional 4x oversampling factor carries ``(512*4)**2`` spatial samples for
every wavelength; on the 95-sample 360-830 nm grid that is about 398 million
float64 values, or roughly 3.2 GB for the cube alone. A 50 MP frame at 1x
oversampling is about 38 GB. The scene renderer rasterizes a target at that
resolution, multiplies it by an illuminant, convolves it with a PSF, and boxes
it down; the sensor then casts the cube to float64 and forms one channel plane
per CFA colour. Each step can transiently double the working set, so a request
that looks reasonable in pixel count can exhaust memory and be killed by the
operating system before a Python exception can explain what happened.

v0.1.0 does not implement chunked spectral processing. Instead, both
:func:`camsim.scene.targets.render` and :meth:`camsim.sensor.Sensor.expose`
estimate the peak byte count from the requested shape *before* allocating the
large arrays and raise a clear ``ValueError`` when the estimate exceeds a
configurable limit. The estimate is deliberately conservative: it counts the
input cube, one working copy, and the output/downsample buffer, using
``float64`` itemsize because the numerical paths promote to float64. It is an
engineering guard, not a promise that a request just below the limit will fit:
the operating system, other processes, and library workspaces also need room.

The limit defaults to 8 GiB and can be overridden per call with
``max_memory_bytes=`` or for a process with the
``CAMSIM_MAX_SIMULATION_BYTES`` environment variable. The default admits the
validated 1024 x 1024, 1x-oversampling benchmark (about 2.2 GiB estimated peak)
and the 512 x 512, 4x-oversampling scene (about 6.6 GB) on a machine with
enough physical memory, while rejecting a 50 MP spectral render (about 106 GiB
estimated peak; one float64 cube alone is about 38 GB).
A caller who has the RAM can raise the limit explicitly; a caller who does not
should reduce ``shape`` or ``oversampling``, or process the scene in chunks
outside camsim and feed the resulting ``SpectralIrradiance`` to the sensor.

Failure artefacts guarded here: an oversized request that triggers the OOM
killer with no diagnostic; a guard that silently clamps the request instead of
raising; a limit parsed from an environment variable that is negative, zero, or
not an integer; and an estimate that omits the float64 promotion or the output
buffer and therefore understates the true peak.
"""

from __future__ import annotations

import os
from typing import Final

import numpy as np

__all__ = [
    "DEFAULT_SIMULATION_MEMORY_BYTES",
    "MEMORY_LIMIT_ENV",
    "ensure_memory_budget",
    "estimate_scene_peak_bytes",
    "estimate_sensor_peak_bytes",
    "resolve_memory_limit",
]

DEFAULT_SIMULATION_MEMORY_BYTES: Final[int] = 8 * 1024**3
"""Default peak-memory budget for one scene render or sensor exposure (8 GiB)."""

MEMORY_LIMIT_ENV: Final[str] = "CAMSIM_MAX_SIMULATION_BYTES"
"""Environment variable overriding the default peak-memory budget, in bytes."""

_FLOAT64_BYTES: Final[int] = 8


def resolve_memory_limit(max_memory_bytes: int | None) -> int:
    """Return the effective peak-memory limit in bytes.

    Parameters
    ----------
    max_memory_bytes:
        Per-call override. ``None`` reads ``CAMSIM_MAX_SIMULATION_BYTES`` when
        it is set, otherwise uses :data:`DEFAULT_SIMULATION_MEMORY_BYTES`.
        A supplied value must be a positive integer.

    Raises
    ------
    ValueError
        If the explicit value is not a positive integer, or the environment
        variable is set to something that is not a positive integer. Silently
        falling back to the default would hide a mistyped deployment setting.
    """
    if max_memory_bytes is not None:
        return _positive_limit(max_memory_bytes, "max_memory_bytes")
    raw = os.environ.get(MEMORY_LIMIT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_SIMULATION_MEMORY_BYTES
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{MEMORY_LIMIT_ENV} must be a positive integer number of bytes; got {raw!r}"
        ) from exc
    return _positive_limit(parsed, MEMORY_LIMIT_ENV)


def estimate_scene_peak_bytes(
    shape: tuple[int, int],
    wavelength_count: int,
    oversampling: int,
) -> int:
    """Estimate the peak bytes of an oversampled scene render.

    The target is rasterized at ``oversampling`` samples per sensor pixel in
    each axis. The estimator counts the reflectance/target array, one
    float64 working copy of the spectral cube, and the sensor-grid output
    buffer. A PSF or crosstalk convolution can add another temporary copy, so
    the estimate already includes one working copy beyond the input.
    """
    height, width = _positive_pair(shape)
    samples = height * width * oversampling * oversampling * wavelength_count
    cube_bytes = samples * _FLOAT64_BYTES
    output_bytes = height * width * wavelength_count * _FLOAT64_BYTES
    return int(2 * cube_bytes + output_bytes)


def estimate_sensor_peak_bytes(
    shape: tuple[int, int],
    wavelength_count: int,
    channel_count: int,
) -> int:
    """Estimate the peak bytes of one :meth:`Sensor.expose` integration.

    The sensor casts the input cube to float64, forms one float64 channel plane
    per CFA colour, and keeps the mosaicked mean and quantized planes. The
    estimate includes two full cube copies (input and working), two plane-sized
    float64 buffers per channel, and one ``uint16`` code plane per channel.
    """
    height, width = _positive_pair(shape)
    cube_bytes = height * width * wavelength_count * _FLOAT64_BYTES
    plane_bytes = height * width * _FLOAT64_BYTES
    code_bytes = height * width * 2
    return int(2 * cube_bytes + channel_count * (2 * plane_bytes + code_bytes) + 2 * plane_bytes)


def ensure_memory_budget(
    required_bytes: int,
    limit_bytes: int,
    *,
    context: str,
    guidance: str,
) -> None:
    """Raise a clear ``ValueError`` when ``required_bytes`` exceeds the limit.

    ``context`` names the operation (for example ``"scene.render"``) and
    ``guidance`` states the concrete remedy. The message reports both values in
    GiB so the caller does not have to divide by 1024**3 to understand it.
    """
    if required_bytes <= limit_bytes:
        return
    required_gib = required_bytes / float(1024**3)
    limit_gib = limit_bytes / float(1024**3)
    raise ValueError(
        f"{context} estimated peak memory {required_gib:.2f} GiB exceeds the "
        f"configured simulation limit {limit_gib:.2f} GiB. {guidance} "
        f"Raise the limit with max_memory_bytes= or {MEMORY_LIMIT_ENV} only if "
        "the machine has enough RAM; v0.1.0 does not chunk spectral cubes."
    )


def _positive_limit(value: object, field: str) -> int:
    """Return ``value`` as a positive Python integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a positive integer number of bytes; got {value!r}")
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{field} must be a positive integer number of bytes; got {limit}")
    return limit


def _positive_pair(shape: object) -> tuple[int, int]:
    """Return a validated ``(height, width)`` pair of positive integers."""
    try:
        dimensions = np.asarray(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"shape must be a (height, width) pair of positive integers; got {shape!r}"
        ) from exc
    if dimensions.shape != (2,) or not np.issubdtype(dimensions.dtype, np.integer):
        raise ValueError(
            f"shape must be a (height, width) pair of positive integers; got {shape!r}"
        )
    height = int(dimensions[0])
    width = int(dimensions[1])
    if height <= 0 or width <= 0:
        raise ValueError(
            f"shape must be a (height, width) pair of positive integers; got {shape!r}"
        )
    return height, width

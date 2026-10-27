"""RAW sensor frames, ISP stage traces, and display-referred rendered images.

A camera pipeline moves through three representations that must never be
confused:

* **Scene radiance** is physical light. After the lens and colour filter array,
  photon counts become electrons in each photodiode.
* **RAW data** are digital numbers (DN) produced by the ADC. They are integer
  codes that include a black-level offset, so the signal-bearing value is
  ``DN - black_level_dn``. :class:`RawFrame` couples a 2-D ``uint16`` array to
  every acquisition and geometry parameter the ISP needs, plus optional
  defective-pixel and PDAF masks and a hash of the sensor configuration.
* **Rendered images** are display referred. A non-linear transfer function
  (sRGB OETF or a tone curve) has already been applied, so values are floats in
  ``[0, 1]`` and are not proportional to scene radiance. :class:`RenderedImage`
  keeps the optional pre-transfer ``linear_rgb`` buffer so metrics can compute
  colour differences in a linear space.

Typical parameter ranges: phone and consumer sensors quantize to 10 or 12 bits
(8-bit legacy modes and 14-/16-bit scientific or industrial parts exist);
black level is commonly 64 DN at 10 bits and 256 DN at 12 bits; analog gain
spans 0-30 dB; digital gain 1-8x; exposures run from microseconds to seconds;
sensor temperature is reported in degrees Celsius over roughly -20 to 70 C;
pixel pitches range from 0.7 um in dense phone arrays to about 10 um in large
machine-vision pixels.

Failure artefacts guarded here: values above ``2**bit_depth - 1`` mean the
data were scaled for a deeper container (or an 8-bit image was widened without
scaling), and truncating them loses highlights; a black level that is not
subtracted leaves a pedestal that becomes a colour cast after white balance;
treating RAW DN as display values applies gamma twice; and confusing a mono
``(H, W, 1)`` image with an RGB ``(H, W, 3)`` one silently mis-indexes
channels. Each condition raises with a message naming the offending field
instead of being coerced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Final, Literal, Self, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

ColorSpace: TypeAlias = Literal["sRGB", "DisplayP3"]
"""Output colour spaces supported by :class:`RenderedImage`."""

ChannelLayout: TypeAlias = Literal["RGB", "MONO"]
"""Channel layouts supported by :class:`RenderedImage`."""

_MIN_BIT_DEPTH: Final[int] = 8
_MAX_BIT_DEPTH: Final[int] = 16

# Rendered data should already be clipped to the display range, but a final
# tone curve evaluated in float64 and stored in float32 can land a few ULPs
# outside it. A 1e-6 absolute allowance absorbs that round-off without
# accepting visibly out-of-range (for example 0-255) data.
_DISPLAY_TOLERANCE: Final[float] = 1e-6


# The public contract is a str-valued Enum rather than StrEnum: member values
# are ordinary strings for JSON sidecars and config consumers, while the enum
# keeps lookup and type identity explicit.
class CFAPattern(str, Enum):  # noqa: UP042
    """Colour-filter-array tiles and CFA-free readout modes.

    Bayer patterns are 2x2 tiles named in row-major order from the top-left
    pixel: ``RGGB`` places red at (0, 0), green at (0, 1) and (1, 0), and blue
    at (1, 1). The four Bayer names correspond to the four possible placements
    of the red site. ``RCCC`` is an automotive pattern with one red and three
    clear sites per tile. ``MONO`` denotes a sensor with no colour filters at
    all: it has higher effective quantum efficiency and needs no demosaicing.
    """

    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"
    MONO = "MONO"
    RCCC = "RCCC"


@dataclass(eq=False)
class StageOutput:
    """One named ISP stage's output in a processing trace.

    Parameters
    ----------
    data:
        Stage result. ISP stages normally store a float array whose trailing
        axis is the channel count; the type is intentionally permissive so
        stages such as the colour-correction matrix can record a small matrix.
    description:
        Human-readable note naming the operation applied, for example
        ``"sRGB OETF applied"``.
    metadata:
        Stage-specific parameters or statistics. Defaults to an empty dict.

    Notes
    -----
    Trace outputs are snapshots: later stages must not mutate arrays received
    from an earlier stage, otherwise the trace no longer describes the
    processing order that produced the final image.
    """

    data: Any
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dict; got {type(self.metadata).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping with array data as nested lists."""
        data = self.data.tolist() if isinstance(self.data, np.ndarray) else self.data
        return {
            "data": data,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a stage output from :meth:`to_dict` output."""
        data = payload["data"]
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)
        return cls(
            data=data,
            description=payload["description"],
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(eq=False)
class RawFrame:
    """A self-describing RAW sensor frame.

    Parameters
    ----------
    data:
        Sensor readout after gain and quantization, shape ``(H, W)``, dtype
        ``uint16``. The array stores ADC codes including ``black_level_dn``;
        it is not an electron count and is not proportional to display
        brightness. Values must not exceed ``2**bit_depth - 1``.
    cfa:
        Colour-filter pattern of the sensor. A string matching a
        :class:`CFAPattern` member is accepted and normalised to the enum.
    bit_depth:
        ADC resolution in bits, 8-16 inclusive. Every code in a valid frame
        lies in ``[0, 2**bit_depth - 1]``.
    black_level_dn:
        ADC code that corresponds to zero signal,
        ``0 <= black_level_dn < 2**bit_depth``. Subtract it before gain
        correction, white balance, or demosaicing.
    exposure_s:
        Integration time in seconds, finite and positive.
    analog_gain_db:
        Analog programmable-gain-amplifier setting in decibels, finite and
        non-negative (0 dB is unity gain).
    digital_gain:
        Digital multiplier applied after the ADC, finite and positive.
    mode:
        Sensor readout mode, for example ``"LCG"`` (low conversion gain) or
        ``"HCG"`` (high conversion gain), as defined by the active sensor
        configuration.
    temperature_c:
        Sensor temperature in degrees Celsius; finite and may be negative.
    seed:
        Non-negative seed of the random stream that generated this frame.
    frame_index:
        Non-negative index of the frame within its capture sequence.
    pixel_pitch_um:
        Square pixel pitch in micrometres, finite and positive.
    defect_mask:
        Optional boolean mask, same shape as ``data``, marking defective or
        hot pixels that a defect-correction stage should replace.
    pdaf_mask:
        Optional boolean mask, same shape as ``data``, marking phase-detection
        autofocus pixels whose signal is not a normal imaging sample.
    sensor_config_hash:
        Non-empty identifier of the sensor configuration that produced the
        frame. It lets caches and reports detect a configuration change.

    Notes
    -----
    Equality is identity based because frames own large arrays; compare fields
    (or arrays with :func:`numpy.array_equal`) explicitly.
    """

    data: NDArray[np.uint16]
    cfa: CFAPattern
    bit_depth: int
    black_level_dn: float
    exposure_s: float
    analog_gain_db: float
    digital_gain: float
    mode: str
    temperature_c: float
    seed: int
    frame_index: int
    pixel_pitch_um: float
    defect_mask: NDArray[np.bool_] | None = field(default=None, kw_only=True)
    pdaf_mask: NDArray[np.bool_] | None = field(default=None, kw_only=True)
    sensor_config_hash: str

    def __post_init__(self) -> None:
        self.bit_depth = _validate_bit_depth(self.bit_depth)
        self.data = _validate_raw_data(self.data, self.bit_depth)
        self.cfa = _validate_cfa(self.cfa)
        self.black_level_dn = _validate_black_level(self.black_level_dn, self.bit_depth)
        self.exposure_s = _validate_positive_real(self.exposure_s, "exposure_s (exposure)")
        self.analog_gain_db = _validate_non_negative_real(
            self.analog_gain_db, "analog_gain_db (analog gain)"
        )
        self.digital_gain = _validate_positive_real(
            self.digital_gain, "digital_gain (digital gain)"
        )
        self.mode = _validate_non_empty_text(self.mode, "mode")
        self.temperature_c = _validate_finite_real(
            self.temperature_c, "temperature_c (temperature)"
        )
        self.seed = _validate_non_negative_int(self.seed, "seed")
        self.frame_index = _validate_non_negative_int(self.frame_index, "frame_index")
        self.pixel_pitch_um = _validate_positive_real(
            self.pixel_pitch_um, "pixel_pitch_um (pixel pitch)"
        )
        self.defect_mask = _validate_mask(self.defect_mask, "defect_mask", self.data.shape)
        self.pdaf_mask = _validate_mask(self.pdaf_mask, "pdaf_mask", self.data.shape)
        self.sensor_config_hash = _validate_non_empty_text(
            self.sensor_config_hash, "sensor_config_hash"
        )

    def with_data(self, data: NDArray[np.uint16]) -> RawFrame:
        """Return a copy with ``data`` replaced and all metadata preserved.

        The replacement data is validated exactly like constructor data, so a
        caller cannot smuggle floats or out-of-range codes into the pipeline.
        Masks are tied to pixel positions, so changing the data shape requires
        clearing ``defect_mask`` and ``pdaf_mask`` first.
        """
        return replace(self, data=data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping, including data and masks as lists."""
        return {
            "data": self.data.tolist(),
            "cfa": self.cfa.value,
            "bit_depth": self.bit_depth,
            "black_level_dn": self.black_level_dn,
            "exposure_s": self.exposure_s,
            "analog_gain_db": self.analog_gain_db,
            "digital_gain": self.digital_gain,
            "mode": self.mode.value if isinstance(self.mode, Enum) else self.mode,
            "temperature_c": self.temperature_c,
            "seed": self.seed,
            "frame_index": self.frame_index,
            "pixel_pitch_um": self.pixel_pitch_um,
            "defect_mask": None if self.defect_mask is None else self.defect_mask.tolist(),
            "pdaf_mask": None if self.pdaf_mask is None else self.pdaf_mask.tolist(),
            "sensor_config_hash": self.sensor_config_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a frame from :meth:`to_dict` output or decoded JSON."""
        return cls(
            data=_payload_uint16_data(payload["data"]),
            cfa=payload["cfa"],
            bit_depth=payload["bit_depth"],
            black_level_dn=payload["black_level_dn"],
            exposure_s=payload["exposure_s"],
            analog_gain_db=payload["analog_gain_db"],
            digital_gain=payload["digital_gain"],
            mode=payload["mode"],
            temperature_c=payload["temperature_c"],
            seed=payload["seed"],
            frame_index=payload["frame_index"],
            pixel_pitch_um=payload["pixel_pitch_um"],
            defect_mask=_payload_mask(payload.get("defect_mask")),
            pdaf_mask=_payload_mask(payload.get("pdaf_mask")),
            sensor_config_hash=payload["sensor_config_hash"],
        )


@dataclass(eq=False)
class RenderedImage:
    """A display-referred image ready for viewing or metrics.

    Parameters
    ----------
    data:
        Display-referred pixels after the output transfer function. Shape is
        ``(H, W, 3)`` for RGB and ``(H, W, 1)`` for mono, stored as float32
        within ``[0, 1]``. A tolerance of 1e-6 absorbs round-off before the
        final clip; values are not linear light and must not be fed to
        algorithms that expect radiance.
    color_space:
        Output colour space, ``"sRGB"`` or ``"DisplayP3"``. It fixes the
        primaries and transfer function used to interpret both ``data`` and
        ``linear_rgb``.
    channels:
        ``"RGB"`` for a three-channel colour image or ``"MONO"`` for a single
        luminance channel.
    linear_rgb:
        Optional pre-transfer linear RGB buffer with the same shape as
        ``data``. It may exceed 1.0 (highlights are not clipped) and is the
        buffer colour metrics linearize from; it is not display data.
    stages:
        Optional ISP trace mapping stage names to :class:`StageOutput`
        snapshots. ``None`` when tracing is disabled.
    metadata:
        Free-form capture and processing provenance such as the white point,
        exposure, or ISP configuration hash. Defaults to an empty dict.

    Notes
    -----
    Equality is identity based because images own large arrays.
    """

    data: NDArray[np.float32]
    color_space: ColorSpace
    channels: ChannelLayout
    linear_rgb: NDArray[np.floating[Any]] | None = None
    stages: dict[str, StageOutput] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.color_space not in ("sRGB", "DisplayP3"):
            raise ValueError(f"color_space must be 'sRGB' or 'DisplayP3'; got {self.color_space!r}")
        if self.channels not in ("RGB", "MONO"):
            raise ValueError(f"channels must be 'RGB' or 'MONO'; got {self.channels!r}")
        if not isinstance(self.data, np.ndarray):
            raise ValueError(
                f"data must be a float32 display-referred array; got {type(self.data).__name__}"
            )
        if not np.issubdtype(self.data.dtype, np.floating):
            raise ValueError(
                f"data must be a float32 display-referred array; got dtype {self.data.dtype}"
            )
        expected_channels = 3 if self.channels == "RGB" else 1
        if self.data.ndim != 3 or self.data.shape[2] != expected_channels:
            raise ValueError(
                f"channels={self.channels!r} requires data shape "
                f"(H, W, {expected_channels}); got {self.data.shape}"
            )
        if self.data.shape[0] == 0 or self.data.shape[1] == 0:
            raise ValueError(f"data must contain at least one pixel; got shape {self.data.shape}")
        display = np.asarray(self.data, dtype=np.float32)
        if not np.all(np.isfinite(display)):
            raise ValueError(
                "data display values must be finite and within the display range [0, 1]"
            )
        minimum = float(display.min())
        maximum = float(display.max())
        if minimum < -_DISPLAY_TOLERANCE or maximum > 1.0 + _DISPLAY_TOLERANCE:
            raise ValueError(
                "data display values must lie within the display range [0, 1] "
                f"(tolerance {_DISPLAY_TOLERANCE:g}); got [{minimum:g}, {maximum:g}]"
            )
        self.data = display
        if self.linear_rgb is not None:
            if not isinstance(self.linear_rgb, np.ndarray):
                raise ValueError("linear_rgb must be a numpy array matching data.shape")
            if not np.issubdtype(self.linear_rgb.dtype, np.floating):
                raise ValueError(
                    f"linear_rgb must be a floating-point array; got dtype {self.linear_rgb.dtype}"
                )
            if self.linear_rgb.shape != display.shape:
                raise ValueError(
                    f"linear_rgb shape {self.linear_rgb.shape} must match data shape "
                    f"{display.shape}"
                )
        if self.stages is not None and not isinstance(self.stages, dict):
            raise ValueError("stages must be a dict[str, StageOutput] or None")
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dict; got {type(self.metadata).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping with arrays as nested lists."""
        stages: dict[str, Any] | None
        if self.stages is None:
            stages = None
        else:
            stages = {name: stage.to_dict() for name, stage in self.stages.items()}
        return {
            "data": self.data.tolist(),
            "color_space": self.color_space,
            "channels": self.channels,
            "linear_rgb": None if self.linear_rgb is None else self.linear_rgb.tolist(),
            "stages": stages,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct an image from :meth:`to_dict` output or decoded JSON."""
        linear_payload = payload.get("linear_rgb")
        linear_rgb = (
            None if linear_payload is None else np.asarray(linear_payload, dtype=np.float32)
        )
        stages_payload = payload.get("stages")
        stages: dict[str, StageOutput] | None = None
        if stages_payload is not None:
            if not isinstance(stages_payload, Mapping):
                raise ValueError("stages must be a mapping of stage name to StageOutput")
            stages = {
                str(name): StageOutput.from_dict(stage) for name, stage in stages_payload.items()
            }
        return cls(
            data=np.asarray(payload["data"], dtype=np.float32),
            color_space=payload["color_space"],
            channels=payload["channels"],
            linear_rgb=linear_rgb,
            stages=stages,
            metadata=dict(payload.get("metadata") or {}),
        )


def _validate_bit_depth(value: object) -> int:
    """Return an ADC bit depth in ``[8, 16]``, rejecting bools and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"bit depth must be an integer between 8 and 16; got {value!r}")
    depth = int(value)
    if not _MIN_BIT_DEPTH <= depth <= _MAX_BIT_DEPTH:
        raise ValueError(f"bit depth must be between 8 and 16; got {depth}")
    return depth


def _validate_raw_data(value: object, bit_depth: int) -> NDArray[np.uint16]:
    """Return frame data after shape, dtype, and code-range checks."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"data must be a 2-D uint16 numpy array; got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(f"data must be a 2-D (H, W) array; got shape {value.shape}")
    if value.dtype != np.uint16:
        raise ValueError(f"data must have dtype uint16; got dtype {value.dtype}")
    if value.size == 0:
        raise ValueError("data must contain at least one pixel")
    maximum = (1 << bit_depth) - 1
    largest = int(value.max())
    if largest > maximum:
        raise ValueError(
            f"data values exceed the {bit_depth}-bit maximum allowed by bit depth "
            f"({maximum} DN); got {largest}"
        )
    return cast(NDArray[np.uint16], value)


def _validate_cfa(value: object) -> CFAPattern:
    """Return ``value`` as a :class:`CFAPattern`, accepting member strings."""
    if isinstance(value, CFAPattern):
        return value
    if isinstance(value, str):
        try:
            return CFAPattern(value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in CFAPattern)
            raise ValueError(f"cfa must be one of {allowed}; got {value!r}") from exc
    raise ValueError(f"cfa must be a CFAPattern; got {type(value).__name__}")


def _validate_black_level(value: object, bit_depth: int) -> float:
    """Return a black level in ``[0, 2**bit_depth)``."""
    level = _validate_finite_real(value, "black level (black_level_dn)")
    if level < 0.0 or level >= float(1 << bit_depth):
        raise ValueError(
            "black level (black_level_dn) must satisfy 0 <= black_level_dn < 2**bit_depth; "
            f"got {level!r} for bit depth {bit_depth}"
        )
    return level


def _validate_finite_real(value: object, field: str) -> float:
    """Return ``value`` as a finite Python float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a real number; got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return result


def _validate_positive_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, strictly positive float."""
    result = _validate_finite_real(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive; got {result!r}")
    return result


def _validate_non_negative_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, non-negative float."""
    result = _validate_finite_real(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative; got {result!r}")
    return result


def _validate_non_negative_int(value: object, field: str) -> int:
    """Return ``value`` as a non-negative Python int."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer; got {result}")
    return result


def _validate_non_empty_text(value: object, field: str) -> str:
    """Return a non-empty, non-whitespace string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string; got {value!r}")
    return value


def _validate_mask(value: object, field: str, shape: tuple[int, ...]) -> NDArray[np.bool_] | None:
    """Return a boolean mask matching ``shape``, or ``None``."""
    if value is None:
        return None
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"{field} must be None or a boolean numpy array; got {type(value).__name__}"
        )
    if value.dtype != np.bool_:
        raise ValueError(f"{field} must have dtype bool; got dtype {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{field} shape {value.shape} must match data shape {shape}")
    return cast(NDArray[np.bool_], value)


def _payload_uint16_data(value: object) -> NDArray[np.uint16]:
    """Convert JSON integer lists to uint16 without silent truncation."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data must be a rectangular 2-D array; {exc}") from exc
    if array.dtype == np.uint16:
        return array
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"data must be an integer array; got dtype {array.dtype}")
    if array.size:
        largest = int(array.max())
        smallest = int(array.min())
        if smallest < 0 or largest > int(np.iinfo(np.uint16).max):
            raise ValueError("data values must be non-negative integers that fit in uint16")
    return cast(NDArray[np.uint16], array.astype(np.uint16))


def _payload_mask(value: object) -> NDArray[np.bool_] | None:
    """Convert a JSON boolean list to an array, leaving arrays for validation."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return cast(NDArray[np.bool_], value)
    try:
        return cast(NDArray[np.bool_], np.asarray(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mask must be a boolean array; {exc}") from exc

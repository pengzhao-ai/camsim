"""Persist display-referred rendered images as PNG/TIFF plus a JSON sidecar.

A :class:`~camsim.core.frames.RenderedImage` holds values *after* the output
transfer function: floats in ``[0, 1]`` that are proportional to display
intensity, not to scene radiance. A file format stores integer codes, so this
module quantizes and records exactly how it did so:

* **PNG** is written at 8 bits per channel. That is the format's conventional
  container, and its 256 levels limit the measurable colour and noise
  resolution for anything beyond visual inspection.
* **TIFF** is written at 16 bits for a monochrome image and at 8 bits for RGB,
  because Pillow cannot reliably write a three-channel 16-bit TIFF. The sidecar
  records the bit depth actually written, so a reader never assumes precision
  the file does not have.
* **Lossy formats are rejected.** JPEG, WebP, and similar codecs alter pixel
  values non-uniformly; running an MTF, SNR, or delta-E metric on a file that
  went through DCT quantization would measure the codec, not the camera.

The ``<stem>.json`` sidecar carries ``kind``, ``schema_version``, ``format``,
``bit_depth``, ``color_space``, ``channels``, the file name, and a JSON-safe
copy of the image metadata. It deliberately omits ``linear_rgb`` and ``stages``:
those are analysis and diagnostics buffers, not display data, and a TIFF cannot
represent the unclipped linear values anyway. Losing the sidecar still leaves a
viewable image; :func:`load_image` then infers 8/16-bit gray versus RGB and
assumes sRGB, which is correct for a file written here but not guaranteed for
one produced elsewhere.

Failure artefacts guarded here: treating the 8-bit display codes as linear
light (every noise and MTF metric then reads the sRGB toe as sensor behaviour);
saving a gamma-encoded buffer as ``.npy`` and later loading it as linear; a
sidecar from another image supplying the wrong colour space or channel layout;
and silently accepting a JPEG that a collaborator renamed to ``.png``. Values
are clipped only at the display boundary, where the contract already requires
``[0, 1]``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import IO, Any, Final, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from camsim.core.frames import ChannelLayout, ColorSpace, RenderedImage

__all__ = ["load_image", "save_image"]

_SCHEMA_VERSION: Final[str] = "1.0"
"""Sidecar schema version understood by this loader."""

_KIND: Final[str] = "camsim.rendered_image"
"""Sidecar ``kind`` marker, distinct from the RAW-frame sidecar."""

_FORMAT_BY_SUFFIX: Final[dict[str, str]] = {
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
"""The only accepted containers. Everything else is lossy or ambiguous."""

_CHANNEL_LAYOUTS: Final[frozenset[str]] = frozenset({"RGB", "MONO"})
"""Channel layouts a rendered image can declare."""

_COLOR_SPACES: Final[frozenset[str]] = frozenset({"sRGB", "DisplayP3"})
"""Output colour spaces supported by :class:`RenderedImage`."""

_READABLE_MODES: Final[frozenset[str]] = frozenset({"L", "RGB", "I;16", "I;16B", "I;16L"})
"""Pillow modes this loader can turn back into display values."""

_ACCEPTED_FORMATS: Final[frozenset[str]] = frozenset({"PNG", "TIFF"})
"""Formats Pillow is allowed to report after opening the file."""


def _atomic_write(target: Path, writer: Callable[[IO[bytes]], None]) -> None:
    """Write ``target`` through a sibling temporary file and an atomic rename."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(target: Path, payload: Mapping[str, Any]) -> None:
    """Serialize ``payload`` as strict, deterministic JSON."""
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=True,
    )
    data = (text + "\n").encode("utf-8")

    def writer(handle: IO[bytes]) -> None:
        handle.write(data)

    _atomic_write(target, writer)


def _json_safe(value: object, where: str = "metadata") -> Any:
    """Return a JSON-safe copy of a metadata value, converting NumPy containers.

    NumPy scalars become Python scalars, arrays become lists, tuples become
    lists, enums become their values, and paths become strings. Non-finite
    floats become ``None`` because ``NaN`` and ``Infinity`` are not valid JSON;
    the sidecar must parse in strict readers. An object that cannot be reduced
    raises with its location instead of being silently stringified.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value, where)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), where)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            # A zero-dimensional array is a scalar; enumerate() would raise.
            return _json_safe(value.item(), where)
        return [_json_safe(item, f"{where}[{index}]") for index, item in enumerate(value.tolist())]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    raise ValueError(
        f"metadata value at {where} has type {type(value).__name__}, which has no "
        "deterministic JSON representation"
    )


def _quantize(values: NDArray[np.float32], bit_depth: int) -> NDArray[Any]:
    """Clip display values to ``[0, 1]`` and round to integer codes."""
    maximum = float((1 << bit_depth) - 1)
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if bit_depth == 8:
        return cast(NDArray[Any], np.rint(clipped * maximum).astype(np.uint8))
    return cast(NDArray[Any], np.rint(clipped * maximum).astype(np.uint16))


def save_image(image: RenderedImage, path: str | Path) -> Path:
    """Write a rendered image and its sidecar.

    Parameters
    ----------
    image:
        A display-referred :class:`~camsim.core.frames.RenderedImage` whose
        values lie in ``[0, 1]``. ``linear_rgb`` and ``stages`` are not saved.
    path:
        Destination with a ``.png``, ``.tif``, or ``.tiff`` suffix. A path
        without a suffix receives ``.png``. Other suffixes are rejected so a
        lossy codec can never enter a measurement chain.

    Returns
    -------
    pathlib.Path
        The file actually written (with ``.png`` appended when needed). The
        sidecar is ``<stem>.json`` beside it.

    Raises
    ------
    ValueError
        If ``image`` is not a :class:`RenderedImage` or the suffix is not a
        supported lossless format.
    OSError
        If the parent directory cannot be created or a file cannot be written.
    """
    if not isinstance(image, RenderedImage):
        raise ValueError(f"image must be a RenderedImage; got {type(image).__name__}")

    target = Path(path)
    suffix = target.suffix.lower()
    if not suffix:
        target = target.with_suffix(".png")
        suffix = ".png"
    format_name = _FORMAT_BY_SUFFIX.get(suffix)
    if format_name is None:
        supported = ", ".join(sorted(_FORMAT_BY_SUFFIX))
        raise ValueError(
            f"unsupported image suffix {suffix!r}; save_image writes lossless "
            f"{supported} files only (lossy formats such as JPEG alter the pixels "
            "and invalidate every measurement taken from the file)"
        )

    bit_depth = 16 if format_name == "TIFF" and image.channels == "MONO" else 8
    values = np.asarray(image.data, dtype=np.float32)
    if image.channels == "MONO":
        array = _quantize(values[..., 0], bit_depth)
    else:
        array = _quantize(values, 8)
    pil_image = Image.fromarray(array)

    target.parent.mkdir(parents=True, exist_ok=True)

    def writer(handle: IO[bytes]) -> None:
        pil_image.save(handle, format=format_name)

    _atomic_write(target, writer)

    payload = {
        "kind": _KIND,
        "schema_version": _SCHEMA_VERSION,
        "format": format_name,
        "bit_depth": bit_depth,
        "color_space": image.color_space,
        "channels": image.channels,
        "file": target.name,
        "metadata": _json_safe(image.metadata),
    }
    _write_json(target.with_suffix(".json"), payload)
    return target


def _read_json(path: Path) -> Any:
    """Read a UTF-8 JSON sidecar, converting syntax errors to ``ValueError``."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON sidecar: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: sidecar is not valid UTF-8: {exc}") from exc


def _require_mapping(value: object, what: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a field-specific ``ValueError``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a JSON object; got {type(value).__name__}")
    return value


def _read_sidecar(path: Path, source: Path) -> Mapping[str, Any]:
    """Load an image sidecar and validate its identity fields."""
    mapping = _require_mapping(_read_json(path), f"image sidecar {path}")
    version = mapping.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported sidecar schema_version {version!r}; this build "
            f"reads version {_SCHEMA_VERSION!r}"
        )
    kind = mapping.get("kind")
    if kind != _KIND:
        raise ValueError(
            f"{path}: sidecar kind must be {_KIND!r}; got {kind!r} (a RAW-frame "
            "sidecar cannot describe a PNG/TIFF image)"
        )
    file_name = mapping.get("file")
    if file_name is not None:
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError(f"{path}: sidecar file must be a plain file name; got {file_name!r}")
        if file_name != source.name:
            raise ValueError(
                f"{path}: sidecar names file {file_name!r} but {source.name!r} was "
                "loaded; the image and sidecar were renamed or mixed up"
            )
    return mapping


def _infer_channels(array: NDArray[Any], source: Path) -> tuple[ChannelLayout, NDArray[Any]]:
    """Return the channel layout for an image array and add the mono axis."""
    if array.ndim == 2:
        return "MONO", array[..., np.newaxis]
    if array.ndim == 3 and array.shape[2] == 3:
        return "RGB", array
    raise ValueError(
        f"{source}: unsupported image array shape {array.shape}; expected (H, W) "
        "grayscale or (H, W, 3) RGB"
    )


def load_image(path: str | Path) -> RenderedImage:
    """Load a PNG or TIFF rendered image and its optional sidecar.

    Parameters
    ----------
    path:
        A ``.png``, ``.tif``, or ``.tiff`` file. The sidecar is
        ``<stem>.json``; when it is absent, the loader infers ``RGB`` versus
        ``MONO`` from the array shape and assumes the sRGB colour space with
        empty metadata.

    Returns
    -------
    RenderedImage
        Display-referred float32 data in ``[0, 1]``, scaled from the stored
        8- or 16-bit codes. ``linear_rgb`` and ``stages`` are ``None`` because
        neither is stored; metrics that need linear light decode the transfer
        function from ``data``.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the suffix is unsupported, the file is a lossy/unsupported format
        or Pillow mode, the sidecar's schema version/kind/format/bit depth/
        channels disagree with the file, or the array shape and dtype cannot
        be interpreted as display data.
    OSError
        If Pillow cannot decode the image.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in _FORMAT_BY_SUFFIX:
        supported = ", ".join(sorted(_FORMAT_BY_SUFFIX))
        raise ValueError(
            f"unsupported image suffix {suffix!r}; load_image reads lossless {supported} files only"
        )
    if not source.is_file():
        raise FileNotFoundError(f"image file not found: {source}")

    sidecar_path = source.with_suffix(".json")
    sidecar = _read_sidecar(sidecar_path, source) if sidecar_path.is_file() else None

    try:
        with Image.open(source) as pil_image:
            pil_image.load()
            image_format = pil_image.format
            mode = pil_image.mode
            array = np.asarray(pil_image)
    except (UnidentifiedImageError, OSError) as exc:
        raise OSError(f"{source}: cannot decode image: {exc}") from exc

    if image_format not in _ACCEPTED_FORMATS:
        raise ValueError(
            f"{source}: actual image format {image_format!r} is not PNG or TIFF; "
            "lossy formats are rejected because their compression alters the pixels"
        )
    if mode not in _READABLE_MODES:
        raise ValueError(
            f"{source}: unsupported image mode {mode!r}; expected 8-bit RGB, "
            "8-bit grayscale, or 16-bit grayscale"
        )
    if array.dtype == np.uint8:
        bit_depth = 8
    elif array.dtype.kind == "u" and array.dtype.itemsize == 2:
        # Pillow reports big-endian 16-bit TIFFs as mode I;16B/I;16L, whose
        # NumPy dtype is ">u2" or "<u2", not the native uint16 alias. Accept
        # any two-byte unsigned type and normalise it before scaling.
        bit_depth = 16
        array = np.ascontiguousarray(array.astype(np.uint16, copy=False))
    else:
        raise ValueError(
            f"{source}: unsupported pixel dtype {array.dtype}; expected uint8 or uint16"
        )

    channels, expanded = _infer_channels(array, source)
    if channels == "RGB" and bit_depth != 8:
        raise ValueError(
            f"{source}: a {bit_depth}-bit three-channel image is not supported; "
            "RGB files are read at 8 bits and grayscale files at 8 or 16 bits"
        )
    if expanded.shape[0] == 0 or expanded.shape[1] == 0:
        raise ValueError(f"{source}: image must contain at least one pixel")

    color_space: ColorSpace = "sRGB"
    metadata: dict[str, Any] = {}
    if sidecar is not None:
        declared_format = sidecar.get("format")
        if declared_format != image_format:
            raise ValueError(
                f"{sidecar_path}: sidecar declares format {declared_format!r} but the "
                f"file is {image_format!r}"
            )
        declared_depth = sidecar.get("bit_depth")
        if isinstance(declared_depth, bool) or declared_depth not in {8, 16}:
            raise ValueError(f"{sidecar_path}: bit_depth must be 8 or 16; got {declared_depth!r}")
        if declared_depth != bit_depth:
            raise ValueError(
                f"{sidecar_path}: sidecar declares {declared_depth}-bit data but the "
                f"file contains {bit_depth}-bit samples"
            )
        declared_channels = sidecar.get("channels")
        if declared_channels not in _CHANNEL_LAYOUTS:
            raise ValueError(
                f"{sidecar_path}: channels must be 'RGB' or 'MONO'; got {declared_channels!r}"
            )
        if declared_channels != channels:
            raise ValueError(
                f"{sidecar_path}: sidecar declares channels={declared_channels!r} but "
                f"the image array is {channels!r}"
            )
        declared_space = sidecar.get("color_space")
        if declared_space not in _COLOR_SPACES:
            raise ValueError(
                f"{sidecar_path}: color_space must be 'sRGB' or 'DisplayP3'; got {declared_space!r}"
            )
        color_space = cast(ColorSpace, declared_space)
        declared_metadata = sidecar.get("metadata", {})
        if declared_metadata is None:
            declared_metadata = {}
        metadata = dict(_require_mapping(declared_metadata, f"{sidecar_path}: metadata"))

    maximum = float((1 << bit_depth) - 1)
    display = np.ascontiguousarray(expanded.astype(np.float32) / maximum, dtype=np.float32)
    return RenderedImage(
        data=display,
        color_space=color_space,
        channels=channels,
        linear_rgb=None,
        stages=None,
        metadata=metadata,
    )

"""Persist RAW sensor frames as ``.npy`` arrays with a mandatory JSON sidecar.

A RAW frame is *not* a photograph. Its pixels are integer ADC codes that still
contain the black-level pedestal, and the same numbers mean different things
under a different colour-filter pattern, bit depth, conversion gain, exposure,
or temperature. A bare ``.npy`` file therefore preserves the array and loses
every fact the ISP needs. This module writes the array and a sidecar together:

* ``<name>.npy`` — the ``uint16`` codes, exactly as the sensor produced them,
  with no scaling, gamma, or clipping applied.
* ``<name>.json`` — the schema version, every scalar/string :class:`RawFrame`
  field, the data file name/shape/dtype, and (when present) the name of the
  mask archive. The sidecar is the authority on how to interpret the array.
* ``<name>_masks.npz`` — compressed ``defect_mask``, ``hot_mask``, and
  ``pdaf_mask`` boolean arrays. Masks share the frame's megapixel geometry, so
  embedding them as JSON booleans would inflate the sidecar by orders of
  magnitude. ``hot_mask`` is stored separately because the sensor applies its
  dark-current multiplier before DPC and a caller may want the hot subset even
  though ``defect_mask`` contains the union.

Typical RAW data span 10-16 bits per pixel; a 50 MP 10-bit frame is 100 MB of
``uint16``. Storing it losslessly is deliberate — a JPEG-compressed frame would
change the very noise statistics that the characterization and metrics layers
measure. File names are deterministic functions of the frame index and seed
(``raw_000123_seed7.npy``) so reruns overwrite the same artifacts instead of
accumulating timestamped variants.

Failure artefacts guarded here: a missing sidecar turns a self-describing frame
into an unexplained array; a sidecar/data mismatch after a rename pairs one
frame's metadata with another's pixels; a mask archive whose arrays have the
wrong shape would misalign every defective-pixel and PDAF correction; codes
above ``2**bit_depth - 1`` mean the data were widened without rescaling; and
loading the JSON as an image would apply a display transfer function to linear
code values. Each condition raises with the offending file and field named.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import IO, Any, Final, cast
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from camsim.core.frames import RawFrame

__all__ = ["load_raw", "save_raw"]

_SCHEMA_VERSION: Final[str] = "1.0"
"""Sidecar schema version. A loader rejects versions it does not understand."""

_KIND: Final[str] = "camsim.raw_frame"
"""Sidecar ``kind`` marker, so an image sidecar cannot be mistaken for a RAW one."""

_SCALAR_FIELDS: Final[tuple[str, ...]] = (
    "cfa",
    "bit_depth",
    "black_level_dn",
    "exposure_s",
    "analog_gain_db",
    "digital_gain",
    "mode",
    "temperature_c",
    "seed",
    "frame_index",
    "pixel_pitch_um",
    "sensor_config_hash",
)
"""Every :class:`RawFrame` field except the data array and the two masks."""

_MASK_FIELDS: Final[tuple[str, ...]] = ("defect_mask", "hot_mask", "pdaf_mask")
"""Optional boolean arrays stored in the compressed mask archive."""

_SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset({".npy", ".json"})
"""A loader accepts the data file or its sidecar, nothing else."""


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


def _json_scalar(value: object) -> Any:
    """Return an enum or NumPy scalar as a plain JSON value, unchanged otherwise."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_stem(raw: RawFrame) -> str:
    """Return the deterministic file stem for one frame.

    The index orders frames within a capture sequence and the seed identifies
    the random stream, so replaying the same frame overwrites the same files.
    """
    return f"raw_{raw.frame_index:06d}_seed{raw.seed}"


def _frame_payload(raw: RawFrame) -> dict[str, Any]:
    """Return the scalar/string metadata block recorded in the sidecar."""
    return {name: _json_scalar(getattr(raw, name)) for name in _SCALAR_FIELDS}


def save_raw(raw: RawFrame, directory: str | Path) -> Path:
    """Write a RAW frame and its sidecar into ``directory``.

    Parameters
    ----------
    raw:
        A :class:`~camsim.core.frames.RawFrame`. Its ``data`` are ADC codes;
        they are stored verbatim as ``uint16`` with no pedestal subtraction or
        renormalisation.
    directory:
        Destination directory. It is created when missing. Files are named
        ``raw_{frame_index:06d}_seed{seed}.npy`` and ``.json``, plus
        ``_masks.npz`` when either mask is present.

    Returns
    -------
    pathlib.Path
        The ``.npy`` data file. The sidecar shares its stem.

    Raises
    ------
    ValueError
        If ``raw`` is not a :class:`RawFrame`.
    OSError
        If the directory cannot be created or a file cannot be written.
    """
    if not isinstance(raw, RawFrame):
        raise ValueError(f"raw must be a RawFrame; got {type(raw).__name__}")
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _frame_stem(raw)
    data_path = out_dir / f"{stem}.npy"
    sidecar_path = out_dir / f"{stem}.json"
    mask_path = out_dir / f"{stem}_masks.npz"
    masks: dict[str, NDArray[np.bool_]] = {}
    for name in _MASK_FIELDS:
        mask = getattr(raw, name)
        if mask is not None:
            masks[name] = mask

    def write_data(handle: IO[bytes]) -> None:
        np.save(handle, raw.data, allow_pickle=False)

    _atomic_write(data_path, write_data)

    def write_masks(handle: IO[bytes]) -> None:
        np.savez_compressed(handle, **cast(dict[str, Any], masks))

    if masks:
        _atomic_write(mask_path, write_masks)

    payload: dict[str, Any] = {
        "kind": _KIND,
        "schema_version": _SCHEMA_VERSION,
        "data": {
            "file": data_path.name,
            "shape": [int(raw.data.shape[0]), int(raw.data.shape[1])],
            "dtype": "uint16",
        },
        "masks": None
        if not masks
        else {
            "file": mask_path.name,
            "arrays": [name for name in _MASK_FIELDS if name in masks],
        },
        "frame": _frame_payload(raw),
    }
    # The sidecar is written last: a reader never sees metadata pointing at an
    # array that has not landed yet.
    _write_json(sidecar_path, payload)
    return data_path


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


def _require_field(mapping: Mapping[str, Any], name: str, what: str) -> Any:
    """Return a required sidecar field or raise naming the missing key."""
    if name not in mapping:
        raise ValueError(f"{what} is missing required field {name!r}")
    return mapping[name]


def _require_file_name(value: object, what: str) -> str:
    """Return a plain file name that stays inside the sidecar's directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty file name; got {value!r}")
    if Path(value).name != value or "\\" in value or value in {".", ".."}:
        raise ValueError(
            f"{what} must be a plain file name in the sidecar directory; got {value!r}"
        )
    return value


def _require_shape(value: object, what: str) -> tuple[int, int]:
    """Return a positive ``(height, width)`` pair from the sidecar."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} must be a two-element [height, width] list; got {value!r}")
    dimensions: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{what} dimensions must be positive integers; got {value!r}")
        dimensions.append(int(item))
    return dimensions[0], dimensions[1]


def _load_npy(path: Path) -> NDArray[Any]:
    """Read a ``.npy`` array without allowing pickle deserialisation."""
    try:
        with open(path, "rb") as handle:
            array = np.load(handle, allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"{path}: not a valid NumPy .npy array: {exc}") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{path}: expected a NumPy array; got {type(array).__name__}")
    return array


def _load_masks(
    payload: object,
    sidecar_path: Path,
    shape: tuple[int, int],
) -> dict[str, NDArray[np.bool_]]:
    """Load and validate the referenced mask archive."""
    if payload is None:
        return {}
    mapping = _require_mapping(payload, f"{sidecar_path}: masks")
    filename = _require_file_name(mapping.get("file"), "masks.file")
    arrays = mapping.get("arrays")
    if not isinstance(arrays, list) or not arrays:
        raise ValueError(
            f"{sidecar_path}: masks.arrays must be a non-empty list of mask names; got {arrays!r}"
        )
    names: list[str] = []
    for item in arrays:
        if not isinstance(item, str) or item not in _MASK_FIELDS:
            raise ValueError(
                f"{sidecar_path}: unknown mask name {item!r}; expected one of {_MASK_FIELDS}"
            )
        if item in names:
            raise ValueError(f"{sidecar_path}: mask {item!r} is listed twice")
        names.append(item)

    archive_path = sidecar_path.parent / filename
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"RAW mask archive not found: {archive_path}; the sidecar references "
            f"{filename!r}. Re-save the frame with save_raw so the array, sidecar, "
            "and masks stay together."
        )
    loaded: dict[str, NDArray[np.bool_]] = {}
    try:
        archive = np.load(archive_path, allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"{archive_path}: cannot open the RAW mask archive: {exc}") from exc
    if isinstance(archive, np.ndarray):
        raise ValueError(
            f"{archive_path}: the RAW mask archive must be a compressed .npz container; "
            "the file is a plain .npy array"
        )
    try:
        with archive:
            available = set(archive.files)
            for name in names:
                if name not in available:
                    raise ValueError(
                        f"{archive_path}: mask {name!r} is referenced by the sidecar "
                        "but missing from the archive"
                    )
                mask = np.asarray(archive[name])
                if mask.dtype != np.bool_:
                    raise ValueError(
                        f"{archive_path}: mask {name!r} must have dtype bool; got {mask.dtype}"
                    )
                if mask.shape != shape:
                    raise ValueError(
                        f"{archive_path}: mask {name!r} shape {mask.shape} must match "
                        f"frame shape {shape}"
                    )
                loaded[name] = cast(NDArray[np.bool_], mask)
    except (ValueError, EOFError, BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(archive_path) in str(exc):
            raise
        raise ValueError(f"{archive_path}: invalid RAW mask archive: {exc}") from exc
    return loaded


def _load_raw_frame(path: Path, expected_data_name: str | None = None) -> RawFrame:
    """Read a validated sidecar and its arrays and rebuild a :class:`RawFrame`."""
    if not path.is_file():
        raise FileNotFoundError(f"RAW sidecar not found: {path}")
    mapping = _require_mapping(_read_json(path), f"RAW sidecar {path}")

    version = mapping.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported sidecar schema_version {version!r}; this build "
            f"reads version {_SCHEMA_VERSION!r}"
        )
    kind = mapping.get("kind")
    if kind != _KIND:
        raise ValueError(
            f"{path}: sidecar kind must be {_KIND!r}; got {kind!r} (an image sidecar "
            "cannot be loaded as a RAW frame)"
        )

    data_meta = _require_mapping(_require_field(mapping, "data", f"sidecar {path}"), "data")
    filename = _require_file_name(_require_field(data_meta, "file", "data"), "data.file")
    expected_shape = _require_shape(_require_field(data_meta, "shape", "data"), "data.shape")
    dtype = data_meta.get("dtype")
    if dtype != "uint16":
        raise ValueError(f"{path}: data.dtype must be 'uint16' (RAW ADC codes); got {dtype!r}")

    if expected_data_name is not None and filename != expected_data_name:
        raise ValueError(
            f"{path}: the sidecar names data file {filename!r}, which does not match "
            f"the loaded file {expected_data_name!r}; the pair was renamed or mixed up"
        )
    data_path = path.parent / filename
    if not data_path.is_file():
        raise FileNotFoundError(
            f"RAW data file not found: {data_path}; the sidecar references "
            f"{filename!r}. Re-save the frame with save_raw."
        )
    data = _load_npy(data_path)
    if data.dtype != np.uint16:
        raise ValueError(f"{data_path}: RAW data must have dtype uint16; got {data.dtype}")
    if data.ndim != 2:
        raise ValueError(f"{data_path}: RAW data must be 2-D (H, W); got shape {data.shape}")
    if data.shape != expected_shape:
        raise ValueError(
            f"{data_path}: RAW data shape {data.shape} does not match the sidecar "
            f"shape {expected_shape}"
        )

    frame_meta = _require_mapping(_require_field(mapping, "frame", f"sidecar {path}"), "frame")
    fields = {
        name: _require_field(frame_meta, name, f"sidecar {path} frame") for name in _SCALAR_FIELDS
    }
    masks = _load_masks(mapping.get("masks"), path, expected_shape)

    try:
        return RawFrame(
            data=cast(NDArray[np.uint16], data),
            cfa=fields["cfa"],
            bit_depth=fields["bit_depth"],
            black_level_dn=fields["black_level_dn"],
            exposure_s=fields["exposure_s"],
            analog_gain_db=fields["analog_gain_db"],
            digital_gain=fields["digital_gain"],
            mode=fields["mode"],
            temperature_c=fields["temperature_c"],
            seed=fields["seed"],
            frame_index=fields["frame_index"],
            pixel_pitch_um=fields["pixel_pitch_um"],
            defect_mask=masks.get("defect_mask"),
            hot_mask=masks.get("hot_mask"),
            pdaf_mask=masks.get("pdaf_mask"),
            sensor_config_hash=fields["sensor_config_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid RAW frame metadata: {exc}") from exc


def load_raw(path: str | Path) -> RawFrame:
    """Load a RAW frame from its ``.npy`` data file or ``.json`` sidecar.

    Parameters
    ----------
    path:
        Either the data file (its sibling ``<stem>.json`` is required) or the
        sidecar itself (whose ``data.file`` names the array). Other extensions
        are rejected because a ``.tiff`` or ``.png`` cannot carry RAW codes
        without a display transform.

    Returns
    -------
    RawFrame
        Reconstructed with every scalar field, the exact masks, and the
        original ``uint16`` codes. The frame is validated by the normal
        :class:`RawFrame` constructor, so a corrupted sidecar that names an
        impossible bit depth, black level, shape, or mask cannot load.

    Raises
    ------
    FileNotFoundError
        If the sidecar, data file, or a referenced mask archive is missing.
    ValueError
        If the suffix is unsupported, the JSON is malformed, the schema
        version or kind is unknown, the array dtype/shape disagrees with the
        sidecar, a mask is missing or mis-shaped, or any frame field is
        physically invalid.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"load_raw expects a .npy RAW data file or its .json sidecar; got {source}"
        )
    if suffix == ".npy":
        _validate_sidecar_pair(source, source.with_suffix(".json"))
        return _load_raw_frame(source.with_suffix(".json"), expected_data_name=source.name)
    return _load_raw_frame(source)


def _validate_sidecar_pair(data_path: Path, sidecar_path: Path) -> None:
    """Require the sidecar that belongs to a data file passed to :func:`load_raw`."""
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"RAW sidecar not found: {sidecar_path}; save_raw writes the .npy data "
            "and .json sidecar as a pair, and the sidecar is required to interpret "
            "the ADC codes"
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"RAW data file not found: {data_path}")

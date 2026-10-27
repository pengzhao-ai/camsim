"""The unified ISP pipeline: RAW sensor codes to a display-referred image.

The pipeline is the reconstruction chain that turns a sensor's quantized
readout into something a viewer, a metric, or a machine-vision model can use::

    RAW --DPC--> BLC --> LSC --> AWB --> demosaic --> CCM
        --> denoise --> sharpen --> tone --> display-referred output

Each arrow is a stage whose physics and numerics are documented in its own
module. This module owns only what no individual stage can own: the enforced
order, the applicability rules for each CFA layout, the optional trace, the
final linear/display buffer pair, and the provenance metadata attached to the
rendered image.

**Why the order matters.** The ADC records `DN = signal + black_level`, so the
pedestal must come out before any multiplicative stage. A gain applied to the
pedestal turns a constant offset into a colour cast that no later diagonal
balance can remove, and lens shading computed on uncorrected codes is biased
low. White balance must run before demosaicing because its statistics are
per-CFA-channel samples; interpolating first and then balancing mixes
reconstructed channels whose noise is spatially correlated. The colour
correction matrix is a linear map between the camera's spectral basis and a
colorimetric basis, so it belongs in the linear domain after demosaicing and
before denoise and sharpen. Metamerism is the reason a single matrix is only an
approximation: different scene spectra can produce the same three camera
responses yet different tristimulus values, so a CCM fitted under one
illuminant cannot correct every spectrum exactly. Fitting is a training
operation and is deliberately out of scope here -- the pipeline applies a
matrix supplied in the configuration.

**Units and normalisation.** BLC is also the boundary between ADC codes and
normalised linear light. After subtracting the pedestal, the pipeline divides
by the white span ``(2**bit_depth - 1) - black_level_dn``, so an ADC-saturated
pixel maps to exactly 1.0 and every later stage sees the ``[0, 1]`` linear
units its own documentation assumes. White-balance gains are ratios and are
unaffected by the common divisor; demosaic, CCM, denoise, sharpen, and tone all
become scale-appropriate. The division is not a clip: the sensor ADC has
already clipped at full scale, and ``linear_rgb`` is never clamped. Monochrome
and RCCC frames use the same divisor. Disabling BLC is a diagnostic bypass: the
pipeline then stays in raw code units, tone mapping saturates almost every
pixel, and the absolute denoise/sharpen thresholds no longer mean what their
documentation says.

**Trace contract.** Tracing is opt-in. With ``trace=True`` the returned
``RenderedImage.stages`` is a dictionary whose insertion order is the pipeline
order. It contains one ``StageOutput`` for every executed stage, plus a final
``"output"`` entry holding the rendered display array and the output contract
(``color_space``, ``channels``, ``bit_depth``). Skipped and inapplicable stages
are omitted, never recorded as identity snapshots: a stage that did not run
must not appear to have run. ``trace=False`` leaves ``stages`` as ``None`` so
large sweeps do not pay for a contact sheet. Stage snapshots are references to
the arrays the stages returned; every stage returns a fresh array, so a later
stage cannot mutate an earlier snapshot.

**Applicability.** A mono sensor has no colour filters: AWB, demosaicing, and
CCM are colour operations and are skipped, while DPC, BLC, LSC, denoise,
sharpen, and tone still apply. Mono stays two-dimensional through the whole
pipeline and gains its trailing channel axis only at the display boundary.
RCCC has one red and three clear sites; its AWB and CCM are deferred, so those
stages are skipped and the demosaic stage always uses the documented
``"rccc_baseline"`` reconstruction, whose ``(R, C, C)`` output is a
shape/linearity placeholder rather than a colour-accurate image. A colour frame
(Bayer or RCCC) with demosaicing disabled is rejected: every downstream colour
stage and the three-channel rendered image require an RGB image, so mono is the
only supported no-demosaic path.

**Tone and output consistency.** ``config.output.color_space`` must equal
``config.tone.color_space``; the tone stage defines the primaries of both the
display array and the linear buffer. ``output.bit_depth`` is container
metadata only: ``RenderedImage.data`` is always float32 in ``[0, 1]`` and is
never requantized here. ``linear_rgb`` is the pre-clamp, pre-OETF linear image
in the output primaries (exposure compensation applied, then the sRGB-to-P3
matrix for DisplayP3, no clamp), so highlights may exceed one and colour
metrics can linearize the display buffer back to it. Disabling tone is a
diagnostic linear mode: it requires an sRGB output, keeps the unclipped linear
values in ``linear_rgb``, and clips that buffer to ``[0, 1]`` only to satisfy
the display contract.

**Real-world parameter ranges.** Consumer sensors quantize to 8-16 bits; black
levels are typically 64 DN at 10 bits and 256 DN at 12 bits. White-balance
gains span roughly 0.5-4, colour matrices have diagonal entries near 0.5-1.5
and off-diagonals around -0.5-0.5, denoise radii 0.5-3 px, and sharpening
amounts 0.2-1.5. Tone exposure trims run to a few stops either way. The
pipeline is deterministic: no random state is read or written, and identical
frames and configurations produce bit-identical outputs within one environment.

**Failure artefacts guarded here.** A pedestal left in place (a white-balance
colour cast), a black level that leaves no positive white span (a division by
zero), a missing CCM matrix (silent desaturation or a runtime fit with no
reference), a colour frame without demosaicing (shape/colour inconsistency),
mismatched tone and output colour spaces (a mislabelled buffer), P3 output with
tone disabled (no defined primaries for the diagnostic buffer), reordered or
identity-recorded stages (a trace that misrepresents the processing), mutating
the caller's RAW data, accepting a non-RAW object as a frame, and attaching
provenance that does not match the configuration actually applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from camsim.config.io import load_isp
from camsim.config.schema import ISPConfig, ToneConfig
from camsim.core.frames import (
    CFAPattern,
    ChannelLayout,
    RawFrame,
    RenderedImage,
    StageOutput,
)
from camsim.isp.awb import AWBResult, apply_awb, estimate_awb
from camsim.isp.blc import subtract_black_level
from camsim.isp.ccm import apply_ccm
from camsim.isp.colorspace import linear_srgb_to_p3
from camsim.isp.demosaic import DemosaicMethod, demosaic
from camsim.isp.denoise import denoise
from camsim.isp.dpc import correct_defects
from camsim.isp.lsc import apply_lsc
from camsim.isp.sharpen import unsharp_mask
from camsim.isp.tone import apply_tone

__all__ = ["ISP"]

# The four Bayer placements. RCCC and MONO are colour-filter-array layouts too,
# but they are not Bayer: they bypass AWB and CCM for different documented
# reasons.
_BAYER_PATTERNS: Final[frozenset[CFAPattern]] = frozenset(
    {CFAPattern.RGGB, CFAPattern.BGGR, CFAPattern.GRBG, CFAPattern.GBRG}
)


class ISP:
    """The enforced RAW-to-rendered-image pipeline for one configuration.

    Parameters
    ----------
    config:
        A validated :class:`~camsim.config.schema.ISPConfig`. The stages run in
        the fixed order ``dpc, blc, lsc, awb, demosaic, ccm, denoise, sharpen,
        tone, output``; each stage's ``enabled`` toggle selects whether it
        runs, subject to the CFA applicability rules. The output and tone
        colour spaces must agree, and a disabled tone stage requires sRGB.

    Raises
    ------
    ValueError
        If ``config`` is not an :class:`ISPConfig`, the tone and output colour
        spaces differ, or tone is disabled with a non-sRGB output. These are
        configuration errors that would otherwise produce a mislabelled image.

    Notes
    -----
    The configuration is captured by reference and the provenance hash is
    recomputed from its current state on every ``process`` call, so a mutation
    before processing cannot leave a stale hash. Individual stage functions
    re-validate mutated config values they consume, so an accidental mutation
    fails loudly rather than running with an out-of-domain parameter.
    """

    def __init__(self, config: ISPConfig) -> None:
        _validate_config(config)
        self._config = config

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Build a pipeline from a JSON ISP preset on disk.

        The file is read and validated through the production loader
        (:func:`camsim.config.io.load_isp`), so a pipeline built from a preset
        follows exactly the same validation path as one built programmatically.
        Filesystem, JSON-syntax, and schema errors propagate unchanged.
        """
        return cls(load_isp(path))

    @property
    def config(self) -> ISPConfig:
        """The validated configuration captured at construction."""
        return self._config

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form of the current configuration.

        The hash identifies the parameters that produced a rendered image and
        is copied into ``RenderedImage.metadata["isp_config_hash"]``. It is
        computed on demand from the configuration's current state, so a
        mutation after construction cannot leave a stale provenance value.
        """
        return _config_hash(self._config)

    def process(self, raw: RawFrame, trace: bool = False) -> RenderedImage:
        """Run the configured ISP stages on one RAW frame.

        Parameters
        ----------
        raw:
            A self-describing :class:`~camsim.core.frames.RawFrame` whose
            ``data`` are ADC codes including the black-level pedestal. The
            frame is never mutated; every stage returns a new array.
        trace:
            When true, record a :class:`~camsim.core.frames.StageOutput` for
            every executed stage, in pipeline order, plus a final ``"output"``
            entry. When false (the default), ``RenderedImage.stages`` is
            ``None`` and no trace structures are allocated.

        Returns
        -------
        A :class:`~camsim.core.frames.RenderedImage` whose ``data`` is float32
        in ``[0, 1]`` with shape ``(H, W, 3)`` for Bayer/RCCC output and
        ``(H, W, 1)`` for mono. BLC subtracts the pedestal and normalises the
        result by the white span, so the linear stages run in ``[0, 1]`` units.
        ``linear_rgb`` is the matching pre-clamp, pre-OETF linear buffer
        (unclipped), and ``metadata`` carries the ISP name and hash, the
        sensor configuration hash, the colour space, channel layout, container
        bit depth, exposure, pixel pitch, and the white-balance gains and
        method (or ``None`` when AWB was skipped).

        Raises
        ------
        ValueError
            If ``raw`` is not a :class:`RawFrame`; if the frame's black level
            leaves no positive white span; if a Bayer frame has the CCM enabled
            without a supplied matrix; if a colour frame has demosaicing
            disabled; or if a mutated configuration violates the tone/output
            colour-space rules. Each condition would otherwise produce a
            silently wrong image or trace.
        """
        if not isinstance(raw, RawFrame):
            raise ValueError(f"raw must be a RawFrame; got {type(raw).__name__}")
        config = self._config
        # Re-check the cross-field rules: a config mutated after construction
        # must fail here rather than produce a mislabelled image.
        _validate_config(config)
        isp_config_hash = _config_hash(config)
        cfa = _coerce_cfa(raw.cfa)
        channels: ChannelLayout = "MONO" if cfa is CFAPattern.MONO else "RGB"
        stages: dict[str, StageOutput] | None = {} if trace else None

        image: NDArray[np.float32]
        if config.dpc.enabled:
            image = correct_defects(raw, config.dpc)
            _record_stage(
                stages,
                "dpc",
                image,
                f"Defective-pixel correction with the {config.dpc.method!r} method",
                {"method": config.dpc.method},
            )
        else:
            # The pedestal remains on purpose: black-level correction is a
            # separate stage, so a caller who disables only DPC sees the same
            # RAW codes as a float32 buffer.
            image = np.array(raw.data, dtype=np.float32, copy=True)

        # Diagnostic bypass: without BLC the pipeline stays in raw code units.
        # Tone mapping then saturates almost every pixel, and the absolute
        # denoise/sharpen thresholds, documented for [0, 1] linear light, no
        # longer have their usual meaning.
        if config.blc.enabled:
            image = subtract_black_level(image, raw.black_level_dn, bit_depth=raw.bit_depth)
            white_span_dn = float(2**raw.bit_depth - 1) - float(raw.black_level_dn)
            if white_span_dn <= 0.0:
                raise ValueError(
                    f"black_level_dn ({raw.black_level_dn:g}) leaves no positive white "
                    f"span for bit_depth={raw.bit_depth}; the frame has no usable signal "
                    "range to normalise against"
                )
            # ADC saturation maps to exactly 1.0. The division is not a clip:
            # the sensor ADC already clipped at full scale.
            image = np.ascontiguousarray(
                image.astype(np.float64) / white_span_dn,
                dtype=np.float32,
            )
            _record_stage(
                stages,
                "blc",
                image,
                f"Black-level pedestal of {raw.black_level_dn:g} DN subtracted, then "
                f"divided by the {white_span_dn:g} DN white span",
                {
                    "black_level_dn": float(raw.black_level_dn),
                    "white_span_dn": white_span_dn,
                    "bit_depth": raw.bit_depth,
                },
            )

        if config.lsc.enabled and config.lsc.mode != "none":
            image = apply_lsc(image, config.lsc, cfa=cfa)
            _record_stage(
                stages,
                "lsc",
                image,
                f"Lens-shading correction in {config.lsc.mode!r} mode",
                _lsc_metadata(config),
            )

        awb_result: AWBResult | None = None
        if config.awb.enabled and _is_bayer(cfa):
            awb_result = estimate_awb(image, cfa, config.awb)
            image = apply_awb(image, awb_result.gains, cfa=cfa)
            _record_stage(
                stages,
                "awb",
                image,
                f"{awb_result.method} white balance applied to the {cfa.value} mosaic",
                {
                    "gains": list(awb_result.gains),
                    "illuminant": list(awb_result.illuminant_estimate),
                    "method": awb_result.method,
                    "confidence": awb_result.confidence,
                },
            )

        if cfa is not CFAPattern.MONO:
            if not config.demosaic.enabled:
                raise ValueError(
                    f"demosaicing is required for a {cfa.value} colour frame; only a "
                    "MONO sensor can render with demosaic.enabled=False because the "
                    "downstream colour stages and the RGB output need three channels "
                    "per pixel"
                )
            method: DemosaicMethod = (
                "rccc_baseline" if cfa is CFAPattern.RCCC else config.demosaic.method
            )
            image = demosaic(image, cfa, method)
            _record_stage(
                stages,
                "demosaic",
                image,
                f"{method} demosaicing of the {cfa.value} mosaic",
                {"method": method, "cfa": cfa.value},
            )

        if config.ccm.enabled and _is_bayer(cfa):
            matrix = config.ccm.matrix
            if matrix is None:
                raise ValueError(
                    "colour correction is enabled for a Bayer frame but "
                    "config.ccm.matrix is None; the pipeline applies a precomputed "
                    "matrix and does not fit one at runtime. Fit the matrix offline "
                    "with camsim.isp.ccm.fit_ccm against reference patches and store "
                    "it in the configuration; the default preset supplies the "
                    "identity matrix."
                )
            image = apply_ccm(image, matrix)
            _record_stage(
                stages,
                "ccm",
                image,
                "3x3 colour-correction matrix applied in linear RGB",
                {
                    "matrix": [list(row) for row in matrix],
                    "ridge": config.ccm.ridge,
                    "row_sum": config.ccm.row_sum,
                },
            )

        if config.denoise.enabled:
            image = denoise(image, config.denoise)
            _record_stage(
                stages,
                "denoise",
                image,
                f"{config.denoise.method} denoise with luma "
                f"{config.denoise.luma_strength:g} and chroma "
                f"{config.denoise.chroma_strength:g}",
                {
                    "method": config.denoise.method,
                    "luma_strength": config.denoise.luma_strength,
                    "chroma_strength": config.denoise.chroma_strength,
                    "radius": config.denoise.radius,
                },
            )

        if config.sharpen.enabled:
            image = unsharp_mask(image, config.sharpen)
            _record_stage(
                stages,
                "sharpen",
                image,
                f"Thresholded unsharp mask with amount {config.sharpen.amount:g}",
                {
                    "amount": config.sharpen.amount,
                    "radius": config.sharpen.radius,
                    "threshold": config.sharpen.threshold,
                    "luma_only": config.sharpen.luma_only,
                },
            )

        display: NDArray[np.float32]
        linear: NDArray[np.float32]
        if config.tone.enabled:
            display = np.ascontiguousarray(
                np.clip(apply_tone(image, config.tone), 0.0, 1.0), dtype=np.float32
            )
            linear = _linear_output(image, config.tone)
            _record_stage(
                stages,
                "tone",
                display,
                _tone_description(config.tone),
                {
                    "color_space": config.tone.color_space,
                    "exposure_compensation_ev": config.tone.exposure_compensation_ev,
                    "gamma": config.tone.gamma,
                    "transfer": _transfer_description(config.tone),
                },
            )
        else:
            # Diagnostic linear mode: keep the unclipped values for metrics and
            # clip only the buffer that has to satisfy the display contract.
            linear = np.ascontiguousarray(image, dtype=np.float32)
            display = np.ascontiguousarray(
                np.clip(linear, np.float32(0.0), np.float32(1.0)), dtype=np.float32
            )

        rendered = _expand_channel_axis(display, channels)
        linear_output = _expand_channel_axis(linear, channels)
        _record_stage(
            stages,
            "output",
            rendered,
            f"Display-referred {config.output.color_space} {channels} image "
            f"({config.output.bit_depth}-bit container metadata)",
            {
                "color_space": config.output.color_space,
                "channels": channels,
                "bit_depth": config.output.bit_depth,
            },
        )

        return RenderedImage(
            data=rendered,
            color_space=config.output.color_space,
            channels=channels,
            linear_rgb=linear_output,
            stages=stages,
            metadata={
                "isp_name": config.name,
                "isp_config_hash": isp_config_hash,
                "sensor_config_hash": raw.sensor_config_hash,
                "color_space": config.output.color_space,
                "channels": channels,
                "bit_depth": config.output.bit_depth,
                "exposure_s": raw.exposure_s,
                "pixel_pitch_um": raw.pixel_pitch_um,
                "awb_gains": None if awb_result is None else list(awb_result.gains),
                "awb_method": None if awb_result is None else awb_result.method,
            },
        )


def _validate_config(config: object) -> None:
    """Enforce the pipeline-level configuration rules.

    The per-stage schemas validate their own fields. These rules span stages:
    the tone stage defines the output primaries, so the output declaration must
    agree, and a disabled tone stage cannot define a Display P3 buffer at all.
    """
    if not isinstance(config, ISPConfig):
        raise ValueError(f"config must be an ISPConfig; got {type(config).__name__}")
    if config.output.color_space != config.tone.color_space:
        raise ValueError(
            "config.output.color_space "
            f"({config.output.color_space!r}) must match config.tone.color_space "
            f"({config.tone.color_space!r}); the tone stage defines the primaries of "
            "the rendered image and its linear buffer"
        )
    if not config.tone.enabled and config.tone.color_space != "sRGB":
        raise ValueError(
            "tone mapping is disabled, which is a diagnostic linear-output mode; it "
            "requires color_space='sRGB' for both tone and output, got "
            f"{config.tone.color_space!r}"
        )


def _coerce_cfa(cfa: object) -> CFAPattern:
    """Return the frame's CFA as a :class:`CFAPattern`.

    ``RawFrame`` validates the field at construction, but frames are mutable;
    re-checking turns a later mutation into a clear error instead of a wrong
    applicability decision.
    """
    if isinstance(cfa, CFAPattern):
        return cfa
    if isinstance(cfa, str):
        try:
            return CFAPattern(cfa)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in CFAPattern)
            raise ValueError(f"cfa must be one of {allowed}; got {cfa!r}") from exc
    raise ValueError(f"cfa must be a CFAPattern; got {type(cfa).__name__}")


def _is_bayer(cfa: CFAPattern) -> bool:
    """Return ``True`` for the four Bayer layouts, the colour path with AWB/CCM."""
    return cfa in _BAYER_PATTERNS


def _record_stage(
    stages: dict[str, StageOutput] | None,
    name: str,
    data: Any,
    description: str,
    metadata: dict[str, Any],
) -> None:
    """Insert one stage snapshot when tracing is enabled.

    ``stages`` is ``None`` without tracing, so the call is a no-op and no
    ``StageOutput`` objects are allocated for large untraced sweeps.
    """
    if stages is not None:
        stages[name] = StageOutput(data=data, description=description, metadata=metadata)


def _expand_channel_axis(
    array: NDArray[np.float32], channels: ChannelLayout
) -> NDArray[np.float32]:
    """Add the trailing channel axis required by ``RenderedImage``.

    Mono stays two-dimensional through every stage; only the rendered-image
    boundary exposes ``(H, W, 1)`` so the channel layout is unambiguous. A
    three-channel image is returned as a fresh C-contiguous float32 array.
    """
    if channels == "MONO":
        return np.ascontiguousarray(array[..., np.newaxis], dtype=np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


def _linear_output(image: NDArray[np.float32], tone: ToneConfig) -> NDArray[np.float32]:
    """Return the pre-clamp, pre-OETF linear image in the output primaries.

    Exposure compensation is a linear-light gain, and the sRGB-to-P3 matrix is
    also linear, so both run before the tone curve and before any clamp.
    ``apply_tone`` has already validated the exposure gain, so the scale here
    cannot overflow. Mono input has no colour-space matrix, matching
    :func:`camsim.isp.tone.apply_tone`, which leaves a single channel alone.
    Highlights above one are preserved deliberately: ``linear_rgb`` is for
    metrics, not for display.
    """
    scale = 2.0**tone.exposure_compensation_ev
    working = image.astype(np.float64) * scale
    if image.ndim == 3 and tone.color_space == "DisplayP3":
        return linear_srgb_to_p3(working)
    return np.ascontiguousarray(working, dtype=np.float32)


def _transfer_description(tone: ToneConfig) -> str:
    """Name the output transfer function for a human-readable trace entry."""
    if tone.gamma is None:
        return "sRGB OETF"
    return f"power curve with gamma {tone.gamma:g}"


def _tone_description(tone: ToneConfig) -> str:
    """Describe the final exposure, primaries, and transfer in one line."""
    return (
        f"{tone.color_space} output at {tone.exposure_compensation_ev:+g} EV "
        f"with the {_transfer_description(tone)}"
    )


def _lsc_metadata(config: ISPConfig) -> dict[str, Any]:
    """Return the lens-shading parameters that determined the stage output.

    A full-resolution flat-field map is summarised by its shape rather than
    copied into the trace; the map itself is already part of the configuration
    hash and the preset that produced it.
    """
    lsc = config.lsc
    metadata: dict[str, Any] = {"mode": lsc.mode}
    if lsc.mode == "radial":
        metadata["coefficients"] = None if lsc.coefficients is None else list(lsc.coefficients)
        metadata["center"] = None if lsc.center is None else list(lsc.center)
    if lsc.map is not None:
        metadata["map_shape"] = list(np.asarray(lsc.map).shape)
    return metadata


def _config_hash(config: ISPConfig) -> str:
    """Return the SHA-256 of the configuration's canonical JSON form.

    Pydantic's JSON-mode dump converts tuples to lists and, through the
    ``Array2D`` serializer, an LSC gain map supplied as a NumPy array to nested
    lists, so the hash does not depend on how the caller stored the map. Sorted
    keys, compact separators, and strict (NaN-free) JSON make the encoding
    independent of field order and of Python's whitespace defaults.
    """
    payload = config.model_dump(mode="json")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    """Convert a NumPy value that JSON mode did not already normalise."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"configuration value of type {type(value).__name__} is not JSON serialisable")

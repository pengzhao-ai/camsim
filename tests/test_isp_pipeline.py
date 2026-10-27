"""Tests for the unified ISP pipeline, its opt-in trace, and the rendered image.

The pipeline is the integration point of Phase 3: it wires the stages built in
Tasks 18-23 into the fixed reconstruction chain

``DPC -> BLC -> LSC -> AWB -> demosaic -> CCM -> denoise -> sharpen -> tone``

and returns a :class:`~camsim.core.frames.RenderedImage`. The order is not an
implementation detail. The black pedestal must come out before any gain, lens
shading before the white-balance estimate, white balance before the mosaic is
interpolated, and the colour matrix while the data are still linear. BLC also
normalises the pedestal-free codes by the white span, so ADC saturation maps to
1.0 and every later stage works in the ``[0, 1]`` linear units its config
documents. Tone mapping is the only non-linear step.

These tests pin the contract rather than the numerics of individual stages
(which have their own suites): stage order and trace membership, opt-in
tracing, the equality of the pipeline with a manual chain of the same stage
functions, linearity before tone, mono and RCCC applicability, Bayer error
paths, display colour spaces, the unclipped linear buffer against the clipped
display buffer, metadata and configuration hashing, determinism, and the lazy
public export. Failure artefacts guarded here: a stage recorded as an identity
instead of being omitted, a trace in the wrong order, a missing pedestal
subtraction, a missing CCM matrix, a colour frame with demosaicing disabled,
mismatched tone/output colour spaces, mutation of the caller's RAW data, and
non-finite display values.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import camsim
from camsim.config.io import PRESETS_DIR, load_isp
from camsim.config.schema import ISPConfig, LSCConfig
from camsim.core.frames import CFAPattern, RawFrame, StageOutput
from camsim.isp.awb import apply_awb, estimate_awb
from camsim.isp.blc import subtract_black_level
from camsim.isp.ccm import apply_ccm
from camsim.isp.colorspace import linear_srgb_to_p3
from camsim.isp.demosaic import demosaic
from camsim.isp.dpc import correct_defects
from camsim.isp.pipeline import ISP
from camsim.isp.tone import apply_tone

DEFAULT_ISP_PATH = PRESETS_DIR / "isp" / "default.json"

_DEFAULT_STAGES = ["dpc", "blc", "awb", "demosaic", "ccm", "tone", "output"]


def _bright_frame(raw: RawFrame) -> RawFrame:
    """Return a copy whose codes sit at the ADC maximum (a white field)."""
    full_scale = np.full(raw.data.shape, 2**raw.bit_depth - 1, dtype=np.uint16)
    return raw.with_data(full_scale)


def _uniform_frame(raw: RawFrame, signal_fraction: float) -> RawFrame:
    """Return a uniform frame at ``signal_fraction`` of the white span."""
    white_span = float(2**raw.bit_depth - 1) - float(raw.black_level_dn)
    code = round(float(raw.black_level_dn) + signal_fraction * white_span)
    code = min(max(code, 0), 2**raw.bit_depth - 1)
    return raw.with_data(np.full(raw.data.shape, code, dtype=np.uint16))


def _linear_before_tone(raw: RawFrame, config: ISPConfig) -> np.ndarray:
    """Reproduce the default preset's linear chain with the stage functions.

    This is deliberately an independent transcription of the documented order,
    so the pipeline test fails if the pipeline reorders, skips, or re-applies a
    stage. It assumes the default preset: DPC, BLC, AWB, demosaic, and CCM are
    enabled, LSC/denoise/sharpen are disabled. BLC is the stage that normalises
    the pedestal-free codes by the white span, so the manual chain must divide
    there too.
    """
    image = correct_defects(raw, config.dpc)
    image = subtract_black_level(image, raw.black_level_dn, bit_depth=raw.bit_depth)
    white_span_dn = float(2**raw.bit_depth - 1) - float(raw.black_level_dn)
    image = np.ascontiguousarray(image.astype(np.float64) / white_span_dn, dtype=np.float32)
    awb = estimate_awb(image, raw.cfa, config.awb)
    image = apply_awb(image, awb.gains, cfa=raw.cfa)
    image = demosaic(image, raw.cfa, config.demosaic.method)
    matrix = config.ccm.matrix
    assert matrix is not None
    return apply_ccm(image, matrix)


# ---------------------------------------------------------------------------
# Brief tests (verbatim)
# ---------------------------------------------------------------------------


def test_pipeline_order_and_trace(sensor_config, isp_config, sensor, raw_frame):
    isp = camsim.ISP(isp_config)
    out = isp.process(raw_frame, trace=True)
    assert out.stages is not None
    assert list(out.stages)[:2] == ["dpc", "blc"]
    assert out.data.shape == (raw_frame.data.shape[0], raw_frame.data.shape[1], 3)
    assert 0.0 <= out.data.min() and out.data.max() <= 1.0  # noqa: SIM300


def test_linearity_before_tone(sensor_config, isp_config, sensor, raw_frame):
    # Scaling the raw signal scales the linear output before tone.
    isp = camsim.ISP(isp_config)
    a = isp.process(raw_frame, trace=True)
    scaled_data = np.clip(
        (raw_frame.data.astype(np.float32) - raw_frame.black_level_dn) * 2
        + raw_frame.black_level_dn,
        0,
        2**raw_frame.bit_depth - 1,
    ).astype(np.uint16)
    scaled = raw_frame.with_data(scaled_data)
    b = isp.process(scaled, trace=True)
    np.testing.assert_allclose(b.stages["ccm"].data, 2 * a.stages["ccm"].data, rtol=1e-3, atol=1e-4)


def test_mono_pipeline_bypasses_demosaic_and_awb(mono_sensor_config, isp_config, mono_raw_frame):
    isp = camsim.ISP(isp_config)
    out = isp.process(mono_raw_frame, trace=True)
    assert out.channels == "MONO"
    assert "demosaic" not in out.stages


# ---------------------------------------------------------------------------
# Stage selection and ordering
# ---------------------------------------------------------------------------


def test_stage_disable(isp_config, raw_frame):
    """Disabled and inapplicable stages are omitted, never recorded as identity."""
    config = isp_config.model_copy(deep=True)
    config.dpc.enabled = False
    config.awb.enabled = False
    config.tone.enabled = False
    # LSC is enabled but its mode is "none": still a no-op and therefore omitted.
    config.lsc.enabled = True
    out = camsim.ISP(config).process(raw_frame, trace=True)
    assert list(out.stages) == ["blc", "demosaic", "ccm", "output"]
    for skipped in ("dpc", "lsc", "awb", "denoise", "sharpen", "tone"):
        assert skipped not in out.stages
    assert out.data.min() >= 0.0 and out.data.max() <= 1.0


def test_disabling_dpc_starts_from_raw_codes_with_the_pedestal(isp_config, raw_frame):
    """Without DPC and BLC the pipeline works on raw codes, pedestal included."""
    config = isp_config.model_copy(deep=True)
    config.dpc.enabled = False
    config.blc.enabled = False
    out = camsim.ISP(config).process(raw_frame, trace=True)
    assert "dpc" not in out.stages
    assert "blc" not in out.stages
    raw_float = raw_frame.data.astype(np.float32)
    awb = estimate_awb(raw_float, raw_frame.cfa, config.awb)
    expected = apply_awb(raw_float, awb.gains, cfa=raw_frame.cfa)
    np.testing.assert_array_equal(out.stages["awb"].data, expected)


def test_final_output_matches_manual_chain(isp_config, raw_frame):
    """The default preset reproduces an explicit chain of the stage functions."""
    out = camsim.ISP(isp_config).process(raw_frame, trace=True)
    expected = _linear_before_tone(raw_frame, isp_config)
    np.testing.assert_array_equal(out.stages["ccm"].data, expected)
    np.testing.assert_array_equal(out.linear_rgb, expected)
    np.testing.assert_array_equal(out.data, apply_tone(expected, isp_config.tone))


def test_blc_stage_normalizes_to_the_white_span(isp_config, raw_frame):
    """The BLC trace data is pedestal-free and scaled so ADC full scale maps to one."""
    config = isp_config.model_copy(deep=True)
    config.dpc.enabled = False
    isp = camsim.ISP(config)
    out = isp.process(raw_frame, trace=True)
    black = float(raw_frame.black_level_dn)
    white_span_dn = float(2**raw_frame.bit_depth - 1) - black
    expected = np.maximum(raw_frame.data.astype(np.float64) - black, 0.0) / white_span_dn
    np.testing.assert_array_equal(out.stages["blc"].data, expected.astype(np.float32))
    assert out.stages["blc"].metadata["white_span_dn"] == white_span_dn

    saturated = isp.process(_bright_frame(raw_frame), trace=True)
    np.testing.assert_array_equal(
        saturated.stages["blc"].data,
        np.ones(raw_frame.data.shape, dtype=np.float32),
    )


def test_default_pipeline_renders_a_usable_low_light_image(isp_config, raw_frame):
    """Normalisation keeps the default profile finite, unclipped, and non-constant."""
    out = camsim.ISP(isp_config).process(raw_frame)
    height, width = raw_frame.data.shape
    assert out.data.shape == (height, width, 3)
    assert np.all(np.isfinite(out.data))
    assert float(out.data.max()) < 1.0
    assert float(out.data.min()) > 0.0
    assert 0.0 < float(out.data.mean()) < 1.0
    assert float(np.ptp(out.data)) > 0.0
    assert not np.all(out.data == 1.0)


def test_brighter_frame_has_higher_linear_signal_and_only_tone_clips(isp_config, raw_frame):
    """The linear chain is monotonic; only the tone stage clips the display."""
    config = isp_config.model_copy(deep=True)
    # A pinned blue gain lets a bright frame exceed one before tone while the
    # raw codes themselves still fit the ADC, isolating the tone stage's clip.
    config.awb.gains = (1.0, 1.0, 3.0)
    isp = camsim.ISP(config)
    dim = isp.process(_uniform_frame(raw_frame, 0.25), trace=True)
    bright = isp.process(_uniform_frame(raw_frame, 0.8), trace=True)
    assert bright.stages["ccm"].data.mean() > dim.stages["ccm"].data.mean()
    assert np.all(np.isfinite(bright.stages["ccm"].data))
    assert float(bright.stages["ccm"].data.max()) > 1.0
    assert float(bright.stages["ccm"].data.min()) >= 0.0
    assert float(bright.data.max()) == 1.0
    assert float(dim.data.max()) < 1.0
    assert bright.linear_rgb is not None
    assert float(bright.linear_rgb.max()) > 1.0


def test_rggb_pipeline(isp_config, raw_frame):
    """An explicit RGGB frame runs every colour stage in pipeline order."""
    frame = replace(raw_frame, cfa=CFAPattern.RGGB)
    out = camsim.ISP(isp_config).process(frame, trace=True)
    assert out.channels == "RGB"
    assert out.data.shape == (frame.data.shape[0], frame.data.shape[1], 3)
    assert list(out.stages) == _DEFAULT_STAGES
    assert isinstance(out.stages["awb"], StageOutput)
    assert out.stages["demosaic"].metadata["cfa"] == "RGGB"
    assert out.metadata["awb_method"] == "gray_world"


def test_rccc_baseline_runs(isp_config, raw_frame):
    """RCCC skips AWB/CCM and reconstructs with the documented baseline."""
    frame = replace(raw_frame, cfa=CFAPattern.RCCC)
    out = camsim.ISP(isp_config).process(frame, trace=True)
    assert out.channels == "RGB"
    assert "awb" not in out.stages
    assert "ccm" not in out.stages
    assert out.stages["demosaic"].metadata["method"] == "rccc_baseline"
    assert out.metadata["awb_gains"] is None
    assert out.metadata["awb_method"] is None
    assert out.data.shape == (frame.data.shape[0], frame.data.shape[1], 3)
    assert np.all(np.isfinite(out.data))


def test_trace_is_opt_in(isp_config, raw_frame):
    """Tracing is off by default and leaves ``RenderedImage.stages`` as None."""
    isp = camsim.ISP(isp_config)
    untraced = isp.process(raw_frame)
    assert untraced.stages is None
    traced = isp.process(raw_frame, trace=True)
    assert traced.stages is not None
    assert list(traced.stages) == _DEFAULT_STAGES


# ---------------------------------------------------------------------------
# Output buffers, colour spaces, and tone consistency
# ---------------------------------------------------------------------------


def test_mono_output_is_single_channel(isp_config, mono_raw_frame):
    """Mono stays two-dimensional internally and gains one channel at the boundary."""
    out = camsim.ISP(isp_config).process(mono_raw_frame, trace=True)
    height, width = mono_raw_frame.data.shape
    assert out.channels == "MONO"
    assert out.data.shape == (height, width, 1)
    assert out.linear_rgb is not None
    assert out.linear_rgb.shape == out.data.shape
    assert list(out.stages) == ["dpc", "blc", "tone", "output"]
    assert out.stages["tone"].data.shape == (height, width)
    assert out.stages["output"].metadata["channels"] == "MONO"


def test_mono_denoise_and_sharpen_use_the_two_dimensional_path(isp_config, mono_raw_frame):
    """Enhancement stages keep mono single-channel rather than expanding to RGB."""
    config = isp_config.model_copy(deep=True)
    config.denoise.enabled = True
    config.denoise.luma_strength = 0.5
    config.sharpen.enabled = True
    config.sharpen.amount = 0.5
    out = camsim.ISP(config).process(mono_raw_frame, trace=True)
    height, width = mono_raw_frame.data.shape
    assert list(out.stages) == ["dpc", "blc", "denoise", "sharpen", "tone", "output"]
    assert out.stages["denoise"].data.shape == (height, width)
    assert out.stages["sharpen"].data.shape == (height, width)
    assert out.data.shape == (height, width, 1)


def test_mono_runs_with_demosaic_disabled(isp_config, mono_raw_frame):
    """Mono is the supported no-demosaic path, so the toggle is irrelevant."""
    config = isp_config.model_copy(deep=True)
    config.demosaic.enabled = False
    out = camsim.ISP(config).process(mono_raw_frame, trace=True)
    assert "demosaic" not in out.stages
    assert out.channels == "MONO"


def test_display_p3_pipeline(isp_config, raw_frame):
    """P3 output carries the P3-linear buffer and the P3 tone stage."""
    config = isp_config.model_copy(deep=True)
    config.tone.color_space = "DisplayP3"
    config.output.color_space = "DisplayP3"
    out = camsim.ISP(config).process(raw_frame, trace=True)
    assert out.color_space == "DisplayP3"
    assert out.metadata["color_space"] == "DisplayP3"
    assert out.stages["tone"].metadata["color_space"] == "DisplayP3"
    linear_srgb = _linear_before_tone(raw_frame, config)
    np.testing.assert_array_equal(out.linear_rgb, linear_srgb_to_p3(linear_srgb))
    np.testing.assert_array_equal(out.data, apply_tone(linear_srgb, config.tone))


def test_linear_rgb_is_unclipped_while_display_is_clipped(isp_config, raw_frame):
    """A pinned gain overflows the linear buffer but not the display contract."""
    config = isp_config.model_copy(deep=True)
    config.awb.gains = (1.0, 1.0, 3.0)
    out = camsim.ISP(config).process(_bright_frame(raw_frame), trace=True)
    assert out.linear_rgb is not None
    assert float(out.linear_rgb.max()) > 1.0
    assert float(out.data.min()) >= 0.0
    assert float(out.data.max()) == 1.0


def test_tone_disabled_clips_display_and_keeps_linear_unclipped(isp_config, raw_frame):
    """Disabled tone is a diagnostic mode: clipped display, untouched linear data."""
    config = isp_config.model_copy(deep=True)
    config.tone.enabled = False
    config.awb.gains = (1.0, 1.0, 3.0)
    bright = _bright_frame(raw_frame)
    out = camsim.ISP(config).process(bright, trace=True)
    assert "tone" not in out.stages
    assert out.linear_rgb is not None
    assert float(out.linear_rgb.max()) > 1.0
    np.testing.assert_array_equal(out.linear_rgb, _linear_before_tone(bright, config))
    np.testing.assert_array_equal(out.data, np.clip(out.linear_rgb, 0.0, 1.0))


def test_no_nan_output(isp_config, raw_frame):
    """Every rendered and linear sample is finite and inside the display range."""
    out = camsim.ISP(isp_config).process(raw_frame, trace=True)
    assert np.all(np.isfinite(out.data))
    assert out.linear_rgb is not None
    assert np.all(np.isfinite(out.linear_rgb))
    assert float(out.data.min()) >= 0.0 and float(out.data.max()) <= 1.0


# ---------------------------------------------------------------------------
# Error paths and applicability
# ---------------------------------------------------------------------------


def test_ccm_matrix_is_required_for_colour_frames(isp_config, raw_frame, mono_raw_frame):
    """A colour frame needs a supplied matrix; mono skips the CCM entirely."""
    config = isp_config.model_copy(deep=True)
    config.ccm.matrix = None
    isp = camsim.ISP(config)
    with pytest.raises(ValueError, match="fit_ccm"):
        isp.process(raw_frame)
    mono_out = isp.process(mono_raw_frame, trace=True)
    assert "ccm" not in mono_out.stages


def test_output_and_tone_colour_spaces_must_match(isp_config):
    """The tone stage defines the primaries of the rendered image."""
    config = isp_config.model_copy(deep=True)
    config.output.color_space = "DisplayP3"
    with pytest.raises(ValueError, match="color_space"):
        camsim.ISP(config)


def test_tone_disabled_requires_srgb_output(isp_config):
    """Disabled tone cannot define a P3 buffer, so it is an sRGB diagnostic mode."""
    config = isp_config.model_copy(deep=True)
    config.tone.enabled = False
    config.tone.color_space = "DisplayP3"
    config.output.color_space = "DisplayP3"
    with pytest.raises(ValueError, match="sRGB"):
        camsim.ISP(config)


def test_disabled_demosaic_on_colour_frames_raises(isp_config, raw_frame):
    """Only mono supports the no-demosaic path; a colour frame needs RGB."""
    config = isp_config.model_copy(deep=True)
    config.demosaic.enabled = False
    with pytest.raises(ValueError, match="demosaic"):
        camsim.ISP(config).process(raw_frame)


def test_process_and_construction_validate_types(isp_config, raw_frame):
    """The constructor wants an ISPConfig and process wants a RawFrame."""
    with pytest.raises(ValueError, match="ISPConfig"):
        camsim.ISP(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="RawFrame"):
        camsim.ISP(isp_config).process(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Metadata, hashing, and determinism
# ---------------------------------------------------------------------------


def test_metadata_records_capture_and_configuration(isp_config, raw_frame):
    """Rendered metadata carries provenance, capture settings, and AWB results."""
    isp = camsim.ISP(isp_config)
    out = isp.process(raw_frame, trace=True)
    assert set(out.metadata) == {
        "isp_name",
        "isp_config_hash",
        "sensor_config_hash",
        "color_space",
        "channels",
        "bit_depth",
        "exposure_s",
        "pixel_pitch_um",
        "awb_gains",
        "awb_method",
        "ccm_matrix",
    }
    assert out.metadata["isp_name"] == "default"
    assert out.metadata["isp_config_hash"] == isp.config_hash
    assert out.metadata["sensor_config_hash"] == raw_frame.sensor_config_hash
    assert out.metadata["color_space"] == "sRGB"
    assert out.metadata["channels"] == "RGB"
    assert out.metadata["bit_depth"] == 8
    assert out.metadata["exposure_s"] == raw_frame.exposure_s
    assert out.metadata["pixel_pitch_um"] == raw_frame.pixel_pitch_um
    assert out.metadata["awb_gains"] is not None
    assert len(out.metadata["awb_gains"]) == 3
    assert out.metadata["awb_method"] == "gray_world"


def test_config_hash_is_stable_and_canonical():
    """The hash is stable across loads, changes with parameters, and ignores storage form."""
    first = camsim.ISP(load_isp(DEFAULT_ISP_PATH))
    second = camsim.ISP(load_isp(DEFAULT_ISP_PATH))
    assert first.config_hash == second.config_hash

    changed_config = load_isp(DEFAULT_ISP_PATH).model_copy(deep=True)
    changed_config.tone.exposure_compensation_ev = 0.5
    assert camsim.ISP(changed_config).config_hash != first.config_hash

    common = {"schema_version": "1.0", "name": "hash", "provenance": {"source": "test"}}
    as_array = ISPConfig(
        **common,
        lsc=LSCConfig(enabled=True, mode="flat_field", map=np.ones((4, 4))),
    )
    as_list = ISPConfig(
        **common,
        lsc=LSCConfig(enabled=True, mode="flat_field", map=np.ones((4, 4)).tolist()),
    )
    assert camsim.ISP(as_array).config_hash == camsim.ISP(as_list).config_hash


def test_config_hash_is_recomputed_at_process_time(isp_config, raw_frame):
    """A config mutated after construction cannot carry a stale provenance hash."""
    isp = camsim.ISP(isp_config)
    before = isp.process(raw_frame).metadata["isp_config_hash"]
    isp.config.tone.exposure_compensation_ev = 0.5
    after = isp.process(raw_frame).metadata["isp_config_hash"]
    assert after == isp.config_hash
    assert after != before


def test_stage_metadata_records_parameters(isp_config, raw_frame):
    """Every executed stage records the parameters that determined its output."""
    config = isp_config.model_copy(deep=True)
    config.lsc = LSCConfig(enabled=True, mode="radial", coefficients=[1.0, 0.1])
    config.denoise.enabled = True
    config.denoise.method = "gaussian"
    config.denoise.luma_strength = 0.2
    config.denoise.chroma_strength = 0.1
    config.denoise.radius = 1.0
    config.sharpen.enabled = True
    config.sharpen.amount = 0.5
    config.sharpen.radius = 1.0
    config.sharpen.threshold = 0.01
    config.sharpen.luma_only = True

    out = camsim.ISP(config).process(raw_frame, trace=True)
    assert list(out.stages) == [
        "dpc",
        "blc",
        "lsc",
        "awb",
        "demosaic",
        "ccm",
        "denoise",
        "sharpen",
        "tone",
        "output",
    ]
    assert out.stages["dpc"].metadata["method"] == config.dpc.method
    white_span_dn = float(2**raw_frame.bit_depth - 1) - float(raw_frame.black_level_dn)
    assert out.stages["blc"].metadata == {
        "black_level_dn": float(raw_frame.black_level_dn),
        "white_span_dn": white_span_dn,
        "bit_depth": raw_frame.bit_depth,
    }
    assert out.stages["lsc"].metadata["mode"] == "radial"

    awb_metadata = out.stages["awb"].metadata
    assert set(awb_metadata) == {"gains", "illuminant", "method", "confidence"}
    assert len(awb_metadata["gains"]) == 3
    assert len(awb_metadata["illuminant"]) == 3
    assert awb_metadata["method"] == "gray_world"
    assert 0.0 <= awb_metadata["confidence"] <= 1.0

    assert out.stages["demosaic"].metadata["method"] == config.demosaic.method
    assert out.stages["demosaic"].metadata["cfa"] == "RGGB"

    ccm_metadata = out.stages["ccm"].metadata
    assert ccm_metadata["matrix"] == config.ccm.matrix
    assert ccm_metadata["ridge"] == config.ccm.ridge
    assert ccm_metadata["row_sum"] is config.ccm.row_sum

    denoise_metadata = out.stages["denoise"].metadata
    assert denoise_metadata["method"] == "gaussian"
    assert denoise_metadata["luma_strength"] == 0.2
    assert denoise_metadata["chroma_strength"] == 0.1
    assert denoise_metadata["radius"] == 1.0

    sharpen_metadata = out.stages["sharpen"].metadata
    assert sharpen_metadata["amount"] == 0.5
    assert sharpen_metadata["radius"] == 1.0
    assert sharpen_metadata["threshold"] == 0.01
    assert sharpen_metadata["luma_only"] is True

    tone_metadata = out.stages["tone"].metadata
    assert tone_metadata["color_space"] == "sRGB"
    assert tone_metadata["exposure_compensation_ev"] == 0.0
    assert tone_metadata["gamma"] is None
    assert isinstance(tone_metadata["transfer"], str) and tone_metadata["transfer"]

    assert all(stage.description.strip() for stage in out.stages.values())


def test_output_stage_wraps_the_rendered_display(isp_config, raw_frame):
    """The final trace entry is the display array plus the output contract."""
    out = camsim.ISP(isp_config).process(raw_frame, trace=True)
    output = out.stages["output"]
    np.testing.assert_array_equal(output.data, out.data)
    assert output.metadata == {
        "color_space": out.color_space,
        "channels": out.channels,
        "bit_depth": 8,
    }


def test_isp_from_json(raw_frame):
    """``from_json`` loads the shipped preset through the production loader."""
    isp = camsim.ISP.from_json(DEFAULT_ISP_PATH)
    assert isp.config.name == "default"
    out = isp.process(raw_frame)
    direct = camsim.ISP(load_isp(DEFAULT_ISP_PATH)).process(raw_frame)
    np.testing.assert_array_equal(out.data, direct.data)
    assert out.metadata == direct.metadata


def test_isp_is_a_lazy_public_export():
    """``camsim.ISP`` resolves through PEP 562 without eager imports."""
    assert "ISP" in camsim.__all__
    assert "ISP" in dir(camsim)
    assert camsim._LAZY_EXPORTS["ISP"] == ("camsim.isp.pipeline", "ISP")
    assert camsim.ISP is ISP


def test_pipeline_is_deterministic_and_leaves_raw_untouched(isp_config, raw_frame):
    """Identical calls are bit-identical and never mutate the caller's RAW codes."""
    before = raw_frame.data.copy()
    isp = camsim.ISP(isp_config)
    first = isp.process(raw_frame, trace=True)
    second = isp.process(raw_frame, trace=True)
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.linear_rgb, second.linear_rgb)
    np.testing.assert_array_equal(first.stages["ccm"].data, second.stages["ccm"].data)
    np.testing.assert_array_equal(raw_frame.data, before)


def test_rendered_metadata_records_the_applied_ccm(isp_config, raw_frame):
    """The default preset records its identity baseline in the image metadata."""
    identity = np.eye(3).tolist()
    assert np.asarray(isp_config.ccm.matrix).tolist() == identity
    out = camsim.ISP(isp_config).process(raw_frame)
    assert out.metadata["ccm_matrix"] == identity

    fitted = [[1.04, -0.02, -0.02], [-0.03, 1.06, -0.03], [-0.02, -0.04, 1.06]]
    config = isp_config.model_copy(
        update={"ccm": isp_config.ccm.model_copy(update={"matrix": fitted})}
    )
    fitted_out = camsim.ISP(config).process(raw_frame)
    assert fitted_out.metadata["ccm_matrix"] == fitted

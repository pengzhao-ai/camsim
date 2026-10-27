"""Example 04: EMVA 1288-style characterization of the flagship sensor.

This walkthrough measures the shipped sensor the way EMVA Standard 1288
characterizes a physical part: flat fields at several irradiance levels give a
photon transfer curve, dark frames give dark-signal non-uniformity, and a
high-signal flat field gives photo-response non-uniformity. The same
measurement chain is exercised by the G2 gate.

The physics in the figures:

* **Photon transfer curve (PTC).** For a pixel with mean signal ``mu_DN`` and
  temporal variance ``sigma_DN**2`` in black-level-corrected digital numbers,
  Poisson shot noise plus input-referred read noise gives
  ``sigma_DN**2 = K * mu_DN + sigma_read_DN**2`` to first order. ``K`` is the
  system gain in DN per electron; the fit is performed against the exact
  quantized Poisson/Gaussian model because the LCG read noise is only about a
  quarter of an ADC code.
* **SNR versus irradiance.** ``SNR_dB = 20 log10(signal_e / noise_e)`` grows as
  ``sqrt(signal)`` in the shot-noise regime and flattens at the read-noise
  floor. The example plots both row-demeaned pixel noise and total frame noise
  including common-mode row noise.
* **Dark current versus temperature.** Thermally generated carriers follow
  ``D(T) = D_ref * 2**((T - T_ref) / T_double)``. The measured points are
  differential rates relative to 25 C so the constant read-noise rectification
  offset cancels; at the reference temperature the dark signal is far below one
  ADC code.
* **LCG versus HCG.** Dual conversion gain trades full-well capacity for lower
  input-referred read noise. The comparison figure reports both sides in
  electrons and decibels.
* **DSNU and PRNU histograms.** DSNU is the per-pixel dark offset; PRNU is the
  per-pixel relative responsivity deviation. The raw dark distribution is
  compressed by the sensor's zero-charge clamp and sub-LSB ADC, so the figure
  annotates both the raw spread and the input-referred DSNU recovered by the
  characterization.

The example is deterministic: every frame uses ``base_seed + frame_index`` and
the fixed-pattern maps come from the preset's defect seed. It writes
``g2_*.png`` figures under ``CAMSIM_FIG_DIR`` (default ``results/figures``) and
``g2_sensor_results.json`` under ``--report-dir`` (default
``results/reports``). The JSON records the measured values, the design
tolerances, and a pass/fail verdict for each check. ``--quick`` reduces the
number of frames and temperature points for a smoke test, and writes
``g2_sensor_results_quick.json`` so it can never clobber the committed gate
evidence; quick mode is not gate evidence.

This is a **simulated, EMVA 1288-style characterization**, not a certified
measurement of a physical sensor. The preset is a representative engineering
design point, not a claim about a commercial part.

Failure artefacts guarded here: a percent-valued PRNU mistaken for a fraction;
including saturated levels in the PTC; mixing LCG and HCG frames in one fit;
measuring read noise without removing row noise or ADC quantization; treating
the raw post-clamp dark spread as the configured DSNU; and reporting a gate as
passed when a measured value lies outside its tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray

import camsim
from camsim.characterization import EMVAReport, characterize_sensor
from camsim.config.io import load_sensor
from camsim.config.schema import SensorConfig
from camsim.core.frames import RawFrame
from camsim.core.spectral import SpectralIrradiance
from camsim.sensor.cfa import cfa_masks
from camsim.sensor.noise import dark_current_e_per_s
from camsim.viz import (
    figure_dir,
    plot_dark_current_vs_temperature,
    plot_fixed_pattern_histograms,
    plot_mode_comparison,
    plot_ptc,
    plot_sensor_spectral_response,
    plot_snr_vs_irradiance,
    save_figure,
    set_style,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SENSOR_PRESET: Final[Path] = (
    REPO_ROOT / "src" / "camsim" / "config" / "presets" / "sensors" / "flagship_phone_50mp.json"
)
DEFAULT_REPORT_DIR: Final[Path] = REPO_ROOT / "results" / "reports"
REPORT_NAME: Final[str] = "g2_sensor_results.json"
QUICK_REPORT_NAME: Final[str] = "g2_sensor_results_quick.json"
DARK_BASE_SEED: Final[int] = 21

# G2 LCG sweep from the task brief: from about 0.4 DN to well above the
# 6000 e- full well at a 10 ms exposure. The characterization appends
# geometrically lower levels until the PTC has a low-signal anchor.
LCG_LEVELS: Final[tuple[float, ...]] = (
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    4e-2,
    8e-2,
)
# HCG highlights begin to clip near 4.6e-4 W/m^2/nm and the mean reaches the
# 2000 e- full well near 5.7e-4 W/m^2/nm, so its sweep is shorter than LCG's.
# It starts deep enough in the noise floor to resolve the 0.7 e- read noise.
HCG_LEVELS: Final[tuple[float, ...]] = (
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    4e-4,
    1e-3,
)
EXPOSURE_S: Final[float] = 0.01
DARK_EXPOSURE_S: Final[float] = 0.1
FRAMES: Final[int] = 32
QUICK_FRAMES: Final[int] = 8
DARK_FRAMES: Final[int] = 16
QUICK_DARK_FRAMES: Final[int] = 4
DARK_TEMPERATURES_C: Final[tuple[float, ...]] = (25.0, 35.0, 45.0, 55.0, 65.0, 70.0)
QUICK_DARK_TEMPERATURES_C: Final[tuple[float, ...]] = (25.0, 45.0, 70.0)

# Flat-field geometry used by the characterization and the fixed-pattern
# measurements. A 64x64 Bayer frame gives 1024 red, 2048 green, and 1024 blue
# samples per level.
FRAME_SHAPE: Final[tuple[int, int]] = (64, 64)
WAVELENGTHS_NM: Final[NDArray[np.float64]] = np.arange(360.0, 831.0, 5.0)


def _flat_cube(level: float, shape: tuple[int, int] = FRAME_SHAPE) -> SpectralIrradiance:
    """Return a spatially flat spectral cube at ``level`` W/m^2/nm."""
    data = np.full((*shape, WAVELENGTHS_NM.size), level, dtype=np.float64)
    return SpectralIrradiance.from_array(
        data,
        WAVELENGTHS_NM,
        "W/m^2/nm",
        pixel_pitch_um=1.12,
    )


def _valid_mask(frame: RawFrame) -> NDArray[np.bool_]:
    """Return pixels that are neither defective nor PDAF sites."""
    valid = np.ones(frame.data.shape, dtype=np.bool_)
    if frame.defect_mask is not None:
        valid &= ~frame.defect_mask
    if frame.pdaf_mask is not None:
        valid &= ~frame.pdaf_mask
    return valid


def _signal_dn(frame: RawFrame) -> NDArray[np.float64]:
    """Return black-level-corrected DN as float64."""
    return np.asarray(frame.data, dtype=np.float64) - float(frame.black_level_dn)


def _characterize_modes(
    config: SensorConfig,
    frames: int,
) -> tuple[EMVAReport, EMVAReport]:
    """Characterize the LCG and HCG readout modes with fixed seeds."""
    lcg = characterize_sensor(
        camsim.Sensor(config, seed=5),
        irradiance_levels=list(LCG_LEVELS),
        frames_per_level=frames,
        mode="LCG",
        exposure_s=EXPOSURE_S,
        base_seed=9,
    )
    hcg = characterize_sensor(
        camsim.Sensor(config, seed=1),
        irradiance_levels=list(HCG_LEVELS),
        frames_per_level=frames,
        mode="HCG",
        exposure_s=EXPOSURE_S,
        base_seed=3,
    )
    return lcg, hcg


def measure_fixed_pattern(
    config: SensorConfig,
    report: EMVAReport,
    *,
    frames: int,
    base_seed: int,
    exposure_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure per-pixel DSNU and PRNU maps from dark and flat frames.

    The dark map is the per-pixel mean signal of averaged dark frames in
    electrons; it is the raw post-clamp distribution, not the input-referred
    DSNU that the characterization deconvolves, and it still contains the
    temporal noise of the frame average. The flat map is the per-pixel relative
    deviation of an averaged flat field at the level the report used for PRNU,
    normalised separately for each CFA channel: red, green, and blue filters
    have different mean responses, so normalising the mosaic as one population
    would report the colour pattern as a 25 % PRNU. Defective and PDAF pixels
    are excluded because their response is intentionally not representative.
    """
    level = float(report.provenance["prnu_measurement_level_w_per_m2_nm"])
    gain = report.system_gain_dn_per_e
    sensor = camsim.Sensor(config, seed=1)

    dark_stack: list[NDArray[np.float64]] = []
    valid: NDArray[np.bool_] | None = None
    for index in range(frames):
        frame = sensor.expose(_flat_cube(0.0), exposure_s, mode="LCG", seed=base_seed + index)
        if valid is None:
            valid = _valid_mask(frame)
        dark_stack.append(_signal_dn(frame))
    flat_stack: list[NDArray[np.float64]] = []
    for index in range(frames):
        frame = sensor.expose(
            _flat_cube(level), exposure_s, mode="LCG", seed=base_seed + 1000 + index
        )
        flat_stack.append(_signal_dn(frame))
    assert valid is not None
    dark_mean_dn = np.stack(dark_stack, axis=0).mean(axis=0)
    flat_mean_dn = np.stack(flat_stack, axis=0).mean(axis=0)
    dsnu_e = dark_mean_dn[valid] / gain
    prnu_parts: list[NDArray[np.float64]] = []
    for mask in cfa_masks(config.cfa, flat_mean_dn.shape).values():
        selected = mask & valid
        if not bool(selected.any()):
            continue
        channel_mean = float(flat_mean_dn[selected].mean())
        prnu_parts.append(flat_mean_dn[selected] / channel_mean - 1.0)
    prnu_fraction = np.concatenate(prnu_parts)
    return np.asarray(dsnu_e, dtype=np.float64), np.asarray(prnu_fraction, dtype=np.float64)


def measure_dark_current(
    config: SensorConfig,
    report: EMVAReport,
    *,
    temperatures_c: Sequence[float],
    frames: int,
    base_seed: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Measure differential dark-current rates against the 25 C reference.

    The sensor clamps charge at zero after read noise, so a zero-illumination
    frame has a positive mean code even with no dark current. Subtracting the
    mean at 25 C removes that constant rectification offset; the difference
    divided by exposure and the measured system gain is the dark-current
    increase relative to 25 C. The reference-temperature point itself is
    therefore zero by construction and is not plotted as a rate.

    The returned model rates are the same differential quantity,
    ``D(T) - D(25 C)``, not the absolute rate ``D(T)``. Comparing a differential
    measurement with an absolute model would bias every point by the 1.2 e-/s
    reference term (about 2 % at 65-70 C and much more at low temperature).
    """
    gain = report.system_gain_dn_per_e
    sensor = camsim.Sensor(config, seed=1)
    mean_e_by_temperature: dict[float, float] = {}
    valid: NDArray[np.bool_] | None = None
    for temperature_index, temperature in enumerate(temperatures_c):
        values: list[NDArray[np.float64]] = []
        for index in range(frames):
            frame = sensor.expose(
                _flat_cube(0.0),
                DARK_EXPOSURE_S,
                mode="LCG",
                temperature_c=float(temperature),
                seed=base_seed + 10_000 * (temperature_index + 1) + index,
            )
            if valid is None:
                valid = _valid_mask(frame)
            values.append(_signal_dn(frame))
        assert valid is not None
        mean_dn = float(np.stack(values, axis=0).mean(axis=0)[valid].mean())
        mean_e_by_temperature[float(temperature)] = mean_dn / gain
    reference_temperature = float(temperatures_c[0])
    reference = mean_e_by_temperature[reference_temperature]
    measured_temperatures = [float(temperature) for temperature in temperatures_c[1:]]
    measured_rates = [
        (mean_e_by_temperature[temperature] - reference) / DARK_EXPOSURE_S
        for temperature in measured_temperatures
    ]
    reference_rate = dark_current_e_per_s(
        float(config.modes["LCG"].dark_current_e_per_s),  # type: ignore[index]
        reference_temperature,
        config.temperature,
    )
    model_rates = [
        dark_current_e_per_s(
            float(config.modes["LCG"].dark_current_e_per_s),  # type: ignore[index]
            temperature,
            config.temperature,
        )
        - reference_rate
        for temperature in measured_temperatures
    ]
    return measured_temperatures, measured_rates, measured_temperatures, model_rates


def _check(
    name: str,
    measured: float,
    *,
    target: float | None = None,
    rel: float | None = None,
    low: float | None = None,
    high: float | None = None,
) -> dict[str, Any]:
    """Return one serialisable gate check."""
    passed = True
    tolerance: dict[str, float] = {}
    if target is not None:
        tolerance["target"] = target
        if rel is not None:
            tolerance["relative"] = rel
            passed = passed and math.isclose(measured, target, rel_tol=rel, abs_tol=0.0)
    if low is not None:
        tolerance["minimum"] = low
        passed = passed and measured > low
    if high is not None:
        tolerance["maximum"] = high
        passed = passed and measured < high
    return {"name": name, "measured": measured, "tolerance": tolerance, "pass": bool(passed)}


def _build_checks(
    lcg: EMVAReport,
    hcg: EMVAReport,
    measured_temperatures: Sequence[float],
    measured_rates: Sequence[float],
    model_rates: Sequence[float],
) -> list[dict[str, Any]]:
    """Build the G2 pass/fail table from the measured quantities."""
    checks = [
        _check("LCG read noise (e- rms)", lcg.read_noise_e, target=2.0, rel=0.05),
        _check("LCG DSNU (e-)", lcg.dsnu_e, target=0.5, rel=0.10),
        _check("LCG PRNU (%)", lcg.prnu_percent, target=0.5, rel=0.10),
        _check("LCG saturation capacity (e-)", lcg.saturation_capacity_e, target=6000.0, rel=0.10),
        _check("LCG linearity residual", lcg.linearity_residual, high=0.05),
        _check("LCG dynamic range (dB)", lcg.dynamic_range_db, low=30.0, high=90.0),
        _check(
            "LCG system gain relative error",
            abs(lcg.provenance["system_gain_relative_error"]),
            high=0.02,
        ),
        _check("HCG read noise (e- rms)", hcg.read_noise_e, target=0.7, rel=0.05),
        _check("HCG DSNU (e-)", hcg.dsnu_e, target=0.5, rel=0.10),
        _check("HCG PRNU (%)", hcg.prnu_percent, target=0.5, rel=0.10),
        _check("HCG saturation capacity (e-)", hcg.saturation_capacity_e, target=2000.0, rel=0.10),
        _check("HCG linearity residual", hcg.linearity_residual, high=0.05),
        _check("HCG dynamic range (dB)", hcg.dynamic_range_db, low=30.0, high=90.0),
        _check(
            "HCG system gain relative error",
            abs(hcg.provenance["system_gain_relative_error"]),
            high=0.02,
        ),
    ]
    # Dark-current measurements are differential relative to 25 C and are
    # compared with the differential model D(T) - D(25 C). Below about 60 C the
    # differential dark charge is well under one ADC code and the zero-charge
    # clamp plus quantisation compresses the mean, so those points are shown but
    # are not a pass/fail criterion. At 65-70 C the 32-frame, 64x64 dark average
    # resolves the differential rate to better than 5 % of the doubling model.
    for temperature, measured, model in zip(
        measured_temperatures, measured_rates, model_rates, strict=True
    ):
        if temperature >= 65.0 and model > 0.0:
            checks.append(
                _check(
                    f"dark current at {temperature:g} C vs doubling model (e-/s)",
                    measured,
                    target=model,
                    rel=0.05,
                )
            )
    return checks


def _write_json(
    path: Path,
    config: SensorConfig,
    lcg: EMVAReport,
    hcg: EMVAReport,
    checks: Sequence[dict[str, Any]],
    dark: tuple[list[float], list[float], list[float], list[float]],
    dsnu_e: NDArray[np.float64],
    prnu_fraction: NDArray[np.float64],
    *,
    quick: bool,
    dark_frames: int,
    dark_base_seed: int,
) -> dict[str, Any]:
    """Assemble and write the deterministic G2 results JSON."""
    measured_temperatures, measured_rates, model_temperatures, model_rates = dark
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "camsim G2 sensor validation",
        "gate_evidence": not quick,
        "characterization": (
            "simulated EMVA 1288-style characterization; not a certified EMVA 1288 measurement"
        ),
        "environment": {
            "python": platform.python_version(),
            "camsim": camsim.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.system(),
        },
        "preset": {
            "name": config.name,
            "path": SENSOR_PRESET.relative_to(REPO_ROOT).as_posix(),
            "sensor_config_hash": lcg.provenance["sensor_config_hash"],
        },
        "LCG": lcg.to_dict(),
        "HCG": hcg.to_dict(),
        "dark_current": {
            "method": (
                "differential dark-frame mean relative to 25 C, divided by the "
                "0.1 s exposure and the measured LCG system gain"
            ),
            "model": "D(T) - D(25 C) over the same temperature list",
            "exposure_s": DARK_EXPOSURE_S,
            "frames_per_level": dark_frames,
            "base_seed": dark_base_seed,
            "measured_temperatures_c": measured_temperatures,
            "measured_rates_e_per_s": measured_rates,
            "model_temperatures_c": model_temperatures,
            "model_rates_e_per_s": model_rates,
            "reference_temperature_c": 25.0,
            "doubling_k": float(config.temperature.dark_current_doubling_k),
        },
        "fixed_pattern": {
            "dsnu_raw_spatial_sigma_e": float(np.std(dsnu_e, ddof=1)),
            "dsnu_reported_input_referred_e": lcg.dsnu_e,
            "dsnu_configured_e": float(config.noise.dsnu_e),
            "prnu_raw_spatial_sigma_percent": 100.0 * float(np.std(prnu_fraction, ddof=1)),
            "prnu_reported_percent": lcg.prnu_percent,
            "prnu_configured_percent": 100.0 * float(config.noise.prnu_fraction),
            "pixels": int(dsnu_e.size),
        },
        "checks": list(checks),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the characterization, write the G2 figures and JSON, return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use fewer frames and temperature points for a smoke test (not gate evidence)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=(
            f"directory for {REPORT_NAME} (default: results/reports); "
            f"quick mode writes {QUICK_REPORT_NAME}"
        ),
    )
    args = parser.parse_args(argv)

    set_style()
    figures = figure_dir(REPO_ROOT / "results" / "figures")
    config = load_sensor(SENSOR_PRESET)
    frames = QUICK_FRAMES if args.quick else FRAMES
    dark_frames = QUICK_DARK_FRAMES if args.quick else DARK_FRAMES
    temperatures = QUICK_DARK_TEMPERATURES_C if args.quick else DARK_TEMPERATURES_C

    print("camsim example 04: EMVA 1288-style sensor characterization")
    print(f"figure directory: {figures}")
    print(f"frames per level: {frames}{' (quick mode)' if args.quick else ''}")

    lcg, hcg = _characterize_modes(config, frames)
    print(
        f"LCG: read noise {lcg.read_noise_e:.3f} e-, saturation "
        f"{lcg.saturation_capacity_e:.0f} e-, DR {lcg.dynamic_range_db:.1f} dB"
    )
    print(
        f"HCG: read noise {hcg.read_noise_e:.3f} e-, saturation "
        f"{hcg.saturation_capacity_e:.0f} e-, DR {hcg.dynamic_range_db:.1f} dB"
    )

    written: list[Path] = []
    figure, _axes = plot_sensor_spectral_response(
        config,
        title="Flagship preset spectral response (representative synthetic curves)",
    )
    written.append(save_figure(figure, figures / "g2_qe_cfa_ir_cut.png"))
    plt.close(figure)

    reports = {"LCG": lcg, "HCG": hcg}
    figure, _axes = plot_ptc(reports, title="Photon transfer curves (exact quantized fit)")
    written.append(save_figure(figure, figures / "g2_ptc.png"))
    plt.close(figure)

    figure, _axes = plot_snr_vs_irradiance(reports, title="SNR versus sensor-plane irradiance")
    written.append(save_figure(figure, figures / "g2_snr_vs_irradiance.png"))
    plt.close(figure)

    dark = measure_dark_current(
        config,
        lcg,
        temperatures_c=temperatures,
        frames=dark_frames,
        base_seed=DARK_BASE_SEED,
    )
    measured_temperatures, measured_rates, _model_temperatures, model_rates = dark
    figure, _axes = plot_dark_current_vs_temperature(
        config,
        measured={"LCG": (measured_temperatures, measured_rates)},
        differential_reference_c=25.0,
        note=(
            "measured: differential vs 25 C\n"
            "below ~60 C the clamp and sub-LSB\n"
            "ADC compress the differential mean"
        ),
        title="Dark current versus temperature (differential from 25 C)",
    )
    written.append(save_figure(figure, figures / "g2_dark_current_vs_temperature.png"))
    plt.close(figure)

    figure, _axes = plot_mode_comparison(
        reports, title="Dual conversion gain: read-noise versus full-well trade"
    )
    written.append(save_figure(figure, figures / "g2_lcg_hcg_comparison.png"))
    plt.close(figure)

    dsnu_e, prnu_fraction = measure_fixed_pattern(
        config,
        lcg,
        frames=frames,
        base_seed=31,
        exposure_s=EXPOSURE_S,
    )
    figure, _axes = plot_fixed_pattern_histograms(
        dsnu_e,
        prnu_fraction,
        dsnu_configured_e=float(config.noise.dsnu_e),
        prnu_configured_percent=100.0 * float(config.noise.prnu_fraction),
        dsnu_reported_e=lcg.dsnu_e,
        title="Fixed-pattern noise from measured dark and flat frames",
    )
    written.append(save_figure(figure, figures / "g2_dsnu_prnu_histograms.png"))
    plt.close(figure)

    checks = _build_checks(lcg, hcg, measured_temperatures, measured_rates, model_rates)
    report_name = QUICK_REPORT_NAME if args.quick else REPORT_NAME
    report_path = args.report_dir.expanduser() / report_name
    payload = _write_json(
        report_path,
        config,
        lcg,
        hcg,
        checks,
        dark,
        dsnu_e,
        prnu_fraction,
        quick=args.quick,
        dark_frames=dark_frames,
        dark_base_seed=DARK_BASE_SEED,
    )

    print("\nWritten figures:")
    for path in written:
        print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KiB)")
    print(f"\nResults JSON: {report_path}")

    failures = [check for check in checks if not bool(check["pass"])]
    print("\nG2 checks:")
    for check in checks:
        status = "PASS" if bool(check["pass"]) else "FAIL"
        print(f"  [{status}] {check['name']}: {check['measured']:.4g}")
    if args.quick:
        print("\nQuick mode: checks are informational and are not G2 gate evidence.")
        return 0
    if failures:
        print(f"\nG2 gate: FAIL ({len(failures)} check(s) outside tolerance)")
        return 1
    assert payload["gate_evidence"] is True
    print("\nG2 gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

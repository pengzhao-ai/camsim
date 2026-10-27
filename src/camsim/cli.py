"""Command-line interface for camsim.

The CLI is the reproducible boundary between the library and the filesystem.
Every command either validates an input or writes a self-describing artifact:

* ``camsim validate PATH`` loads a sensor or ISP preset and prints its errors
  and warnings. Exit status is part of the contract: ``0`` for a valid file,
  ``1`` for a file that was read but violates the schema, and ``2`` for input
  that could not be read at all.
* ``camsim schema {sensor,isp}`` prints the Pydantic JSON Schema, giving editors
  completion for unit-bearing fields such as ``pixel.pitch_um`` and
  ``adc.full_scale_uv``.
* ``camsim info`` reports the package version, runtime, and shipped presets.
* ``camsim render`` composes one scene render, one sensor exposure, and one ISP
  pass. The RAW frame and the rendered PNG therefore describe the same
  exposure; the frame is saved with its JSON sidecar and mask archive, and a
  ``layout.json`` carries the metrics chart regions, colour reference, pixel
  pitch, and focal length needed by ``camsim measure``.
* ``camsim characterize`` runs the EMVA 1288-style flat-field measurement and
  writes ``emva_<mode>.json``. The numbers are simulated measurements of the
  configured sensor, not a certified characterization.
* ``camsim measure`` reloads a PNG/TIFF image and a layout sidecar, runs the
  full metric harness, and writes the report JSON. A colorchecker image needs a
  colour reference; without one the harness raises rather than assuming D65.
* ``camsim figures`` reruns the committed example gallery (01-04) with
  ``MPLBACKEND=Agg``. It requires a source checkout because the examples are
  not shipped in the wheel.

Physical failure modes are kept distinct from configuration ones. A RAW file
without its sidecar cannot be interpreted (bit depth, CFA, black level, and
gains live there), a display PNG is 8-bit so measurements inherit 1/255
quantization, and a lossy image is rejected before it can contaminate a metric.
The new commands map input and validation failures to exit status ``2`` with a
message on stderr; a gallery example that exits non-zero maps to ``1`` so a
batch caller can tell a bad argument from a failed run. Handlers import the
simulation layers lazily, so ``camsim info`` stays cheap and does not import
Pillow or the simulation packages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from camsim._version import __version__
from camsim.config.io import (
    PRESETS_DIR,
    ValidationReport,
    available_presets,
    isp_schema,
    sensor_schema,
    validate_config,
)

if TYPE_CHECKING:  # pragma: no cover - type-checking only, no runtime imports
    from camsim.config.schema import LensConfig
    from camsim.metrics.chart import ChartLayout
    from camsim.scene.targets import SceneResult

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_UNREADABLE = 2

Handler = Callable[[argparse.Namespace], int]

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
"""Source-checkout root. The installed wheel has no ``examples/`` below it."""

DEFAULT_SENSOR_PRESET: Final[Path] = PRESETS_DIR / "sensors" / "flagship_phone_50mp.json"
DEFAULT_ISP_PRESET: Final[Path] = PRESETS_DIR / "isp" / "default.json"

_SCENE_NAMES: Final[tuple[str, ...]] = (
    "colorchecker",
    "slanted_edge",
    "flat_patch",
    "siemens_star",
)
_READOUT_MODES: Final[tuple[str, ...]] = ("auto", "LCG", "HCG")
_DEFAULT_FRAMES: Final[int] = 8
_QUICK_FRAMES: Final[int] = 4
_MINIMUM_FRAMES: Final[int] = 2

# The reference phone lens of the design (Section 7.1.1). The scene renderer
# and the metrics layout must describe the same lens, so the value is built
# once in the render handler and passed to both.
_LENS_F_NUMBER: Final[float] = 1.8
_LENS_FOCAL_LENGTH_MM: Final[float] = 6.4
_LENS_SENSOR_DIAGONAL_MM: Final[float] = 11.46
_LENS_TRANSMITTANCE: Final[float] = 0.9

_EXAMPLE_SCRIPTS: Final[tuple[str, ...]] = (
    "01_spectral_scene_to_raw.py",
    "02_isp_stage_gallery.py",
    "03_measure_iq.py",
    "04_emva_characterization.py",
)
"""The gallery is run in this order; only examples 02-04 take report options."""


def _cmd_info(_: argparse.Namespace) -> int:
    """Print package metadata and the presets shipped with this install."""
    payload = {
        "name": "camsim",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "presets": available_presets(),
    }
    print(json.dumps(payload, indent=2))
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate one JSON configuration and return the documented exit code."""
    path = Path(args.path)
    try:
        report = validate_config(path)
    except (OSError, UnicodeDecodeError) as exc:
        # A missing, unreadable, or non-UTF-8 file is an input problem, not a
        # schema problem; the distinct exit code lets batch callers branch.
        print(f"camsim validate: cannot read {path}: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    _print_report(path, report)
    return EXIT_OK if report.valid else EXIT_INVALID


def _print_report(path: Path, report: ValidationReport) -> None:
    """Print one line per finding, followed by the overall verdict."""
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}")
    if report.valid:
        suffix = f" ({len(report.warnings)} warning(s))" if report.warnings else ""
        print(f"{path}: valid{suffix}")
    else:
        print(f"{path}: invalid ({len(report.errors)} error(s))")


def _cmd_schema(args: argparse.Namespace) -> int:
    """Print the JSON Schema for the requested configuration kind."""
    schema = sensor_schema() if args.kind == "sensor" else isp_schema()
    print(json.dumps(schema, indent=2))
    return EXIT_OK


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def _parse_shape(value: str) -> tuple[int, int]:
    """Parse a ``HxW`` command-line shape into positive integers."""
    parts = value.lower().strip().split("x")
    if len(parts) != 2:
        raise ValueError(
            f"--shape must be HxW with positive integers (for example 128x128); got {value!r}"
        )
    try:
        height = int(parts[0].strip())
        width = int(parts[1].strip())
    except ValueError as exc:
        raise ValueError(
            f"--shape must be HxW with positive integers (for example 128x128); got {value!r}"
        ) from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"--shape dimensions must be positive; got {value!r}")
    return height, width


def _reference_lens() -> LensConfig:
    """Return the design's reference phone lens configuration."""
    from camsim.config.schema import LensConfig

    return LensConfig(
        f_number=_LENS_F_NUMBER,
        focal_length_mm=_LENS_FOCAL_LENGTH_MM,
        sensor_diagonal_mm=_LENS_SENSOR_DIAGONAL_MM,
        transmittance=_LENS_TRANSMITTANCE,
    )


def _chart_layout(
    scene: SceneResult,
    *,
    binning_factor: int,
    output_shape: tuple[int, int],
    pixel_pitch_um: float,
    focal_length_mm: float,
) -> ChartLayout:
    """Map a rendered scene's native-grid regions onto the output sensor grid.

    The scene renderer always paints on the preset's native pixel grid, but
    :meth:`camsim.sensor.Sensor.expose` may bin internally (for example 2x2).
    The ROI coordinates therefore describe the scene grid, not the binned frame.
    Each half-open region is divided by the integer binning factor, rounding
    its start and end down so a region that covers output pixel ``k`` still
    does after scaling. A region is kept at a minimum extent of one output
    pixel.

    Raises
    ------
    ValueError
        If a scaled region falls outside the output grid, or the scaled chart
        regions overlap because the chart no longer survives the binning. The
        caller maps this to exit status 2 with a message naming the factor.
    """
    from camsim.metrics.chart import ChartLayout, PatchROI

    if (
        isinstance(binning_factor, bool)
        or not isinstance(binning_factor, int)
        or binning_factor < 1
    ):
        raise ValueError(f"binning factor must be a positive integer; got {binning_factor!r}")
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"output shape must be positive; got {output_shape!r}")

    def convert(roi: Any) -> Any:
        x0 = roi.x // binning_factor
        y0 = roi.y // binning_factor
        x1 = (roi.x + roi.width) // binning_factor
        y1 = (roi.y + roi.height) // binning_factor
        scaled_width = max(1, x1 - x0)
        scaled_height = max(1, y1 - y0)
        if x0 < 0 or y0 < 0 or x0 + scaled_width > width or y0 + scaled_height > height:
            raise ValueError(
                f"chart region {roi.name!r} maps outside the {width}x{height} output "
                f"grid at binning factor {binning_factor}; render a larger --shape or "
                "use an unbinned sensor preset"
            )
        return PatchROI(
            name=roi.name,
            x=x0,
            y=y0,
            width=scaled_width,
            height=scaled_height,
        )

    patches = tuple(convert(patch) for patch in scene.layout.patches)
    edge_roi = None if scene.layout.edge_roi is None else convert(scene.layout.edge_roi)
    try:
        return ChartLayout(
            name=scene.layout.name,
            patches=patches,
            edge_roi=edge_roi,
            reference=scene.reference,
            pixel_pitch_um=pixel_pitch_um,
            focal_length_mm=focal_length_mm,
        )
    except ValueError as exc:
        raise ValueError(
            f"the {scene.layout.name!r} layout does not survive binning factor "
            f"{binning_factor} on a {width}x{height} grid: {exc}; render a larger "
            "--shape or use an unbinned sensor preset"
        ) from exc


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON through an atomic temp-file replacement."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    data = (text + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _cmd_render(args: argparse.Namespace) -> int:
    """Render one scene to RAW and a display image, with a metrics layout."""
    try:
        return _render_capture(args)
    except (OSError, ValueError) as exc:
        print(f"camsim render: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE


def _render_capture(args: argparse.Namespace) -> int:
    """Implement ``camsim render`` after the input-validation boundary.

    Notes
    -----
    The default 200 lux capture with the shipped ISP preset (identity colour-
    correction matrix, gray-world auto white balance) is a capture, IO, and
    provenance demonstration, not a calibrated colour rendering. A colour
    validation needs a fitted CCM and a reference matched to the illuminant.

    A binned sensor preset is supported: the scene is rendered on the native
    grid, the sensor bins to the output grid, and the chart layout is scaled by
    the same integer factor so ``camsim measure`` sees regions inside the saved
    RAW frame.
    """
    from camsim.camera import Camera
    from camsim.config.io import load_isp, load_sensor
    from camsim.io.images import save_image
    from camsim.io.raw import save_raw
    from camsim.isp.pipeline import ISP
    from camsim.scene.targets import render
    from camsim.sensor import Sensor

    shape = _parse_shape(args.shape)
    try:
        sensor_config = load_sensor(args.sensor_preset)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load sensor preset {Path(args.sensor_preset)}: {exc}") from exc
    try:
        isp_config = load_isp(args.isp_preset)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load ISP preset {Path(args.isp_preset)}: {exc}") from exc
    pixel_pitch_um = float(sensor_config.pixel.pitch_um)
    lens = _reference_lens()

    scene = render(
        args.scene,
        illuminant="D65",
        lux=args.lux,
        lens=lens,
        pixel_pitch_um=pixel_pitch_um,
        shape=shape,
        oversampling=args.oversampling,
        seed=args.seed,
    )
    sensor = Sensor(sensor_config, seed=args.seed)
    isp = ISP(isp_config)

    auto_exposure = None
    if args.auto_exposure:
        # Auto-exposure probes the real sensor; the capture below replays the
        # measured settings so the saved RAW is the frame that was metered.
        auto_exposure = Camera(lens, sensor, isp).auto_expose(scene)
        exposure_s = auto_exposure.exposure_s
        analog_gain_db: float | None = auto_exposure.analog_gain_db
        digital_gain = auto_exposure.digital_gain
        mode = auto_exposure.mode
    else:
        exposure_s = (
            float(sensor_config.exposure.default_s)
            if args.exposure is None
            else float(args.exposure)
        )
        analog_gain_db = args.analog_gain_db
        digital_gain = 1.0
        mode = args.mode

    raw = sensor.expose(
        scene.irradiance,
        exposure_s,
        mode=mode,
        analog_gain_db=analog_gain_db,
        digital_gain=digital_gain,
        seed=args.seed,
    )
    image = isp.process(raw)

    out_dir = Path(args.out)
    raw_path = save_raw(raw, out_dir)
    # The rendered image gets a distinct stem so its sidecar cannot overwrite
    # the RAW sidecar that shares the frame's name.
    image_path = save_image(image, out_dir / f"{raw_path.stem}_render.png")
    # The scene layout lives on the native grid; binning divides the total
    # pixel count per axis, so the pitch ratio is the exact integer factor.
    sensor_pitch_um = float(sensor.config.pixel.pitch_um)
    pitch_ratio = raw.pixel_pitch_um / sensor_pitch_um
    binning_factor = round(pitch_ratio)
    if binning_factor < 1 or not math.isclose(pitch_ratio, binning_factor, rel_tol=1e-6):
        raise ValueError(
            f"RAW pixel pitch {raw.pixel_pitch_um:g} um is not an integer multiple of "
            f"the sensor grid pitch {sensor_pitch_um:g} um; cannot map the chart layout"
        )
    layout = _chart_layout(
        scene,
        binning_factor=binning_factor,
        output_shape=(int(raw.data.shape[0]), int(raw.data.shape[1])),
        pixel_pitch_um=raw.pixel_pitch_um,
        focal_length_mm=float(lens.focal_length_mm),
    )
    layout_path = out_dir / "layout.json"
    _write_json_file(layout_path, layout.to_dict())

    print(
        f"camsim render: {args.scene} at {args.lux:g} lux, "
        f"{shape[0]}x{shape[1]} px, oversampling {args.oversampling}"
    )
    if auto_exposure is not None:
        verdict = "converged" if auto_exposure.converged else "not converged"
        print(
            f"auto-exposure: {verdict} after {auto_exposure.iterations} probe(s), "
            f"metered {auto_exposure.metered_value:.3f} of full scale"
        )
    print(
        f"capture: {raw.exposure_s:.6g} s, {raw.analog_gain_db:g} dB analog gain, "
        f"{raw.digital_gain:g}x digital gain, mode {raw.mode}"
    )
    print(
        f"raw:     {raw_path} ({raw.bit_depth}-bit {raw.cfa.value}, "
        f"black level {raw.black_level_dn:g} DN)"
    )
    print(f"image:   {image_path} ({image.color_space} {image.channels})")
    edge = " + edge ROI" if layout.edge_roi is not None else ""
    print(f"layout:  {layout_path} ({len(layout.patches)} patch(es){edge})")
    return EXIT_OK


# ---------------------------------------------------------------------------
# characterize
# ---------------------------------------------------------------------------


def _cmd_characterize(args: argparse.Namespace) -> int:
    """Measure one readout mode and write an EMVA-style JSON report."""
    try:
        return _characterize(args)
    except (OSError, ValueError) as exc:
        print(f"camsim characterize: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE


def _characterize(args: argparse.Namespace) -> int:
    """Implement ``camsim characterize`` after the input-validation boundary."""
    from camsim.characterization import characterize_sensor
    from camsim.config.io import load_sensor
    from camsim.sensor import Sensor

    frames = _QUICK_FRAMES if args.quick else int(args.frames)
    if frames < _MINIMUM_FRAMES:
        raise ValueError(f"--frames must be at least {_MINIMUM_FRAMES}; got {frames}")

    try:
        config = load_sensor(args.preset)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load sensor preset {Path(args.preset)}: {exc}") from exc
    sensor = Sensor(config, seed=args.seed)
    report = characterize_sensor(
        sensor,
        frames_per_level=frames,
        mode=args.mode,
        exposure_s=float(args.exposure),
        base_seed=args.seed,
    )

    resolved = str(report.provenance.get("mode_resolved", args.mode))
    configured = config.modes["LCG"] if resolved == "LCG" else config.modes["HCG"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"emva_{resolved}.json"
    report.to_json(report_path)

    configured_noise = float(configured.read_noise_e_rms)
    relative_noise = (
        (report.read_noise_e - configured_noise) / configured_noise
        if configured_noise > 0.0
        else 0.0
    )
    read_noise_ok = abs(relative_noise) <= 0.25
    linearity_ok = report.linearity_residual <= 0.05

    print(
        f"camsim characterize: {config.name} ({resolved}), "
        f"{frames} frame(s) per level, exposure {float(args.exposure):g} s"
    )
    print(f"  system gain:   {report.system_gain_dn_per_e:.4f} DN/e-")
    print(f"  read noise:    {report.read_noise_e:.4f} e- rms (configured {configured_noise:g} e-)")
    print(
        f"  saturation:    {report.saturation_capacity_e:.1f} e- "
        f"(configured full well {float(configured.full_well_e):.0f} e-)"
    )
    print(f"  dynamic range: {report.dynamic_range_db:.2f} dB")
    print(f"  DSNU / PRNU:   {report.dsnu_e:.4f} e- / {report.prnu_percent:.4f} %")
    print(
        f"  linearity:     {report.linearity_residual * 100.0:.3f} % "
        "maximum deviation from a straight line"
    )
    print("sanity checks (not the G2 gate):")
    verdict = "PASS" if read_noise_ok else "OUTSIDE"
    print(f"  read noise within 25% of configured: {verdict} ({relative_noise * 100.0:+.1f}%)")
    print(f"  linearity residual below 5%:         {'PASS' if linearity_ok else 'OUTSIDE'}")
    print("limits: simulated EMVA 1288-style characterization, not a certified measurement;")
    print("        one conversion-gain mode per report; flat-field assumptions apply.")
    if args.quick:
        print(f"        quick mode uses {frames} frames/level and is not gate evidence.")
    print(f"report: {report_path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def _format_float(value: float, spec: str) -> str:
    """Format a finite float or return ``n/a`` for a NaN metric."""
    return format(value, spec) if math.isfinite(value) else "n/a"


def _cmd_measure(args: argparse.Namespace) -> int:
    """Measure one rendered image against a layout and write a metrics JSON."""
    try:
        return _measure(args)
    except (KeyError, OSError, ValueError) as exc:
        print(f"camsim measure: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE


def _measure(args: argparse.Namespace) -> int:
    """Implement ``camsim measure`` after the input-validation boundary."""
    from camsim.core.color import ColorReference
    from camsim.io.images import load_image
    from camsim.metrics.chart import ChartLayout
    from camsim.metrics.summary import measure_all

    image_path = Path(args.image)
    layout_path = Path(args.layout)
    image = load_image(image_path)
    layout = ChartLayout.from_dict(
        cast(Mapping[str, Any], json.loads(layout_path.read_text(encoding="utf-8")))
    )

    reference = None
    if args.reference is not None:
        reference = ColorReference.from_dict(
            cast(Mapping[str, Any], json.loads(Path(args.reference).read_text(encoding="utf-8")))
        )

    report = measure_all(image, layout, reference)

    out_path = (
        Path(args.out)
        if args.out is not None
        else image_path.with_name(f"{image_path.stem}_metrics.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json() + "\n", encoding="utf-8")

    edge = " + edge ROI" if layout.edge_roi is not None else ""
    print(f"camsim measure: {image_path}")
    print(f"layout: {layout.name} ({len(layout.patches)} patch(es){edge})")
    if report.mtf is not None:
        print(
            f"MTF50: {_format_float(report.mtf.mtf50_cp, '.3f')} cycles/pixel, "
            f"MTF20: {_format_float(report.mtf.mtf20_cp, '.3f')} cycles/pixel"
        )
    else:
        print("MTF50: n/a (layout has no edge ROI)")
    if report.acutance is not None:
        print(f"acutance: {report.acutance:.4f}")
    if report.snr is not None:
        print(f"SNR: measured {len(report.snr)} patch(es)")
    if report.color is not None:
        print(
            "colour: mean deltaE76 "
            f"{_format_float(report.color.delta_e76, '.3f')}, "
            "mean deltaE2000 "
            f"{_format_float(report.color.delta_e2000, '.3f')}"
        )
    else:
        print("colour: n/a")
    print(f"report: {out_path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _cmd_figures(args: argparse.Namespace) -> int:
    """Regenerate the committed gallery by running the example scripts."""
    examples_dir = REPOSITORY_ROOT / "examples"
    missing = [name for name in _EXAMPLE_SCRIPTS if not (examples_dir / name).is_file()]
    if missing:
        print(
            "camsim figures: the example gallery scripts are not available under "
            f"{REPOSITORY_ROOT}. This command requires a source checkout; the "
            "installed wheel ships no examples/ directory.",
            file=sys.stderr,
        )
        return EXIT_INVALID

    figures_dir = Path(args.out).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CAMSIM_FIG_DIR"] = str(figures_dir)
    environment["MPLBACKEND"] = "Agg"

    print(f"camsim figures: {'quick' if args.quick else 'full'} gallery")
    print(f"figures: {figures_dir}")
    print(f"reports: {report_dir}")
    for index, name in enumerate(_EXAMPLE_SCRIPTS, start=1):
        command = [sys.executable, str(examples_dir / name)]
        if name != _EXAMPLE_SCRIPTS[0]:
            # Example 01 has no report options; 02-04 write quick/full reports.
            command += ["--report-dir", str(report_dir)]
            if args.quick:
                command += ["--quick"]
        print(f"[{index}/{len(_EXAMPLE_SCRIPTS)}] {name} ... ", end="", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            print("FAILED")
            print(f"camsim figures: cannot run {name}: {exc}", file=sys.stderr)
            return EXIT_INVALID
        if completed.returncode != 0:
            print("FAILED")
            print(
                f"camsim figures: {name} exited with status {completed.returncode}",
                file=sys.stderr,
            )
            if completed.stderr.strip():
                print(completed.stderr.rstrip(), file=sys.stderr)
            return EXIT_INVALID
        print("ok")

    print(f"camsim figures: gallery written to {figures_dir}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="camsim",
        description="Deterministic camera sensor and ISP simulation with IQ metrics.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a sensor or ISP JSON configuration."
    )
    validate.add_argument("path", help="Path to the JSON configuration file.")
    validate.set_defaults(handler=_cmd_validate)

    schema = subparsers.add_parser("schema", help="Print a JSON Schema for a configuration type.")
    schema.add_argument("kind", choices=("sensor", "isp"), help="Configuration kind to export.")
    schema.set_defaults(handler=_cmd_schema)

    info = subparsers.add_parser("info", help="Print package and runtime information.")
    info.set_defaults(handler=_cmd_info)

    render_parser = subparsers.add_parser(
        "render", help="Render a target to RAW and a display-referred image."
    )
    render_parser.add_argument(
        "--scene",
        choices=_SCENE_NAMES,
        default="colorchecker",
        help="Target chart (default: colorchecker).",
    )
    render_parser.add_argument(
        "--lux",
        type=float,
        default=200.0,
        help=(
            "Scene illuminance in lux (default: 200). The default capture is a "
            "capture/IO/provenance demonstration; the preset's identity CCM and "
            "gray-world AWB are not colour-calibrated."
        ),
    )
    render_parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/captures"),
        help="Output directory (default: results/captures).",
    )
    render_parser.add_argument(
        "--shape",
        default="128x128",
        help="Sensor-grid HxW in pixels (default: 128x128).",
    )
    render_parser.add_argument(
        "--oversampling",
        type=int,
        default=2,
        help="Render samples per sensor pixel per axis (default: 2).",
    )
    render_parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Exposure time in seconds (default: the sensor preset default).",
    )
    render_parser.add_argument(
        "--analog-gain-db",
        type=float,
        default=None,
        help="Analog gain in dB (default: the sensor preset default).",
    )
    render_parser.add_argument(
        "--mode",
        choices=_READOUT_MODES,
        default="auto",
        help="Readout mode (default: auto).",
    )
    render_parser.add_argument(
        "--sensor-preset",
        type=Path,
        default=DEFAULT_SENSOR_PRESET,
        help="Sensor JSON preset (default: the shipped flagship preset).",
    )
    render_parser.add_argument(
        "--isp-preset",
        type=Path,
        default=DEFAULT_ISP_PRESET,
        help="ISP JSON preset (default: the shipped default preset).",
    )
    render_parser.add_argument("--seed", type=int, default=0, help="Capture seed (default: 0).")
    render_parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help=(
            "Choose exposure, analog gain, digital gain, and mode with "
            "Camera.auto_expose before capture (overrides --exposure, "
            "--analog-gain-db, and --mode)."
        ),
    )
    render_parser.set_defaults(handler=_cmd_render)

    characterize = subparsers.add_parser(
        "characterize", help="Run an EMVA 1288-style sensor characterization."
    )
    characterize.add_argument(
        "--preset",
        type=Path,
        default=DEFAULT_SENSOR_PRESET,
        help="Sensor JSON preset (default: the shipped flagship preset).",
    )
    characterize.add_argument(
        "--mode",
        choices=_READOUT_MODES,
        default="LCG",
        help=(
            "Readout mode to measure (default: LCG); 'auto' resolves to one mode "
            "and writes emva_<resolved-mode>.json."
        ),
    )
    characterize.add_argument(
        "--frames",
        type=int,
        default=_DEFAULT_FRAMES,
        help=f"Frames per level, at least {_MINIMUM_FRAMES} (default: {_DEFAULT_FRAMES}).",
    )
    characterize.add_argument(
        "--exposure",
        type=float,
        default=1e-3,
        help="Integration time in seconds (default: 1e-3).",
    )
    characterize.add_argument(
        "--out",
        type=Path,
        default=Path("results/reports"),
        help="Report directory (default: results/reports).",
    )
    characterize.add_argument(
        "--quick",
        action="store_true",
        help=f"Use {_QUICK_FRAMES} frames per level for a smoke test (not gate evidence).",
    )
    characterize.add_argument("--seed", type=int, default=0, help="Base frame seed (default: 0).")
    characterize.set_defaults(handler=_cmd_characterize)

    measure = subparsers.add_parser(
        "measure", help="Measure a rendered image against a chart layout."
    )
    measure.add_argument("--image", required=True, help="Rendered PNG or TIFF image.")
    measure.add_argument("--layout", required=True, help="ChartLayout JSON sidecar.")
    measure.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional ColorReference JSON overriding the layout's reference.",
    )
    measure.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report path (default: <image_stem>_metrics.json beside the image).",
    )
    measure.set_defaults(handler=_cmd_measure)

    figures = subparsers.add_parser(
        "figures", help="Regenerate the example gallery under results/figures."
    )
    figures.add_argument(
        "--quick",
        action="store_true",
        help="Run examples 02-04 in quick mode (not gate evidence).",
    )
    figures.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="Figure directory (default: results/figures).",
    )
    figures.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/reports"),
        help="Report directory for examples 02-04 (default: results/reports).",
    )
    figures.set_defaults(handler=_cmd_figures)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``camsim`` command line and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())

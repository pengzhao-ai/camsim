# API reference

This page documents the implemented public modules. Source code is not
embedded; every module's opening docstring states its physical/mathematical
basis, parameter ranges, and the failure artefacts it guards against.

!!! note "Scope"
    `camsim.camera` and `camsim.io` are scaffold placeholders in this version
    and have no public API yet. `camsim.tasks` is gated behind the G5 human
    sign-off and is not documented or advertised as implemented. A module that
    appears here is importable, tested, and used by the validation gates.

## Top-level package

`camsim` exposes its public types lazily (PEP 562), so importing the package
loads only version metadata. The five public names below are registered in
`_LAZY_EXPORTS`.

::: camsim
    options:
      members:
        - ISP
        - Sensor
        - SpectralIrradiance
        - WavelengthGrid
        - LensConfig

## Core contracts

Shared units, spectral cubes, RAW/rendered frames, colour references, and
physical constants. This is the only layer the metrics harness may import from
the simulator.

::: camsim.core

::: camsim.core.units

::: camsim.core.constants

::: camsim.core.spectral
    options:
      members:
        - WavelengthGrid
        - SpectralIrradiance

::: camsim.core.frames
    options:
      members:
        - CFAPattern
        - StageOutput
        - RawFrame
        - RenderedImage

::: camsim.core.color
    options:
      members:
        - ColorReference

## Configuration

Pydantic v2 schemas with units in every field name, cross-field physical
validators, provenance on every preset, and JSON I/O.

::: camsim.config

::: camsim.config.schema
    options:
      members:
        - LensConfig
        - SensorConfig
        - ISPConfig
        - Provenance

::: camsim.config.io
    options:
      members:
        - load_sensor
        - load_isp
        - validate_config
        - ValidationReport
        - sensor_schema
        - isp_schema
        - available_presets

## Scene

Synthetic spectral targets, chart layouts, illuminants, and spectral data.
`camsim.scene.charts` is the sanctioned bridge to the metrics layer.

::: camsim.scene

::: camsim.scene.targets
    options:
      members:
        - SceneResult
        - SceneLayout
        - render
        - colorchecker_target
        - flat_patch_target
        - slanted_edge_target
        - siemens_star_target
        - estimate_edge_angle

::: camsim.scene.charts
    options:
      members:
        - colorchecker_layout
        - flat_patch_layout
        - slanted_edge_layout

::: camsim.scene.illuminants
    options:
      members:
        - illuminant
        - lux_to_spectral_irradiance
        - photopic_luminous_efficiency
        - reference_lab

::: camsim.scene.spectral_data
    options:
      members:
        - SpectralPowerDistribution
        - ColorCheckerData
        - load_colorchecker

::: camsim.scene.glyphs

## Optics

Lens geometry, point-spread functions, propagation, and vignetting.

::: camsim.optics

::: camsim.optics.lens
    options:
      members:
        - LensConfig
        - transmittance_at

::: camsim.optics.psf
    options:
      members:
        - gaussian_psf
        - box_psf
        - airy_psf
        - psf_mtf
        - psf_mtf50

::: camsim.optics.propagation

::: camsim.optics.vignetting

## Sensor

The sensor turns sensor-plane spectral irradiance into a self-describing
`RawFrame`. The package re-exports its public helpers.

::: camsim.sensor

## ISP

The fixed RAW-to-rendered-image pipeline and its individual linear-space
stages. See [ISP stages](isp_stages.md) for the order, invariants, and trace
contract.

::: camsim.isp

::: camsim.isp.pipeline
    options:
      members:
        - ISP

::: camsim.isp.ccm
    options:
      members:
        - fit_ccm

::: camsim.isp.lsc
    options:
      members:
        - apply_lsc
        - calibrate_lsc

::: camsim.isp.demosaic
    options:
      members:
        - demosaic

## Characterization

EMVA 1288-style sensor measurements and the machine-readable measurement
functions behind the G2–G4 reports.

::: camsim.characterization

## Metrics

The independent image-quality harness: chart layouts, e-SFR, SNR, colour
difference, acutance, and the unified report. See [Metrics](metrics.md) for
the definitions and limitations.

::: camsim.metrics

## Visualisation

Colourblind-friendly plotting helpers used by the examples and notebooks. The
presentation layer is excluded from the unit-suite coverage gate; it is
validated by executing the gallery examples and notebooks.

::: camsim.viz

## Command line

The `camsim` console entry point.

::: camsim.cli
    options:
      members:
        - build_parser
        - main

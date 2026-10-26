# Datasheet to JSON mapping

Sensor datasheets describe a part in prose, plots, and unit-bearing tables;
camsim consumes a validated JSON preset. This page defines how each datasheet
parameter becomes a schema field, whether the value is measured, derived, or
assumed, and where the provenance record lives.

**Status: Phase 0 stub.** The conventions and draft table below are fixed
enough to guide Task 6 (spectral data) and the sensor work in Phase 2. The
complete mapping, including the flagship 50 MP preset's exact source URLs and
worked derivations, is finalized in Phase 2 (Task 17) alongside the G2
validation report.

## Conventions

- **Units are in the field name.** `pitch_um` is micrometres, `full_well_e` is
  electrons, `conversion_gain_uv_per_e` is microvolts per electron,
  `read_noise_e_rms` is electrons RMS, `dark_current_e_per_s` is electrons per
  second, `full_scale_uv` is microvolts, and time is seconds. A bare number is
  never accepted for a physical quantity.
- **Every preset carries `provenance`.** `source`, `url`, `retrieved`, and
  `notes` identify the document; `assumptions` lists every value that is an
  engineering estimate rather than a vendor measurement. The preset is
  representative of a class of parts, not a claim about one commercial part.
- **Derived values are computed, not copied.** Field of view and
  pixels-per-degree come from `LensConfig` properties; voltage at full well is
  `full_well_e * conversion_gain_uv_per_e * analog_gain`; dark current at a
  non-reference temperature follows the doubling model. Storing a derived
  value would let it disagree with its inputs.
- **Cross-field checks are part of the mapping.** The schema rejects negative
  or non-finite values, an ADC black level outside the code range, an exposure
  default outside its limits, and HCG full well larger than LCG full well. It
  reports ADC-limited saturation as a warning rather than an error, because
  many sensors intentionally clip at the ADC in high-gain modes.
- **Spectral curves are file references.** `qe_csv`, `cfa_csv`, and
  `ir_cut_csv` name CSV files on a 360–830 nm, 5 nm grid; the files and their
  citations are added in Phase 1 and recorded in `THIRD_PARTY.md`. Digitizing a
  datasheet plot is an assumption and is recorded as such.

## Draft mapping

Values in the right-hand column are from the representative
`flagship_phone_50mp` preset and are illustrative, not a specific part's
datasheet.

| Datasheet parameter | Schema path | Unit | Source type | Preset example |
|---|---|---|---|---|
| Pixel pitch | `pixel.pitch_um` | µm | Datasheet | 1.12 |
| Photodiode fill factor | `pixel.fill_factor` | – | Derived / assumption | 0.85 |
| Microlens collection gain | `pixel.microlens_gain` | – | Assumption | 1.15 |
| Colour filter array | `cfa` | – | Datasheet | `RGGB` |
| Spectral sampling grid | `spectral.wavelength_nm.start/stop/step` | nm | Model choice | 360 / 830 / 5 |
| Quantum efficiency curve | `spectral.qe_csv` | – | Datasheet plot, digitized | `data/spectra/qe_flagship.csv` |
| CFA transmission curves | `spectral.cfa_csv` | – | Datasheet / reference | `data/spectra/cfa_rggb.csv` |
| IR-cut transmission | `spectral.ir_cut_csv` | – | Datasheet plot, digitized | `data/spectra/ir_cut.csv` |
| Spectral crosstalk matrix | `spectral.spectral_crosstalk` | – | Measured / assumption | Identity |
| Low-conversion-gain conversion gain | `modes.LCG.conversion_gain_uv_per_e` | µV/e⁻ | Datasheet or derived | 100.0 |
| LCG full well | `modes.LCG.full_well_e` | e⁻ | Datasheet | 6000 |
| LCG read noise | `modes.LCG.read_noise_e_rms` | e⁻ RMS | Datasheet | 2.0 |
| LCG dark current at 25 °C | `modes.LCG.dark_current_e_per_s` | e⁻/s | Datasheet, normalized | 1.2 |
| High-conversion-gain conversion gain | `modes.HCG.conversion_gain_uv_per_e` | µV/e⁻ | Datasheet | 300.0 |
| HCG full well | `modes.HCG.full_well_e` | e⁻ | Datasheet | 2000 |
| HCG read noise | `modes.HCG.read_noise_e_rms` | e⁻ RMS | Datasheet | 0.7 |
| HCG dark current at 25 °C | `modes.HCG.dark_current_e_per_s` | e⁻/s | Datasheet, normalized | 1.2 |
| Analog gain range and step | `analog_gain.min_db/max_db/step_db/default_db` | dB | Datasheet | 0 / 24 / 1 / 0 |
| ADC resolution | `adc.bits` | bit | Datasheet | 10 |
| ADC full-scale input | `adc.full_scale_uv` | µV | Datasheet | 800000.0 |
| ADC black level | `adc.black_level_dn` | DN | Datasheet | 64 |
| Digital gain | `adc.digital_gain` | – | Datasheet / model | 1.0 |
| Exposure limits and default | `exposure.min_s/max_s/default_s` | s | Datasheet / derived | 1e-4 / 0.1 / 0.01 |
| Reference and operating temperature | `temperature.reference_c/default_c` | °C | Datasheet | 25.0 / 25.0 |
| Dark-current doubling temperature | `temperature.dark_current_doubling_k` | K | Datasheet / literature | 7.0 |
| Dead/hot/stuck pixel rates | `defects.dead_rate/hot_rate/stuck_rate` | – | Assumption | 1e-5 / 1e-5 / 1e-6 |
| Defect-map seed | `defects.seed` | – | Model choice | 1234 |
| PDAF site fraction and QE factor | `pdaf.fraction/qe_factor` | – | Datasheet / assumption | 0.01 / 0.5 |
| Binning factor and domain | `binning.factor/mode` | – | Datasheet | 1 / `analog` |
| Blooming model | `blooming.enabled/fraction/iterations` | – | Model choice | `false` / 0.02 / 2 |
| Provenance | `provenance.source/url/retrieved/notes/assumptions` | – | Mandatory | Public datasheet + assumptions |

## Worked conversions to document in Task 17

The final guide will show the arithmetic for values that are rarely printed
directly:

- **Conversion gain** is often implied by the ADC full-scale voltage and the
  full-well capacity, or quoted in µV/e⁻; the two must agree within the stated
  analog-gain reference.
- **Read noise** may be quoted in DN or µV; dividing by conversion gain and
  analog gain gives electrons RMS.
- **Dark current** is often quoted at 60 °C or in mV/s. Converting to e⁻/s at
  the 25 °C reference requires the doubling temperature and must say so in
  `assumptions`.
- **Exposure limits** may be implied by frame rate or line time rather than a
  table; the derivation and its rounding are recorded.
- **QE and IR-cut curves** are usually graphs. Digitization resolution,
  wavelength registration, and interpolation assumptions are recorded as
  limitations of the preset.

## Open items (resolved in Task 17)

- Exact public datasheet URLs and retrieval dates for the flagship preset.
- Whether each spectral curve is loaded from a licensed dataset, digitized from
  a plot, or modelled from a representative silicon response.
- The final provenance wording used by `camsim validate` and the G2 report.

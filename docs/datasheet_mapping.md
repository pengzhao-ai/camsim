# Datasheet to JSON mapping

Sensor datasheets describe a part in prose, plots, and unit-bearing tables;
camsim consumes a validated JSON preset. This page defines how each datasheet
parameter becomes a schema field, whether the value is measured, derived, or
assumed, and where the provenance record lives.

**Status: complete for the sensor pipeline and carried into the current release.** The mapping covers
the full sensor preset, including the dual-conversion-gain modes, ADC scale,
noise, defects, and the optional spatial-crosstalk kernel. The shipped
`flagship_phone_50mp` preset is a **representative engineering design point**,
not a measurement of a specific commercial part. Every value that is an
estimate rather than a vendor measurement is listed in
`provenance.assumptions`, and the preset's `provenance.source` says so
explicitly.

## The preset and its provenance

The flagship preset describes a 1/1.3-inch, 50 MP stacked phone sensor with
dual conversion gain and an RGGB CFA. Its role is to exercise the full
irradiance → RAW chain with plausible, internally consistent values and to give
the sensor validation a known set of parameters to recover.

- `provenance.source`: representative engineering estimates grounded in public
  phone-sensor datasheet ranges.
- `provenance.notes`: no value is a measurement of a specific commercial part.
- `provenance.assumptions`: the complete list of estimates, including the
  spectral curves, fill factor, microlens gain, dark-current reference,
  fixed-pattern strengths, and the HCG crossover.
- The three spectral CSVs are bundled package resources under
  `src/camsim/data/spectra/` and ship inside the wheel. They are original
  synthetic engineering estimates, not digitized vendor plots.

If a real part is modelled later, add its datasheet URL and retrieval date to
`provenance`, keep the vendor's measurement conditions (temperature, analog
gain, exposure) in `notes`, and state which fields are still assumptions. Do not
replace `source` with a vendor name unless every headline value comes from that
document under the stated conditions.

## Conventions

- **Units are in the field name.** `pitch_um` is micrometres, `full_well_e` is
  electrons, `conversion_gain_uv_per_e` is microvolts per electron,
  `read_noise_e_rms` is electrons RMS, `dark_current_e_per_s` is electrons per
  second, `full_scale_uv` is microvolts, and time is seconds. A bare number is
  never accepted for a physical quantity.
- **Fractions, not percent.** `prnu_fraction` and every spectral value are
  dimensionless fractions in `[0, 1]`. The report field is named
  `prnu_percent` precisely because datasheets quote PRNU in percent.
- **Derived values are computed, not copied.** Field of view and
  pixels-per-degree come from `LensConfig` properties; voltage at full well is
  `full_well_e * conversion_gain_uv_per_e * analog_gain`; dark current at a
  non-reference temperature follows the doubling model. Storing a derived value
  would let it disagree with its inputs.
- **Cross-field checks are part of the mapping.** The schema rejects negative
  or non-finite values, an ADC black level outside the code range, an exposure
  default outside its limits, HCG full well larger than LCG full well,
  unnormalised colour matrices, and a spatial kernel that is ragged, negative,
  or does not sum to one. ADC-limited saturation is reported as a warning
  rather than an error, because many sensors intentionally clip at the ADC in
  high-gain modes.
- **Spectral curves are package resources.** `qe_csv`, `cfa_csv`, and
  `ir_cut_csv` name files relative to `camsim.data` on the common 360–830 nm,
  5 nm grid. The sensor resolves a relative path against the working directory,
  then the repository root, then the installed package; an installed wheel is
  self-contained.

## Complete field map

Values in the right-hand column are from the representative
`flagship_phone_50mp` preset. They are illustrative, not a specific part's
datasheet.

| Datasheet parameter | Schema path | Unit | Source type | Preset value / note |
|---|---|---|---|---|
| Pixel pitch | `pixel.pitch_um` | µm | Representative (datasheet-style) | 1.12 |
| Photodiode fill factor | `pixel.fill_factor` | – | Derived / assumption | 0.85 |
| Microlens collection gain | `pixel.microlens_gain` | – | Assumption | 1.15; effective aperture clips at 1.0 |
| Colour filter array | `cfa` | – | Datasheet | `RGGB` |
| Spectral sampling grid | `spectral.wavelength_nm.start/stop/step` | nm | Model choice | 360 / 830 / 5, 95 samples |
| Quantum-efficiency curve | `spectral.qe_csv` | – | Synthetic engineering estimate | `spectra/qe_flagship.csv`; peak 0.82 |
| CFA transmission curves | `spectral.cfa_csv` | – | Synthetic engineering estimate | `spectra/cfa_rggb.csv` |
| IR-cut transmission | `spectral.ir_cut_csv` | – | Synthetic engineering estimate | `spectra/ir_cut.csv`; 650 nm cut-on |
| Spectral crosstalk matrix | `spectral.spectral_crosstalk` | – | Assumption | 3×3 identity; rows sum to one |
| Spatial charge-diffusion kernel | `spatial_crosstalk` | – | Measured / assumption | `null`; optional normalized 2-D kernel |
| LCG conversion gain | `modes.LCG.conversion_gain_uv_per_e` | µV/e⁻ | Representative (datasheet-style) | 100.0 |
| LCG full well | `modes.LCG.full_well_e` | e⁻ | Representative (datasheet-style) | 6000 |
| LCG read noise | `modes.LCG.read_noise_e_rms` | e⁻ RMS | Representative (datasheet-style) | 2.0 |
| LCG dark current at 25 °C | `modes.LCG.dark_current_e_per_s` | e⁻/s | Representative (datasheet-style), normalized | 1.2 |
| HCG conversion gain | `modes.HCG.conversion_gain_uv_per_e` | µV/e⁻ | Representative (datasheet-style) | 300.0 |
| HCG full well | `modes.HCG.full_well_e` | e⁻ | Representative (datasheet-style) | 2000 |
| HCG read noise | `modes.HCG.read_noise_e_rms` | e⁻ RMS | Representative (datasheet-style) | 0.7 |
| HCG dark current at 25 °C | `modes.HCG.dark_current_e_per_s` | e⁻/s | Representative (datasheet-style), normalized | 1.2 |
| Analog-gain range and step | `analog_gain.min_db/max_db/step_db/default_db` | dB | Datasheet | 0 / 24 / 1 / 0 |
| ADC resolution | `adc.bits` | bit | Representative (datasheet-style) | 10 |
| ADC full-scale input | `adc.full_scale_uv` | µV | Representative (datasheet-style) | 800000.0 |
| ADC black level | `adc.black_level_dn` | DN | Representative (datasheet-style) | 64 |
| ADC digital gain | `adc.digital_gain` | – | Datasheet / model | 1.0 |
| Exposure limits and default | `exposure.min_s/max_s/default_s` | s | Datasheet / derived | 1e-4 / 0.1 / 0.01 |
| Reference and operating temperature | `temperature.reference_c/default_c` | °C | Datasheet | 25.0 / 25.0 |
| Dark-current doubling temperature | `temperature.dark_current_doubling_k` | K | Datasheet / literature | 7.0 |
| Dead-pixel rate | `defects.dead_rate` | – | Assumption | 1e-5 |
| Hot-pixel rate | `defects.hot_rate` | – | Assumption | 1e-5 |
| Stuck-pixel rate | `defects.stuck_rate` | – | Assumption | 1e-6 |
| Defect/PDAF map seed | `defects.seed` | – | Model choice | 1234 |
| DSNU | `noise.dsnu_e` | e⁻ RMS | Assumption | 0.5 |
| PRNU | `noise.prnu_fraction` | – | Assumption | 0.005 (= 0.5 %) |
| Row noise | `noise.row_noise_e_rms` | e⁻ RMS | Assumption | 0.5 |
| Hot-pixel dark multiplier | `noise.hot_pixel_dark_multiplier` | – | Assumption | 100.0 |
| PDAF site fraction | `pdaf.fraction` | – | Datasheet / assumption | 0.01 |
| PDAF QE factor | `pdaf.qe_factor` | – | Assumption | 0.5 |
| PDAF placement | `pdaf.pattern` | – | Datasheet | `sparse_green` |
| Binning factor and domain | `binning.factor/mode` | – | Datasheet | 1 / `analog` |
| Blooming model | `blooming.enabled/fraction/iterations` | – | Model choice | `false` / 0.02 / 2 |
| HCG/LCG auto crossover | `auto_hcg_threshold_e` | e⁻/pixel | Assumption | 200.0 |
| Provenance | `provenance.source/url/retrieved/notes/assumptions` | – | Mandatory | representative + assumptions |

## The percent/fraction trap

The single most common datasheet transcription error is a factor of 100:

| Datasheet phrase | Wrong value | Correct value |
|---|---|---|
| QE = 82 % | `82` in the QE CSV | `0.82` |
| CFA peak transmission = 90 % | `90` | `0.90` |
| PRNU = 0.5 % | `"prnu_fraction": 0.5` | `"prnu_fraction": 0.005` |

`prnu_fraction` is validated as `< 1.0`, but a value of `0.5` is still legal
and would mean 50 % — a 100× gain error that no downstream test can distinguish
from a broken sensor. Always compare the preset against the datasheet's unit
column, and remember that the EMVA report deliberately names its output
`prnu_percent`.

Not every row is a percent trap. Read noise, full well, dark current, pitch,
conversion gain, and ADC codes carry their unit in the field name and are
copied from the datasheet unchanged. Those fields fail differently: a read
noise entered as `2000` instead of `2.0` is a 1000× noise error, not a
fraction error, and a full well entered in ke⁻ instead of e⁻ clips every
highlight. Check the datasheet's unit column for every row, not only the
dimensionless ones.

## Worked conversions

### Conversion gain and ADC system gain

A datasheet may quote conversion gain directly in µV/e⁻, or imply it through
the ADC full-scale voltage and the full-well capacity. At unity analog gain,

```text
system_gain_DN_per_e = conversion_gain_uv_per_e * usable_codes / full_scale_uv
usable_codes = 2**bits - 1 - black_level_dn
```

For the flagship preset's LCG mode:

```text
usable_codes = 1023 - 64 = 959
K_LCG = 100.0 µV/e⁻ * 959 / 800000 µV = 0.119875 DN/e⁻
K_HCG = 300.0 µV/e⁻ * 959 / 800000 µV = 0.359625 DN/e⁻
```

The measured PTC slope must agree with this value at the configured analog gain.
The sensor validation measures 0.1193 DN/e⁻ for LCG and 0.3594 DN/e⁻ for HCG,
within 0.5 % of the nominal values.

The full-well voltage is `full_well_e * conversion_gain_uv_per_e` at unity
gain: 600000 µV for both modes. Because that is below the 800000 µV ADC full
scale, this preset is full-well limited in both modes. Raising analog gain
scales the voltage and can move a mode into ADC-limited saturation; the schema
reports that condition as a warning.

### Read noise from DN or µV

If a datasheet quotes read noise in DN, divide by the system gain at the stated
analog gain:

```text
read_noise_e = read_noise_DN / (K_at_that_gain)
```

If it quotes microvolts, divide by the conversion gain and the analog gain:

```text
read_noise_e = read_noise_uv / (conversion_gain_uv_per_e * 10**(gain_db/20))
```

State the gain mode and analog gain in `provenance.notes`; mixing LCG and HCG
quotes is the read-noise equivalent of mixing units.

### Dark current at a non-reference temperature

Datasheets often quote dark current at 60 °C or as mV/s. The model is

```text
D(T) = D_ref * 2**((T - T_ref) / T_double)
```

To normalize a 60 °C quote to the preset's 25 °C reference with a 7 K doubling
interval, divide by `2**((60-25)/7) = 32`. Record the original temperature and
the assumed doubling interval in `provenance.assumptions`; a preset whose
`reference_c` disagrees with the temperature at which its dark current was
quoted is off by exactly that factor.

### Analog gain in decibels

`gain_db` is a voltage gain, so the linear factor is `10**(gain_db/20)`.
6.0206 dB doubles the signal voltage. The ADC full-scale check and the read
noise conversion both use the same factor.

### Exposure limits

Exposure limits are often implied by frame rate or line time rather than a
table. Derive the maximum from the frame period minus readout overhead, round
conservatively, and record the derivation in `provenance.notes`. The default
must lie inside the limits; the schema rejects a preset whose default exposure
is unreachable.

## Assumptions and limitations of the flagship preset

- The preset is **representative, not measured**. Pixel pitch, full well,
  conversion gain, read noise, ADC scale, and gain range sit inside published
  phone-sensor ranges, but no single commercial part is claimed.
- The QE, CFA, and IR-cut curves are **synthetic engineering estimates**
  authored for this repository. They are not digitized vendor plots; the
  spectral shapes are plausible, not calibrated.
- Fill factor and microlens gain are engineering assumptions. The product is
  clipped at 1.0 before it becomes the effective aperture fraction.
- DSNU, PRNU, row noise, defect rates, the hot-pixel multiplier, and the
  HCG/LCG crossover are representative assumptions, not calibrations.
- The spectral-crosstalk matrix is an identity and the spatial-crosstalk kernel
  is disabled. Enabling either changes the measured PRNU and colour response;
  recalibrate before claiming a sensor validation pass with a non-identity mixing model.
- The model assumes a linear ADC, Gaussian read noise, Poisson shot noise, and
  a spatially uniform flat field. It does not model lens shading, PDAF
  artefacts beyond the flagged sites, temperature gradients, or row-wise gain
  calibration.
- The EMVA characterization is **simulated and EMVA 1288-style**, not a
  certified measurement.

## Reproducing the preset checks

```bash
uv run camsim validate src/camsim/config/presets/sensors/flagship_phone_50mp.json
uv run pytest tests/test_config.py tests/test_preset_validation.py -v
uv run python examples/04_emva_characterization.py
uv run python benchmarks/benchmark_pipeline.py --quick
```

The sensor validation report is `results/reports/g2_sensor.md`; its measured
values and assumptions are regenerated by the example. See
[Validation](validation.md) for the validation status, and the
[API reference](api.md) for the schema and loader signatures.

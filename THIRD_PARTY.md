# Third-party notices

`camsim` is licensed under Apache-2.0 (see `LICENSE`). It depends on, and in
some workflows embeds output from, the third-party material below. This file
records provenance and attribution; it does not relicense any component.

## Runtime dependencies

These packages are installed separately and are not modified or redistributed
by the `camsim` source tree. Their licences are reproduced in the installed
distributions.

| Component | Licence | Copyright / attribution |
|---|---|---|
| [NumPy](https://numpy.org/) | BSD-3-Clause (with bundled 0BSD, MIT, Zlib, CC0-1.0 components) | NumPy Developers |
| [SciPy](https://scipy.org/) | BSD-3-Clause | 2001-2002 Enthought, Inc.; 2003- SciPy Developers |
| [Pydantic](https://docs.pydantic.dev/) | MIT | Pydantic Services Inc. and contributors |
| [Matplotlib](https://matplotlib.org/) | Matplotlib License (PSF-based, BSD-compatible) | Matplotlib Development Team |
| [Pillow](https://python-pillow.org/) | MIT-CMU | Pillow contributors |
| [colour-science](https://www.colour-science.org/) | BSD-3-Clause | Copyright 2013 Colour Developers |

## colour-science

`colour-science` provides colour-space transforms, CIE standard-observer and
illuminant datasets, and reference implementations used to validate camsim's
own numerics. It is a runtime dependency and is not vendored here.

- Project: <https://www.colour-science.org/>
- Licence: BSD-3-Clause, <https://opensource.org/licenses/BSD-3-Clause>
- Copyright: Copyright 2013 Colour Developers
- Maintainer contact: colour-developers@colour-science.org

## ColorChecker spectral data

camsim's ColorChecker ground truth comes from the **BabelColor Average**
dataset distributed with `colour-science`:

- Python path:
  `colour.characterisation.datasets.colour_checkers.SDS_COLOURCHECKERS["BabelColor Average"]`.
- Contents: 24 spectral reflectance distributions on a 380-730 nm grid at
  10 nm. The values are an average derived from measurements of 30
  *ColorChecker Classic* charts.
- Original reference data: *ColorChecker Classic* published by GretagMacbeth
  (2005), as compiled by BabelColor (Danny Pascale).
- Upstream references: BabelColor, "The ColorChecker (since 1976!)"
  (<http://www.babelcolor.com/main_level/ColorChecker.htm>) and
  "ColorChecker RGB and spectra"
  (<http://www.babelcolor.com/download/ColorChecker_RGB_and_spectra.xls>).
  The upstream retrieval date recorded by `colour-science` is 2014-09-26.
- Licence: the `colour-science` distribution is BSD-3-Clause; the bundled
  BabelColor data carries the permission notice below.

The dataset carries the following upstream notices:

```text
Copyright 2013 Colour Developers
BabelColor ColorChecker data: Copyright (C) 2004-2012 Danny Pascale
(www.babelcolor.com); used by permission.
```

camsim reads this dataset at runtime from the installed `colour-science`
package. **No ColorChecker CSV files or copies of the spectra are committed to
this repository**, so the source tree does not redistribute the data; it
relies on the licence terms of the installed distribution. The same source,
BabelColor notice, and licence are written into every returned
`ColorCheckerData.provenance` mapping by `camsim.scene.spectral_data`.

*ColorChecker* is a trademark of X-Rite, Incorporated. The name is used
descriptively to identify the calibration target; camsim is not affiliated with
or endorsed by X-Rite.

## DejaVu fonts

Matplotlib bundles the DejaVu font family in
`matplotlib/mpl-data/fonts/ttf/`. camsim uses Matplotlib's default font for
documentation figures and notebook output. The fonts are not copied into the
camsim source tree.

- Fonts: DejaVu Sans, DejaVu Sans Mono, DejaVu Sans Display, and variants.
- Copyright: Bitstream Vera Fonts Copyright (c) 2003 Bitstream, Inc.;
  DejaVu changes are in the public domain.
- Licence: Bitstream Vera Fonts licence with the DejaVu public-domain
  amendments, <https://dejavu-fonts.github.io/License.html>.
- Upstream: <https://dejavu-fonts.github.io/>.

If a figure is generated outside Matplotlib or a font file is ever vendored,
its licence must be added here before the font is redistributed.

## Spectral datasets (Phase 1)

`camsim.scene.illuminants` and `camsim.scene.spectral_data` use the following
CIE data. All are read at runtime from the installed `colour-science` package
(BSD-3-Clause, <https://www.colour-science.org/>); **no spectral CSV files are
vendored in this repository**.

| Data | `colour-science` source | Use |
|---|---|---|
| CIE 1931 2-degree colour-matching functions | `colour.colorimetry.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]` | XYZ integration for the colour ground truth, 360-830 nm at 1 nm, resampled to 5 nm |
| CIE 1924 photopic luminous efficiency V(lambda) | `colour.colorimetry.datasets.SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"]` | Lux-to-irradiance normalisation, 360-830 nm at 1 nm; V(555 nm) = 1 |
| CIE standard illuminant D65 | `colour.colorimetry.SDS_ILLUMINANTS["D65"]` | D65 relative SPD, 300-780 nm at 5 nm; constant endpoint extrapolation to 830 nm |
| 3000 K and 4000 K Planckian spectra | `colour.colorimetry.blackbody.sd_blackbody` (generated, not a stored dataset) | Blackbody SPD from Planck's law with `c2 = 1.4388e-2 m K`, normalised to 100 at 560 nm |

Normative reference for all colorimetry: CIE 15:2004, *Colorimetry*, 3rd
edition. Upstream BabelColor references and licence notices for the
ColorChecker spectra are recorded in the preceding section. The exact dataset
key, source URL, licence, interpolation, and normalisation used for each
loaded object are recorded in its `provenance` mapping.

## Documentation and development tools

| Component | Licence | Use |
|---|---|---|
| [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) | MIT | Documentation site theme |
| [mkdocstrings](https://mkdocstrings.github.io/) | ISC | API documentation (from Phase 5) |
| [Jupytext](https://jupytext.readthedocs.io/) | MIT | Percent-format notebook sources |
| [nbconvert](https://nbconvert.readthedocs.io/) | BSD-3-Clause | Headless notebook execution |
| [pytest](https://pytest.org/) | MIT | Test suite |
| [Hypothesis](https://hypothesis.readthedocs.io/) | MPL-2.0 | Property-based tests |
| [Ruff](https://docs.astral.sh/ruff/) | MIT | Lint and format |
| [mypy](https://mypy-lang.org/) | MIT | Static type checking |
| [pre-commit](https://pre-commit.com/) | MIT | Local git hooks |
| [uv](https://docs.astral.sh/uv/) | Apache-2.0 OR MIT | Environment and package management |

## Optional extras and reference implementations

| Component | Licence | Use |
|---|---|---|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause-style; bundles third-party components under Apache-2.0, MIT, BSD, BSL-1.0, and compatible licences | Machine-vision experiments (after G5) |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause | ML utilities (optional) |
| [Optuna](https://optuna.org/) | MIT | ISP auto-tuning (optional) |
| [scikit-image](https://scikit-image.org/) | BSD-3-Clause | Reference cross-checks in tests |
| [colour-demosaicing](https://github.com/colour-science/colour-demosaicing) | BSD-3-Clause | Reference demosaic cross-checks in tests |

## Synthetic sensor spectral data

`data/spectra/qe_flagship.csv`, `data/spectra/cfa_rggb.csv`, and
`data/spectra/ir_cut.csv` are **original synthetic engineering estimates**
authored for this repository. They are not copied from a vendor datasheet, a
measurement report, or another dataset, and no third-party copyright applies;
they are distributed under the repository's Apache-2.0 licence.

- Purpose: let the shipped flagship sensor preset construct and run before
  measured curves are available.
- Shape: smooth representative QE (peak 0.82), overlapping RGGB colour-filter
  passbands, and an IR-cut roll-off near 650 nm, interpolated by the sensor
  onto the preset's 360-830 nm, 5 nm grid.
- Provenance: the assumptions are repeated in each CSV's comment header and in
  the preset's `provenance.assumptions`. Task 17 replaces these curves with
  datasheet-grounded values and records the final source, URL, and retrieval
  date in this file and in the preset.

## Adding third-party material

Before adding a dependency, dataset, font, or copied code snippet:

1. Confirm the licence is compatible with Apache-2.0 distribution.
2. Record the source URL, retrieval date, copyright line, and licence here.
3. For datasets, add the citation to the preset or dataset provenance metadata.
4. Do not vendor material whose licence forbids redistribution or requires
   terms that cannot be met by this repository.

Questions about third-party licensing belong in an issue before the material
is merged.

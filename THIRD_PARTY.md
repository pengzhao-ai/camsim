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

## ColorChecker 2005 spectral data

camsim's ColorChecker ground truth is planned to come from the
`ColorChecker 2005` colour checker dataset distributed with `colour-science`
(`colour.characterisation.datasets.colour_checkers`). The dataset carries the
following upstream notices:

```text
Copyright 2013 Colour Developers
BabelColor ColorChecker data: Copyright (C) 2004-2012 Danny Pascale
(www.babelcolor.com); used by permission.
```

- Original reference data: *ColorChecker Classic* published by GretagMacbeth
  (2005), as compiled by BabelColor (Danny Pascale).
- Upstream references: BabelColor, "The ColorChecker (since 1976!)" and
  "ColorChecker RGB and spectra" (<http://www.babelcolor.com>).
- Licence of the `colour-science` distribution: BSD-3-Clause.

**Status:** the exact dataset, whether it is loaded from `colour-science` or
vendored as CSV, and the final citation wording are confirmed in Phase 1
(Task 6) and then recorded here. No ColorChecker spectra are currently
committed to this repository.

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

## Planned spectral datasets (Phase 1)

Task 6 adds CIE 1931 2-degree colour-matching functions, the photopic luminous
efficiency function V(lambda), and D65, 3000 K, and 4000 K illuminant spectra.
The intended source is the `colour-science` dataset collection; if any series
is vendored as CSV, its source URL, retrieval date, copyright, and licence are
recorded in this file and in the dataset provenance metadata. CIE 15:2004 is
the normative reference for colorimetry.

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

## Adding third-party material

Before adding a dependency, dataset, font, or copied code snippet:

1. Confirm the licence is compatible with Apache-2.0 distribution.
2. Record the source URL, retrieval date, copyright line, and licence here.
3. For datasets, add the citation to the preset or dataset provenance metadata.
4. Do not vendor material whose licence forbids redistribution or requires
   terms that cannot be met by this repository.

Questions about third-party licensing belong in an issue before the material
is merged.

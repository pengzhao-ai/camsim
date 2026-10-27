"""Bundled spectral datasets shipped inside the installed camsim package.

The sensor needs a quantum-efficiency curve, CFA transmittance curves, and an
IR-cut transmittance curve before it can convert spectral irradiance into
electrons. Keeping those files inside the package rather than at the repository
root makes a wheel self-contained: an installed sensor must construct from
``site-packages/camsim/data`` without a source checkout or a particular working
directory.

The curves are dimensionless fractions in ``[0, 1]`` on a 360-830 nm, 5 nm
grid. They are representative synthetic engineering estimates, not measurements
of a commercial part; the provenance is repeated in each file's comment header,
in the flagship preset, and in ``THIRD_PARTY.md``.

Real-world failure artefacts guarded here: resolving data relative to the
current working directory (works in the repository, fails after installation);
treating a missing data file as an empty curve (a silently blind sensor); and
vendoring a percent-valued QE curve, which the sensor would reject as a units
error rather than scale down.

Only regular, unpacked wheel installs are supported. A zipimport-style install
would need ``importlib.resources.as_file`` for the lifetime of the sensor, which
is beyond the current packaging contract.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["data_path"]


def data_path(relative: str | Path) -> Path:
    """Return the filesystem path of a file bundled in ``camsim.data``.

    Parameters
    ----------
    relative:
        Path relative to this package, for example ``"spectra/qe_flagship.csv"``.
        An absolute path is returned unchanged when it names an existing file,
        which lets callers pass a user-supplied override through the same
        helper.

    Returns
    -------
    Path
        The resolved file path.

    Raises
    ------
    FileNotFoundError
        If the file is not bundled with the package (or, for an absolute path,
        does not exist). A missing spectral curve must stop sensor construction
        instead of producing an empty response.
    """
    candidate = Path(relative).expanduser()
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"spectral data file not found: {candidate}")
    resource = resources.files(__name__).joinpath(candidate.as_posix())
    if not resource.is_file():
        raise FileNotFoundError(
            f"spectral data file {candidate.as_posix()!r} is not bundled in camsim.data"
        )
    return Path(str(resource))

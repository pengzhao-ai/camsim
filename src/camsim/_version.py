"""Single source of truth for the package version.

``pyproject.toml`` reads this file through hatchling, so installed distribution
metadata and ``camsim.__version__`` cannot drift apart.
"""

__version__ = "0.1.0"

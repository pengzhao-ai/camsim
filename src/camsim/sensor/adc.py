"""ADC quantization, black-level pedestal, and digital gain.

An analog-to-digital converter maps a voltage onto a finite set of integer
codes. With ``b`` bits there are ``2**b`` codes, but the largest code is
reserved for saturation and the bottom of the range carries a black-level
pedestal, so only ``2**b - 1 - black_level_dn`` codes represent signal. The
transfer function is

``DN = black_level_dn + round(V / full_scale_uv * (2**b - 1 - black_level_dn))``

with the signal term clipped to its usable range. One signal code therefore
spans ``full_scale_uv / (2**b - 1 - black_level_dn)`` microvolts. A zero-volt
input yields exactly the black level; a voltage at or above ``full_scale_uv``
yields exactly ``2**b - 1``.

**Why a black level exists.** Read noise and dark-current excursions are
symmetric around zero volts, but an unsigned converter cannot represent
negative codes. The pedestal moves zero signal away from the bottom of the
range so those excursions are recorded instead of clipped. Typical pedestals
are 64 DN at 10 bits and 256 DN at 12 bits. The signal-bearing value is always
``DN - black_level_dn``; subtracting the pedestal twice (in the sensor and
again in the ISP black-level-correction stage) crushes shadows, while not
subtracting it leaves a grey cast that white balance turns into a colour cast.

**Digital gain** multiplies the *signal* code after conversion and before the
black level is re-added. That ordering keeps a dark frame at the black level
when gain is raised; applying gain to the final code would brighten the
pedestal as if it were light. Digital gain cannot recover information the ADC
already quantized away: it amplifies quantization noise along with the signal,
so it is a display/exposure convenience, not a sensitivity improvement.

Real-world ranges: 8-16 bits per pixel (10/12 bits for phone RAW, 14/16 bits
for scientific and industrial parts); full-scale input 0.5-2 V; analog gain
0-30 dB before this stage; digital gain 1-8x after it.

Failure artefacts guarded here: an off-by-one saturation code, a black level
that does not fit the code space, a full-scale voltage of zero or a negative
gain, digital gain applied to the pedestal, overflow when 16-bit values are
stored in a uint16 container, and NaN or infinity reaching an integer cast
(undefined behaviour in NumPy). The output is always uint16 so downstream
buffers have one storage dtype even for 8- and 12-bit modes; values are kept
inside the configured bit-depth range.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["adc_quantize"]

_MIN_BITS = 8
_MAX_BITS = 16


def adc_quantize(
    voltage_uv: ArrayLike,
    bits: int,
    full_scale_uv: float,
    black_level_dn: int,
    digital_gain: float = 1.0,
) -> NDArray[np.uint16]:
    """Quantize a sense-node voltage into black-level-offset ADC codes.

    Parameters
    ----------
    voltage_uv:
        Analog voltage in microvolts, a scalar or any array shape. Every value
        must be finite; negative values are clipped to the black level just as
        an unsigned converter would.
    bits:
        Converter resolution, an integer from 8 to 16 inclusive.
    full_scale_uv:
        Input voltage that produces the largest code, in microvolts, finite
        and strictly positive. The flagship preset uses 800,000 uV (0.8 V).
    black_level_dn:
        Code that represents zero signal. It must be an integer satisfying
        ``0 <= black_level_dn < 2**bits``; typical values are 64 at 10 bits
        and 256 at 12 bits.
    digital_gain:
        Positive finite multiplier applied to the signal code after
        conversion, before the black level is re-added. Defaults to unity.

    Returns
    -------
    A uint16 array with the same shape as ``voltage_uv`` whose values lie in
    ``[0, 2**bits - 1]``. Scalar input returns a zero-dimensional array.

    Raises
    ------
    ValueError
        If the bit depth, black level, full scale, digital gain, or any
        voltage is out of range or non-finite. Each message names the field.

    Notes
    -----
    The mapping rounds to the nearest code with NumPy's ``rint`` (ties to
    even), the same convention as Python's built-in ``round``. The signal code
    is clipped to ``[0, 2**bits - 1 - black_level_dn]`` before digital gain,
    and the final code is clipped to ``[0, 2**bits - 1]`` after the pedestal
    is restored. These are two distinct clips: the first models the converter
    saturating at full scale, the second guarantees a legal unsigned code even
    when digital gain pushes a bright signal past the largest code.

    Charge-domain saturation is *not* applied here.
    :func:`camsim.sensor.gain.clamp_full_well` clips the electron count before
    conversion, so a sensor can be charge limited (full well reached below ADC
    full scale) or ADC limited (full scale reached before full well) and this
    function behaves correctly in both cases.
    """
    bit_depth = _validate_bits(bits)
    black = _validate_black_level(black_level_dn, bit_depth)
    full_scale = _validate_positive_real(full_scale_uv, "full_scale_uv")
    gain = _validate_positive_real(digital_gain, "digital_gain")
    voltages = _finite_array(voltage_uv, "voltage_uv")

    max_code = (1 << bit_depth) - 1
    usable_codes = max_code - black
    # Clipping the voltage ratio to [0, 1] before scaling is equivalent to
    # clipping the signal code afterwards, and it keeps an extreme (but finite)
    # voltage and a tiny full scale from overflowing to infinity. Digital gain
    # may still overflow; any such value saturates, so do not emit a warning.
    with np.errstate(over="ignore"):
        ratio = np.clip(voltages / full_scale, 0.0, 1.0)
        signal_dn = np.rint(ratio * usable_codes)
        gained_dn = np.rint(signal_dn * gain)
        codes = np.clip(gained_dn + black, 0.0, max_code)
    return np.asarray(codes, dtype=np.uint16)


def _validate_bits(bits: object) -> int:
    """Return a supported ADC bit depth as a Python integer."""
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)):
        raise ValueError(f"bits must be an integer in [{_MIN_BITS}, {_MAX_BITS}]; got {bits!r}")
    value = int(bits)
    if not _MIN_BITS <= value <= _MAX_BITS:
        raise ValueError(
            f"bits must be in [{_MIN_BITS}, {_MAX_BITS}]; got {value}. "
            "8-16 bit converters cover the supported RAW formats"
        )
    return value


def _validate_black_level(black_level_dn: object, bits: int) -> int:
    """Return a black level that fits inside the ``bits`` code space."""
    if isinstance(black_level_dn, bool) or not isinstance(black_level_dn, (int, np.integer)):
        raise ValueError(f"black_level_dn must be an integer; got {black_level_dn!r}")
    value = int(black_level_dn)
    code_count = 1 << bits
    if not 0 <= value < code_count:
        raise ValueError(
            f"black_level_dn must satisfy 0 <= black_level_dn < 2**bits = "
            f"{code_count} for bits={bits}; got {value}"
        )
    return value


def _finite_array(values: ArrayLike, field: str) -> NDArray[np.float64]:
    """Return ``values`` as a finite float64 array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a real scalar or array of numbers; got {values!r}"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be finite (no NaN or infinity)")
    return array


def _validate_finite_real(value: object, field: str) -> float:
    """Return ``value`` as a finite Python float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return result


def _validate_positive_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, strictly positive Python float."""
    result = _validate_finite_real(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive; got {result!r}")
    return result

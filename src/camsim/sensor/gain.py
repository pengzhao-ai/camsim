"""Full-well saturation, charge-to-voltage conversion, and analog gain.

The readout chain that follows the photodiode has three distinct stages, and
each has a limit that must not be confused with the others:

**Full-well clipping.** A photodiode stores charge in a fixed capacitance, so it
saturates at ``full_well_e`` electrons. Consumer sensors hold roughly 2,000 to
50,000 e-; dual-conversion-gain pixels trade full well for read noise, with the
high-conversion-gain mode often holding a third of the low-conversion-gain
charge. Saturation is a charge-domain effect: once the well is full, additional
photons are lost (and, in a real array, may spill into neighbours as blooming).
Clipping the electron count first keeps every later stage working with charge
the pixel can actually hold.

**Conversion gain.** The sense node turns charge into voltage as ``V = N * CG``.
Conversion gain is quoted in microvolts per electron, typically 40-500 uV/e-.
A 100 uV/e- node develops 0.6 V at 6000 e-, while a 300 uV/e- node develops the
same 0.6 V at 2000 e-. The choice therefore sets the trade-off between signal
swing and charge capacity, and a conversion gain supplied in V/e- instead of
uV/e- is wrong by a factor of one million.

**Analog gain.** A programmable gain amplifier multiplies the node voltage by
``10 ** (gain_db / 20)`` because the datasheet quotes gain in decibels: 6.0206 dB
is exactly 2x, and 20 dB is 10x. Consumer sensors expose roughly 0-30 dB. Analog
gain applied before the ADC can reduce the *relative* contribution of the ADC's
own input-referred noise, but it cannot improve shot-noise-limited SNR: signal
and shot noise are amplified together. A negative dB value is a legal
attenuation; it is not the same as a negative gain factor.

Ordering is part of the contract: clamp charge to the full well, convert to
voltage, apply analog gain, then quantize. Applying analog gain in the digital
domain instead would amplify quantization noise and could clip before the
brightest highlights are converted.

Failure artefacts guarded here: a negative or non-finite full well (which
would invert or erase the image), non-finite electron counts reaching a cast,
a zero or negative conversion gain, a gain in dB used as a linear multiplier,
and a finite gain that overflows the floating-point range. Every violation
raises with the offending field named rather than silently clamping.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["apply_analog_gain", "clamp_full_well", "electrons_to_voltage"]


def clamp_full_well(electrons: ArrayLike, full_well_e: float) -> NDArray[np.float64]:
    """Clip an electron count to the photodiode's charge capacity.

    Parameters
    ----------
    electrons:
        Charge in electrons, a scalar or any array shape. Values may be
        negative (for example a read-noise excursion before the well is
        read); they are clipped to zero. Every value must be finite.
    full_well_e:
        Charge-domain saturation in electrons, finite and non-negative. Zero
        is accepted and models a fully drained well, which clips all charge.

    Returns
    -------
    A float64 array with the same shape as ``electrons``, in
    ``[0, full_well_e]``. Scalar input returns a zero-dimensional array. The
    input is never modified.

    Notes
    -----
    This is charge-domain clipping, not ADC clipping. The two limits are
    independent: a sensor whose full-well voltage is below the ADC full scale
    is charge limited and never reaches code saturation, while one whose
    conversion gain is too high saturates in the ADC first. The sensor
    pipeline applies this clamp before conversion and lets
    :func:`camsim.sensor.adc.adc_quantize` perform the separate code-domain
    clip.
    """
    well = _validate_non_negative_real(full_well_e, "full_well_e")
    charge = _finite_array(electrons, "electrons")
    return np.clip(charge, 0.0, well)


def electrons_to_voltage(
    electrons: ArrayLike,
    conversion_gain_uv_per_e: float,
) -> NDArray[np.float64]:
    """Convert charge to sense-node voltage with ``V = N * CG``.

    Parameters
    ----------
    electrons:
        Charge in electrons, a scalar or any array shape. Every value must be
        finite. Negative values are accepted because this is a linear
        conversion; the pipeline clips charge first, and the ADC clips any
        resulting negative voltage to the black level.
    conversion_gain_uv_per_e:
        Sense-node conversion gain in microvolts per electron, finite and
        strictly positive. Typical values are 40-500 uV/e-.

    Returns
    -------
    A float64 array of microvolts with the same shape as ``electrons``.

    Raises
    ------
    ValueError
        If the conversion gain is zero, negative, or non-finite; if the charge
        is non-finite; or if the product overflows the float64 range.

    Notes
    -----
    The unit is microvolts, matching the configuration schema's
    ``conversion_gain_uv_per_e`` and ``full_scale_uv``. Keep the whole signal
    chain in one unit: mixing V/e- with uV full scale misplaces every code by
    six orders of magnitude. An overflow here would otherwise reach the ADC as
    infinity and be rejected there with a less specific error.
    """
    gain = _validate_positive_real(conversion_gain_uv_per_e, "conversion_gain_uv_per_e")
    charge = _finite_array(electrons, "electrons")
    # Overflow is detected explicitly below; suppress NumPy's warning so the
    # diagnosis is the ValueError with the offending field, not a stray log.
    with np.errstate(over="ignore"):
        voltage = charge * gain
    if not np.all(np.isfinite(voltage)):
        raise ValueError(
            "electrons times conversion_gain_uv_per_e is not finite; "
            "check the charge and conversion gain magnitudes"
        )
    return voltage


def apply_analog_gain(voltage_uv: ArrayLike, gain_db: float) -> NDArray[np.float64]:
    """Apply programmable analog gain expressed in decibels.

    The model is ``V_out = V_in * 10 ** (gain_db / 20)``. Because the factor is
    a voltage ratio, 6.0206 dB doubles the signal and 20 dB multiplies it by
    ten. Gain is applied before the ADC, so it drives the converter closer to
    full scale and can turn a highlight into a clipped code; the benefit is
    that the signal then sits farther above the ADC's input-referred noise.

    Parameters
    ----------
    voltage_uv:
        Sense-node voltage in microvolts, a scalar or any array shape. Every
        value must be finite.
    gain_db:
        Analog voltage gain in decibels, finite. Negative values attenuate,
        zero is the identity, and 0-30 dB covers consumer sensor settings.

    Returns
    -------
    A float64 array of microvolts with the same shape as ``voltage_uv``.

    Raises
    ------
    ValueError
        If ``gain_db`` or the voltage is non-finite, if the dB factor
        overflows, or if the amplified voltage is not finite.

    Notes
    -----
    Analog gain does not change shot-noise-limited SNR: the signal and its
    Poisson noise are scaled by the same factor. Its real value is reducing the
    relative weight of downstream (ADC) noise, which is why high-gain modes
    exist on dark scenes. Applying this factor after quantization instead would
    amplify quantization noise and the black-level pedestal, so this function
    stays in the analog voltage domain by construction.
    """
    gain = _validate_finite_real(gain_db, "gain_db")
    voltage = _finite_array(voltage_uv, "voltage_uv")
    try:
        factor = math.pow(10.0, gain / 20.0)
    except OverflowError as exc:
        raise ValueError(f"gain_db ({gain:g} dB) overflows a linear voltage gain factor") from exc
    with np.errstate(over="ignore"):
        amplified = voltage * factor
    if not np.all(np.isfinite(amplified)):
        raise ValueError("voltage_uv after analog gain is not finite; check voltage_uv and gain_db")
    return amplified


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


def _validate_non_negative_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, non-negative Python float."""
    result = _validate_finite_real(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative; got {result!r}")
    return result


def _validate_positive_real(value: object, field: str) -> float:
    """Return ``value`` as a finite, strictly positive Python float."""
    result = _validate_finite_real(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive; got {result!r}")
    return result

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import piexif  # type: ignore


def _extract_shutter_time(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    TAG_EXPOSURE_TIME = 33434  # ExposureTime (seconds)
    TAG_SHUTTER_SPEED_VALUE = 37377  # ShutterSpeedValue (APEX Tv)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    if TAG_EXPOSURE_TIME in exif_ifd:
        num, den = exif_ifd[TAG_EXPOSURE_TIME]
        seconds = num / den
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    if TAG_SHUTTER_SPEED_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_SHUTTER_SPEED_VALUE]
        tv = num / den
        seconds = math.pow(2.0, -tv)
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    return None


def _extract_aperture_fnumber(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    TAG_FNUMBER = 33437  # FNumber (f-number)
    TAG_APERTURE_VALUE = 37378  # ApertureValue (APEX Av)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    if TAG_FNUMBER in exif_ifd:
        num, den = exif_ifd[TAG_FNUMBER]
        fnum = num / den
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    if TAG_APERTURE_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_APERTURE_VALUE]
        av = num / den
        fnum = math.pow(2.0, av / 2.0)
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    return None


def _extract_iso(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    # PhotographicSensitivity / ISOSpeedRatings
    TAG_PHOTOGRAPHIC_SENSITIVITY = 34855
    TAG_STANDARD_OUTPUT_SENSITIVITY = 34857  # StandardOutputSensitivity (SOS)
    TAG_RECOMMENDED_EXPOSURE_INDEX = 34858  # RecommendedExposureIndex (REI)
    TAG_ISO_SPEED = 34859  # ISOSpeed
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    candidates: List[int] = [
        TAG_PHOTOGRAPHIC_SENSITIVITY,
        TAG_RECOMMENDED_EXPOSURE_INDEX,
        TAG_STANDARD_OUTPUT_SENSITIVITY,
        TAG_ISO_SPEED,
    ]

    for tag in candidates:
        if tag in exif_ifd:
            value = float(exif_ifd[tag])
            if value > 0.0 and math.isfinite(value):
                return value

    return None


def compute_exposure_from_exif(path: Path) -> Optional[float]:
    """Return exposure in EV stops (log2 of relative exposure) or None if unavailable.

    Relative exposure is computed as (seconds / f^2 * ISO) then converted via log2.
    Returns None if the file format doesn't support EXIF (e.g., PNG).
    """
    try:
        exif = piexif.load(str(path))
    except piexif.InvalidImageDataError:
        # File format doesn't support EXIF (e.g., PNG)
        return None
    shutter_s = _extract_shutter_time(exif)
    aperture_f = _extract_aperture_fnumber(exif)
    iso_value = _extract_iso(exif)

    # If none of the components are available, we cannot compute exposure
    if shutter_s is None and aperture_f is None and iso_value is None:
        return None

    # Use available components; treat missing ones as 1 for exposure calculation
    seconds = shutter_s if shutter_s is not None else 1.0
    f_number = aperture_f if aperture_f is not None else 1.0
    iso = iso_value if iso_value is not None else 1.0

    rel_exposure = (seconds / (f_number * f_number)) * iso
    if rel_exposure <= 0.0 or not math.isfinite(rel_exposure):
        return None
    return math.log2(rel_exposure)

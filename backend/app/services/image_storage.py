"""
Controlled image storage for crop-health uploads.

The blueprint's privacy rules are enforced here, in one place:

- uploads live under the backend's data directory, never a public web root;
- files are addressed by a random key + content hash, so a client cannot
  iterate someone else's photos;
- EXIF is stripped on capture (location and camera metadata are not needed for
  analysis and are not disclosed in the UI);
- nothing in this module returns a URL — the database stores a file reference
  and only the owning analysis record ever shows it;
- deletion is real, and is used by the retention endpoint.
"""

from __future__ import annotations

import hashlib
import io
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..schemas_crop_health import (
    MAX_DECODED_PIXELS,
    MAX_DIMENSION_PX,
    MIN_DIMENSION_PX,
)


class ImageError(ValueError):
    """The bytes are not an acceptable image. Message goes to the client."""


@dataclass
class StoredImage:
    """A sanitised, written image and everything worth recording about it."""

    reference: str
    path: Path
    filename: str
    sha256: str
    bytes_written: int
    width: int
    height: int
    format: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes_written,
            "width": self.width,
            "height": self.height,
            "format": self.format,
        }


def analyse_pixels(image: Image.Image) -> dict[str, Any]:
    """
    The small set of scalar statistics the quality gates and (eventually) the
    model consume. Pure numpy over one downsampled copy of the image.

    Returns dimensions, mean/std luma, a Laplacian variance for sharpness, and
    three colour fractions: leaf-green, dark and yellow. The colour fractions
    are coarse discriminators only — they gate nothing on their own and no
    diagnostic value is claimed for them.
    """
    import numpy as np

    small = image.convert("RGB")
    max_side = max(small.width, small.height)
    if max_side > 1024:
        scale = 1024 / max_side
        small = small.resize(
            (max(1, round(small.width * scale)), max(1, round(small.height * scale)))
        )
    array = np.asarray(small, dtype=np.float32)
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue

    # Laplacian variance: a single flat/blurry region variance collapses while
    # an in-focus leaf keeps texture. It is a gate, not a lens.
    laplacian = (
        luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
        - 4.0 * luma[1:-1, 1:-1]
    )
    sharpness = float(np.var(laplacian))

    total = array.shape[0] * array.shape[1]
    green_mask = (green > red + 8) & (green > blue + 8)
    dark_mask = luma < 60.0
    yellow_mask = (red > 120.0) & (green > 90.0) & (blue < 0.8 * green.min() + 60.0)
    return {
        "width": int(image.width),
        "height": int(image.height),
        "lumaMean": round(float(np.mean(luma)), 2),
        "lumaStd": round(float(np.std(luma)), 2),
        "sharpness": round(sharpness, 2),
        "greenFraction": round(float(np.count_nonzero(green_mask)) / total, 4),
        "darkFraction": round(float(np.count_nonzero(dark_mask)) / total, 4),
        "yellowFraction": round(float(np.count_nonzero(yellow_mask)) / total, 4),
    }


class ImageStore:
    """Writes and deletes sanitised uploads under a controlled root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def sanitise(self, filename: str, content_type: str, payload: bytes) -> StoredImage:
        """
        Decode, strip metadata and re-encode. Decode limits are enforced before
        the pixels are materialised so a malicious header cannot exhaust memory.
        """
        if not payload:
            raise ImageError("the uploaded file is empty")
        from ..schemas_crop_health import ACCEPTED_MIME_TYPES

        extensions = ACCEPTED_MIME_TYPES.get(content_type)
        if extensions is None:
            raise ImageError(
                f"unsupported file type {content_type!r}; send a JPEG, PNG or WebP photo"
            )
        suffix = Path(filename or "").suffix.lower()
        if suffix and not any(ext == suffix or f".{ext}" == suffix for ext in extensions):
            raise ImageError(
                f"file extension {suffix!r} does not match the content type {content_type!r}"
            )

        Image.MAX_IMAGE_PIXELS = MAX_DECODED_PIXELS
        try:
            Image.open(io.BytesIO(payload))
            opened = Image.open(io.BytesIO(payload))
        except ImageError:
            raise
        except Exception as error:
            raise ImageError("the file could not be decoded as an image") from error
        if opened.width > MAX_DIMENSION_PX or opened.height > MAX_DIMENSION_PX:
            raise ImageError(
                f"image is larger than the {MAX_DIMENSION_PX}px decode limit"
            )
        if opened.width < MIN_DIMENSION_PX or opened.height < MIN_DIMENSION_PX:
            raise ImageError(
                f"image is smaller than the {MIN_DIMENSION_PX}px minimum for leaf analysis"
            )

        # EXIF transpose first (so the orientation a phone wrote survives the
        # strip), then a fresh re-encode with no metadata block at all.
        transposed = ImageOps.exif_transpose(opened)
        clean = transposed.convert("RGB")
        buffer = io.BytesIO()
        clean.save(buffer, format="JPEG", quality=90)
        written = buffer.getvalue()

        now = datetime.now(timezone.utc)
        digest = hashlib.sha256(written).hexdigest()
        reference = f"{now.strftime('%Y/%m/%d')}/{secrets.token_hex(8)}.jpg"
        path = self.root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(written)
        return StoredImage(
            reference=reference,
            path=path,
            filename=f"{digest[:12]}.jpg",
            sha256=digest,
            bytes_written=len(written),
            width=clean.width,
            height=clean.height,
            format="JPEG",
        )

    def pixels(self, reference: str) -> tuple["Image.Image", dict[str, Any]]:
        """The stored file, re-opened, plus its scalar statistics."""
        path = self.root / reference
        if not path.exists():
            raise ImageError(f"stored image {reference!r} is gone")
        image = Image.open(path)
        return image, analyse_pixels(image)

    def delete(self, reference: str) -> bool:
        """Deletion is a real filesystem delete, or nothing if already gone."""
        path = self.root / reference
        if not path.exists():
            return False
        path.unlink()
        parent = path.parent
        if parent != self.root and not any(parent.iterdir()):
            parent.rmdir()
        return True

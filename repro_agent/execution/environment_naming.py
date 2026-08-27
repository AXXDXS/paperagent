"""Stable, human-readable names for controller-managed environments."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

_MAX_ENVIRONMENT_NAME_LENGTH = 63
_RESERVED_NAMES = {"base", "root"}


def managed_environment_name(
    requested_name: str | None,
    repository_path: str | Path,
) -> str:
    """Return a filesystem-safe environment name.

    An explicit user value wins.  Otherwise the repository directory name is
    used, which keeps ``conda env list`` readable.  The content fingerprint is
    deliberately *not* part of this display name; it remains in the managed
    marker and opaque ``conda://`` reference used for reuse validation.
    """

    raw = str(requested_name or "").strip()
    if not raw:
        raw = Path(str(repository_path).rstrip("/\\")).name.strip()
    raw = raw or "repro-environment"

    ascii_name = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", ascii_name)
    normalized = re.sub(r"[-_.]{2,}", "-", normalized).strip("-_.").lower()
    if not normalized:
        suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        normalized = f"repro-environment-{suffix}"
    if normalized in _RESERVED_NAMES:
        normalized = f"repro-{normalized}"
    return normalized[:_MAX_ENVIRONMENT_NAME_LENGTH].rstrip("-_.")

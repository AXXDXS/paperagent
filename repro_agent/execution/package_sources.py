"""Deterministic, bounded package-source failover for Conda environments."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit, urlunsplit


class SourceFailureKind(str, Enum):
    UNAVAILABLE = "unavailable"
    PACKAGE_NOT_FOUND = "package_not_found"
    AUTHENTICATION = "authentication"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    LOCAL_FAILURE = "local_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PackageSource:
    source_id: str
    location: str
    private: bool = False

    @property
    def display_location(self) -> str:
        if self.location == "defaults":
            return self.location
        parsed = urlsplit(self.location)
        host = parsed.hostname or "configured-source"
        path = "" if self.private else parsed.path
        return urlunsplit((parsed.scheme, host, path, "", ""))


@dataclass(frozen=True)
class PackageSourcePolicy:
    mode: str
    pip_sources: tuple[PackageSource, ...]
    conda_sources: tuple[PackageSource, ...]

    @classmethod
    def from_environment(cls, *, network_enabled: bool) -> "PackageSourcePolicy":
        pip_values = _configured_values("REPRO_AGENT_PIP_INDEX_URLS")
        singular_pip = os.environ.get("REPRO_AGENT_PIP_INDEX_URL", "").strip()
        if singular_pip:
            pip_values.insert(0, singular_pip)
        return cls.from_settings(
            network_enabled=network_enabled,
            mode=os.environ.get("REPRO_AGENT_MIRROR_POLICY", "auto"),
            pip_index_urls=pip_values,
            conda_channels=_configured_values("REPRO_AGENT_CONDA_CHANNELS"),
        )

    @classmethod
    def from_settings(
        cls,
        *,
        network_enabled: bool,
        mode: str = "auto",
        pip_index_urls: list[str] | tuple[str, ...] = (),
        conda_channels: list[str] | tuple[str, ...] = (),
    ) -> "PackageSourcePolicy":
        mode = str(mode).strip().lower() or "auto"
        if mode not in {"auto", "fixed", "offline"}:
            raise ValueError("REPRO_AGENT_MIRROR_POLICY must be auto, fixed, or offline")
        if not network_enabled:
            mode = "offline"

        pip_sources = _sources(
            list(pip_index_urls),
            builtins=(
                ("tuna", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"),
                ("aliyun", "https://mirrors.aliyun.com/pypi/simple"),
                ("pypi", "https://pypi.org/simple"),
            ),
            mode=mode,
            prefix="pip",
            max_sources=4,
        )
        conda_sources = _sources(
            list(conda_channels),
            builtins=(
                (
                    "tuna-main",
                    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
                ),
                ("defaults", "defaults"),
            ),
            mode=mode,
            prefix="conda",
            max_sources=3,
        )
        return cls(mode=mode, pip_sources=pip_sources, conda_sources=conda_sources)

    def may_failover(
        self,
        *,
        source: PackageSource,
        failure_kind: SourceFailureKind,
    ) -> bool:
        if self.mode != "auto":
            return False
        if source.private:
            # Falling from a private index to a public one can install an
            # attacker-controlled namesake package. Require human approval.
            return False
        return failure_kind in {
            SourceFailureKind.UNAVAILABLE,
            SourceFailureKind.PACKAGE_NOT_FOUND,
        }


def classify_source_failure(stderr: str, stdout: str = "") -> SourceFailureKind:
    evidence = f"{stderr}\n{stdout}".lower()
    if any(
        marker in evidence
        for marker in (
            "401 client error",
            "403 client error",
            "http 401",
            "http 403",
            "unauthorized",
            "authentication failed",
            "forbidden",
        )
    ):
        return SourceFailureKind.AUTHENTICATION
    if any(
        marker in evidence
        for marker in (
            "unsatisfiableerror",
            "resolutionimpossible",
            "conflicting dependencies",
            "the conflict is caused by",
        )
    ):
        return SourceFailureKind.DEPENDENCY_CONFLICT
    if any(
        marker in evidence
        for marker in (
            "packagesnotfounderror",
            "no matching distribution found",
            "could not find a version that satisfies",
            "404 client error",
            "http 404",
        )
    ):
        return SourceFailureKind.PACKAGE_NOT_FOUND
    if any(
        marker in evidence
        for marker in (
            "no space left on device",
            "permission denied",
            "operation not permitted",
            "invalid requirement",
            "cancelled_by_controller",
            "timeout_killed",
            "log_limit_exceeded",
        )
    ):
        return SourceFailureKind.LOCAL_FAILURE
    if any(
        marker in evidence
        for marker in (
            "condahttperror",
            "unavailableinvalidchannel",
            "connection refused",
            "connection reset",
            "connection aborted",
            "connection error",
            "connecttimeout",
            "readtimeout",
            "read timed out",
            "temporary failure in name resolution",
            "name or service not known",
            "nodename nor servname provided",
            "sslerror",
            "certificate verify failed",
            "too many requests",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "hashes are required",
            "hash mismatch",
            "checksum mismatch",
        )
    ):
        return SourceFailureKind.UNAVAILABLE
    return SourceFailureKind.UNKNOWN


def source_attempt(
    source: PackageSource,
    *,
    phase: str,
    exit_code: int,
    failure_kind: SourceFailureKind | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "source": source.source_id,
        "location": source.display_location,
        "exit_code": exit_code,
        "failure_kind": failure_kind.value if failure_kind else "",
    }


def _configured_values(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must contain a JSON list or comma-separated URLs") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{name} must contain a JSON list")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sources(
    configured: list[str],
    *,
    builtins: tuple[tuple[str, str], ...],
    mode: str,
    prefix: str,
    max_sources: int,
) -> tuple[PackageSource, ...]:
    if mode == "offline":
        return ()
    values: list[PackageSource] = []
    known_locations = {location.rstrip("/"): source_id for source_id, location in builtins}
    for index, location in enumerate(configured):
        normalized = _validate_location(location)
        known_id = known_locations.get(normalized.rstrip("/"))
        values.append(
            PackageSource(
                source_id=known_id or f"configured-{prefix}-{index + 1}",
                location=normalized,
                private=known_id is None,
            )
        )
    if mode == "auto" or not values:
        values.extend(
            PackageSource(source_id=source_id, location=location)
            for source_id, location in builtins
        )
    deduplicated: list[PackageSource] = []
    seen: set[str] = set()
    for source in values:
        key = source.location.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(source)
    if mode == "fixed":
        deduplicated = deduplicated[:1]
    return tuple(deduplicated[:max_sources])


def _validate_location(location: str) -> str:
    value = location.strip().rstrip("/")
    if value == "defaults":
        return value
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("package sources must use HTTPS URLs")
    if parsed.username or parsed.password:
        # Credentials may still be supplied by the package manager's native
        # credential mechanism, but never embedded in task-visible URLs.
        raise ValueError("package source URLs must not contain credentials")
    if parsed.query or parsed.fragment or re.search(r"\s", value):
        raise ValueError("invalid package source URL")
    return value

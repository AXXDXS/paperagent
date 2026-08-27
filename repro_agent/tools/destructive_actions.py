"""Detect destructive commands and bind approval to the exact argv."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePath

from repro_agent.tools.base import ToolPermissionError


_DIRECT_DELETE_COMMANDS = {
    "del",
    "erase",
    "rmdir",
    "rm",
    "shred",
    "trash",
    "trash-put",
    "unlink",
}
_SCRIPT_DELETE_PATTERN = re.compile(
    r"(?:^|[;&|\n]\s*)"
    r"(?:rm|rmdir|unlink|shred|trash(?:-put)?|del|erase)\b|"
    r"\bfind\b[^\n;&|]*\s-delete\b|"
    r"\bgit\s+(?:clean|rm)\b|"
    r"\bRemove-Item\b|"
    r"\b(?:os\.(?:remove|unlink|rmdir)|shutil\.rmtree|Path\([^\n]*\)\.unlink)\s*\(",
    re.IGNORECASE,
)
_SQL_DELETE_PATTERN = re.compile(
    r"\b(?:DELETE\s+FROM|DROP\s+(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DestructiveCommand:
    command: tuple[str, ...]
    fingerprint: str
    reasons: tuple[str, ...]


class DestructiveActionConfirmationRequired(ToolPermissionError):
    """Raised before a destructive command until a matching approval exists."""

    def __init__(self, inspection: DestructiveCommand):
        self.command = list(inspection.command)
        self.fingerprint = inspection.fingerprint
        self.reasons = list(inspection.reasons)
        super().__init__(
            "destructive command requires explicit human confirmation "
            f"(fingerprint={inspection.fingerprint}, reasons={', '.join(inspection.reasons)})"
        )


def command_fingerprint(command: list[str]) -> str:
    canonical = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inspect_destructive_command(command: list[str]) -> DestructiveCommand | None:
    """Conservatively identify explicit delete operations in an argv command."""

    if not command:
        return None
    argv = [str(part) for part in command]
    executable = PurePath(argv[0]).name.lower()
    reasons: list[str] = []

    if executable in _DIRECT_DELETE_COMMANDS:
        reasons.append(f"direct delete executable: {executable}")
    if executable == "git" and len(argv) > 1 and argv[1].lower() in {"clean", "rm"}:
        reasons.append(f"destructive git subcommand: {argv[1].lower()}")
    if executable == "find" and any(part.lower() == "-delete" for part in argv[1:]):
        reasons.append("find -delete")

    joined_arguments = "\n".join(argv[1:])
    if _SCRIPT_DELETE_PATTERN.search(joined_arguments):
        reasons.append("delete operation embedded in command arguments")
    if _SQL_DELETE_PATTERN.search(joined_arguments):
        reasons.append("destructive SQL operation")

    if not reasons:
        return None
    return DestructiveCommand(
        command=tuple(argv),
        fingerprint=command_fingerprint(argv),
        reasons=tuple(dict.fromkeys(reasons)),
    )

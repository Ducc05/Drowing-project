"""Small logging helpers that prevent accidental credential disclosure."""

from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = (
    re.compile(r"(?i)(postgres(?:ql)?://[^:/\s]+:)[^@/\s]+(@)"),
    re.compile(r"(?i)(\b(?:password|token|api[_-]?key|client_secret)\s*[=:]\s*)[^\s,;]+"),
)


def redact_secrets(value: object) -> str:
    text = str(value)
    text = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]\2", text)
    return _SECRET_PATTERNS[1].sub(r"\1[REDACTED]", text)


class RedactingFilter(logging.Filter):
    """Redact common inline credentials from messages and string arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            formatted = record.getMessage()
        except (TypeError, ValueError):
            formatted = " ".join((str(record.msg), *(str(value) for value in record.args)))
        record.msg = redact_secrets(formatted)
        record.args = ()
        return True

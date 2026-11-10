"""Exception types raised by pycordis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, final


class CordisError(Exception):
    """Framework error identified by a stable ``code``.

    Example: ``except CordisError as error: if error.code == "INACTIVE_EFFECT": ...``.
    """

    CODES: ClassVar[dict[str, str]] = {
        "INACTIVE_EFFECT": "cannot create effect on inactive context"
    }

    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message if message is not None else self.CODES.get(code, code))
        self.code = code


@final
@dataclass(frozen=True, slots=True)
class Issue:
    """A single validation problem located at ``path`` within a config.

    Example: ``Issue("expected int, got str", ("server", "port"))``.
    """

    message: str
    path: tuple[str | int, ...] = ()


class ValidationError(TypeError):
    """Raised when a plugin config fails schema validation.

    Example: ``except ValidationError as error: print(error.issues)``.
    """

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues = tuple(issues)
        lines = [
            f"  - {i.message} (at {'.'.join(map(str, i.path))})" if i.path else f"  - {i.message}"
            for i in self.issues
        ]
        super().__init__("invalid config:\n" + "\n".join(lines))


class ServiceAccessError(AttributeError):
    """Raised when a context property cannot be read, written or declared.

    Subclasses ``AttributeError`` so ``hasattr`` / ``getattr(ctx, name, default)``
    keep their usual meaning on contexts.

    Example: ``except ServiceAccessError as error: print(error.name)``.
    """

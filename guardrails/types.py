"""
Guardrail Types and Data Structures
Defines standard result containers and actions for all guardrail layers.
"""

from dataclasses import dataclass
from typing import Literal, Optional

GuardrailAction = Literal["allow", "block", "modify", "warn"]


@dataclass
class GuardrailResult:
    """Container holding evaluation outcome from an input or output guardrail."""
    allowed: bool
    action: GuardrailAction
    reason_code: str
    user_message: Optional[str] = None
    sanitized_text: Optional[str] = None

    @classmethod
    def allow(cls, text: Optional[str] = None) -> "GuardrailResult":
        """Factory for an approved check."""
        return cls(
            allowed=True,
            action="allow",
            reason_code="ok",
            user_message=None,
            sanitized_text=text,
        )

    @classmethod
    def block(cls, reason_code: str, user_message: str) -> "GuardrailResult":
        """Factory for a blocked check."""
        return cls(
            allowed=False,
            action="block",
            reason_code=reason_code,
            user_message=user_message,
            sanitized_text=None,
        )

    @classmethod
    def modify(cls, sanitized_text: str, reason_code: str = "modified", user_message: Optional[str] = None) -> "GuardrailResult":
        """Factory for an altered/sanitized text output."""
        return cls(
            allowed=True,
            action="modify",
            reason_code=reason_code,
            user_message=user_message,
            sanitized_text=sanitized_text,
        )

    @classmethod
    def warn(cls, reason_code: str, user_message: Optional[str] = None, text: Optional[str] = None) -> "GuardrailResult":
        """Factory for a flagged but allowed check."""
        return cls(
            allowed=True,
            action="warn",
            reason_code=reason_code,
            user_message=user_message,
            sanitized_text=text,
        )

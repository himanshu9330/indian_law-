"""
LexAI Guardrails Pipeline
Orchestrates sequential execution of input and output guardrail checks.
Stops at the first 'block' action and fails safe if any check raises an exception.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from guardrails.config import GUARDRAILS_CONFIG
from guardrails.logger import log_guardrail_event
from guardrails.types import GuardrailResult

logger = logging.getLogger("lexai.guardrails.pipeline")

GuardrailCheckFn = Callable[[str, Optional[Dict[str, Any]]], GuardrailResult]

INPUT_GUARDRAILS: List[GuardrailCheckFn] = []
OUTPUT_GUARDRAILS: List[GuardrailCheckFn] = []


def register_input_guardrail(fn: GuardrailCheckFn) -> GuardrailCheckFn:
    """Register a check in the input guardrails sequence."""
    INPUT_GUARDRAILS.append(fn)
    return fn


def register_output_guardrail(fn: GuardrailCheckFn) -> GuardrailCheckFn:
    """Register a check in the output guardrails sequence."""
    OUTPUT_GUARDRAILS.append(fn)
    return fn


def clear_guardrails() -> None:
    """Clear registries (useful for testing and reloading)."""
    INPUT_GUARDRAILS.clear()
    OUTPUT_GUARDRAILS.clear()


def run_input_guardrails(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Run each input check in order and stop at the first 'block'.
    Fails safe and fails visible if an individual check raises an exception.
    """
    if not GUARDRAILS_CONFIG.enable_input_guardrails:
        return GuardrailResult.allow(text=text)

    if not INPUT_GUARDRAILS:
        try:
            import guardrails.input_guardrails
        except Exception:
            pass

    current_text = text
    conv_id = context.get("conversation_id") if context else None
    was_modified = False
    last_modify_reason = "modified"
    last_warn_reason = None
    last_warn_message = None

    for check in INPUT_GUARDRAILS:
        try:
            res = check(current_text, context)
            if not res.allowed or res.action == "block":
                log_guardrail_event(
                    layer="input",
                    reason_code=res.reason_code,
                    action=res.action,
                    conversation_id=conv_id,
                    raw_text=text,
                )
                return res

            if res.action == "modify" and res.sanitized_text is not None:
                current_text = res.sanitized_text
                was_modified = True
                last_modify_reason = res.reason_code
                log_guardrail_event(
                    layer="input",
                    reason_code=res.reason_code,
                    action=res.action,
                    conversation_id=conv_id,
                    raw_text=text,
                )
            elif res.action == "warn":
                last_warn_reason = res.reason_code
                last_warn_message = res.user_message
                log_guardrail_event(
                    layer="input",
                    reason_code=res.reason_code,
                    action=res.action,
                    conversation_id=conv_id,
                    raw_text=text,
                )
        except Exception as e:
            # Fail safe and fail visible: log the error and continue safely
            logger.error(f"Input guardrail '{getattr(check, '__name__', str(check))}' threw an error: {e}", exc_info=True)

    if was_modified:
        return GuardrailResult.modify(
            sanitized_text=current_text,
            reason_code=last_warn_reason or last_modify_reason,
            user_message=last_warn_message,
        )
    if last_warn_reason:
        return GuardrailResult.warn(
            reason_code=last_warn_reason,
            user_message=last_warn_message,
            text=current_text,
        )
    return GuardrailResult.allow(text=current_text)


def run_output_guardrails(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Run each output check in order and stop at the first 'block'.
    Fails safe and fails visible if an individual check raises an exception.
    """
    if not GUARDRAILS_CONFIG.enable_output_guardrails:
        return GuardrailResult.allow(text=answer)

    if not OUTPUT_GUARDRAILS:
        try:
            import guardrails.output_guardrails
        except Exception:
            pass

    current_answer = answer
    conv_id = context.get("conversation_id") if context else None
    was_modified = False
    last_modify_reason = "modified"

    for check in OUTPUT_GUARDRAILS:
        try:
            res = check(current_answer, context)
            if not res.allowed or res.action == "block":
                log_guardrail_event(
                    layer="output",
                    reason_code=res.reason_code,
                    action=res.action,
                    conversation_id=conv_id,
                    raw_text=answer,
                )
                return res

            if res.action == "modify" and res.sanitized_text is not None:
                current_answer = res.sanitized_text
                was_modified = True
                last_modify_reason = res.reason_code
                log_guardrail_event(
                    layer="output",
                    reason_code=res.reason_code,
                    action=res.action,
                    conversation_id=conv_id,
                    raw_text=answer,
                )
            elif res.action == "warn":
                log_guardrail_event(
                    layer="output",
                    reason_code=res.reason_code,
                    action=res.action,
                    conversation_id=conv_id,
                    raw_text=answer,
                )
        except Exception as e:
            # Fail safe and fail visible: log the error and continue safely
            logger.error(f"Output guardrail '{getattr(check, '__name__', str(check))}' threw an error: {e}", exc_info=True)

    if was_modified:
        return GuardrailResult.modify(sanitized_text=current_answer, reason_code=last_modify_reason)
    return GuardrailResult.allow(text=current_answer)

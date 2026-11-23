"""
LexAI Guardrails Package
Modular multi-layered input/output safety checks, refusal management, and SQLite event auditing.
"""

from guardrails.config import GUARDRAILS_CONFIG, GuardrailsConfig
from guardrails.logger import init_guardrail_db, log_guardrail_event, mask_snippet
from guardrails.pipeline import (
    clear_guardrails,
    register_input_guardrail,
    register_output_guardrail,
    run_input_guardrails,
    run_output_guardrails,
)
from guardrails.rate_limiter import (
    check_rate_limits,
    init_rate_limiter_db,
    record_rate_limit_event,
    reset_rate_limits,
)
from guardrails.types import GuardrailAction, GuardrailResult

# Import guardrails modules to register all checks on package import
import guardrails.input_guardrails
import guardrails.output_guardrails

__all__ = [
    "GUARDRAILS_CONFIG",
    "GuardrailsConfig",
    "GuardrailAction",
    "GuardrailResult",
    "init_guardrail_db",
    "init_rate_limiter_db",
    "log_guardrail_event",
    "mask_snippet",
    "register_input_guardrail",
    "register_output_guardrail",
    "clear_guardrails",
    "run_input_guardrails",
    "run_output_guardrails",
    "check_rate_limits",
    "record_rate_limit_event",
    "reset_rate_limits",
]

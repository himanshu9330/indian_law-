# LexAI Guardrails Architecture & Reference Guide

LexAI implements a multi-layered, fail-safe guardrails architecture designed specifically for an Indian Legal AI Assistant. It enforces safety, compliance, privacy, and abuse protection without blocking legitimate inquiries into criminal provisions or statutory penalties (such as BNS Section 103 or IPC Section 302).

---

## 1. Core Design Principles

1. **Lawful Legal Explanation Allowed:** Explaining crimes, definitions, procedures, and statutory penalties is in scope. Only actionable intent to commit a crime, evade investigation, destroy evidence, or harass is blocked.
2. **Rule-Based First:** Fast, regex and heuristic checks evaluate requests in `< 1ms`. LLM classification is used only for ambiguous, high-emotion queries.
3. **Fail-Safe & Fail-Visible:** If an individual guardrail raises an exception, the pipeline catches it, logs the event to SQLite, and continues safely. The chat interface never crashes.
4. **Non-Blocking Emergency Routing:** Messages indicating immediate personal crises (domestic violence, suicide, child abuse, active arrest, sexual assault) are **never blocked**. Instead, emergency helpline banners (dynamically read from `HELPLINES` in `core/config.py`) are prepended to the response.
5. **Privacy Preserving:** All PII (Aadhaar, PAN, phone, email, bank account) is masked before transmission to LLMs and before audit logging in SQLite (`guardrail_events`).
6. **Decoupled Architecture:** All guardrail logic lives in the `guardrails/` package, configurable via `guardrails/config.py`.

---

## 2. The 4 Guardrail Layers

```
User Input
    │
    ▼
[ LAYER 1: Abuse Prevention & Rate Limiting ] ────► Block if burst/daily limit or repeated blocks
    │
    ▼
[ LAYER 2: Input Safety & Validation ] ──────────► Block if prompt injection, off-topic, or harmful intent; Mask PII
    │
    ▼
[ LAYER 3: Safety-Critical Emergency Routing ] ──► Prepend emergency helplines (Non-blocking)
    │
    ▼
[ Pipeline: Pinecone RAG + Gemini / Fallback ] ──► 1-retry with fallback on timeout/quota error
    │
    ▼
[ LAYER 4: Output Verification & Grounding ] ────► Soften guarantees, verify citations, mask output PII, append disclaimer
    │
    ▼
Response to User & SQLite Audit Log
```

### Layer 1: Rate Limiting & Abuse Protection (`guardrails/rate_limiter.py`)
- **Per-Minute Limit (`max_messages_per_minute` = 10):** Prevents spam bursts with friendly messaging (*"You're sending messages too fast, please wait a moment"*).
- **Per-Day Limit (`max_messages_per_day` = 100):** Protects daily quota.
- **Per-Conversation Limit (`max_messages_per_conversation` = 50):** Prevents runaway threads.
- **Repeated Block Cooldown:** 5 guardrail blocks in 10 minutes puts the session in a 60-second cooldown.
- **SQLite Persistence:** Counters are stored in SQLite `rate_limit_records` so browser refresh does not reset limits.

### Layer 2: Input Guardrails (`guardrails/input_guardrails.py`)
1. **Validation:** Rejects empty messages, strips non-printable control characters, and enforces character limits (max 4,000 characters).
2. **Prompt Injection & Jailbreak Defense:** Detects instruction overrides, system prompt extraction, hidden comments, DAN mode, and tag attacks.
3. **Scope & Domain Enforcement:** Fast keyword heuristics + LLM fallback redirect recipes, coding homework, and gossip while keeping all legal questions in scope.
4. **Harmful Intent vs. Educational Inquiry:** Actionable crime planning, evidence destruction, and police evasion are blocked. Neutral questions like *"What is the punishment for murder under BNS Section 103?"* pass freely.
5. **PII Masking:** Sanitizes 12-digit Aadhaar (`XXXXXXXX1234`), PAN (`XXXXX1234X`), Indian phone numbers (`+91 XXXXX XXXXX`), emails (`***@***.***`), and bank account numbers before sending text to the LLM.

### Layer 3: Safety-Critical Crisis Routing (`guardrails/emergency_router.py`)
Detects active emergencies and prepends supportive guidance with relevant 24/7 national helplines:
- **Domestic Violence:** Women Helpline (`181`), Police (`112`), NALSA (`15100`).
- **Suicide / Self-Harm:** Tele-MANAS (`14416`), Police (`112`), with brief guidance on Section 115 of the Mental Healthcare Act, 2017 (decriminalization).
- **Child Abuse:** Childline (`1098`), Police (`112`).
- **Active Detention / Arrest:** Legal Aid (`15100`), Police (`112`), Articles 21 & 22 rights.
- **Sexual Assault:** Women Helpline (`181`), Police (`112`).

*Helplines are dynamically queried from `core/config.py` `HELPLINES`—never hardcoded.*

### Layer 4: Output Guardrails (`guardrails/output_guardrails.py`)
1. **Gemini Explicit Safety Settings:** Explicit block thresholds on Harassment, Hate Speech, Sexually Explicit, and Dangerous Content; safe fallback message on blocked/empty responses.
2. **System Prompt Leak Defense:** Intercepts regurgitated prompt instructions and replaces them with a safe greeting.
3. **Output PII Masking:** Re-masks any sensitive numbers or emails in generated text.
4. **Overconfidence Softening:** Replaces guarantees (*"you will definitely win"*, *"100% legal"*) with realistic language (*"you may have a strong legal argument"*) and appends a judicial discretion note.
5. **Citation Grounding Check:** Extracts statute citations (e.g. `BNS Section 103`, `IPC 302`, `Article 21`) and cross-references them against retrieved Pinecone chunks, `BNS_IPC_MAPPING`, and foundational statutes. Unverified sections receive a small verification note without deleting the answer.
6. **Retrieval Confidence Rule:** If retrieval confidence is low (< 0.5) or RAG returns no documents, appends a general knowledge note. If confidence is missing, no note is shown and "0%" is never displayed.
7. **Single Disclaimer Line:** Appends `*This is general legal information, not a substitute for advice from a qualified advocate.*` once at the bottom in small muted text.

---

## 3. Configuration Reference (`guardrails/config.py`)

| Setting Key | Type | Default | Description |
|---|---|---|---|
| `enable_input_guardrails` | `bool` | `True` | Master toggle for all input checks |
| `enable_output_guardrails` | `bool` | `True` | Master toggle for all output checks |
| `enable_rate_limiting` | `bool` | `True` | Enforce per-minute, per-day, and cooldown limits |
| `max_messages_per_minute` | `int` | `10` | Max messages allowed per minute per session |
| `max_messages_per_day` | `int` | `100` | Max messages allowed per day per session |
| `max_messages_per_conversation` | `int` | `50` | Max user messages in a single thread |
| `repeated_block_threshold` | `int` | `5` | Block count triggering cooldown |
| `block_cooldown_seconds` | `int` | `60` | Cooldown duration after repeated blocks |
| `max_context_messages` | `int` | `10` | Most recent $N$ messages sent to LLM as context |
| `max_message_length` | `int` | `4000` | Maximum character length per user query |
| `rag_confidence_threshold` | `float` | `0.5` | Threshold below which general knowledge note applies |
| `enable_injection_check` | `bool` | `True` | Toggle prompt injection detector |
| `enable_scope_check` | `bool` | `True` | Toggle scope and domain enforcement |
| `enable_harmful_intent_check` | `bool` | `True` | Toggle harmful intent / crime helper block |
| `enable_pii_masking` | `bool` | `True` | Toggle Aadhaar/PAN/phone/email masking |
| `enable_overconfidence_check` | `bool` | `True` | Toggle overconfidence softening |
| `enable_citation_grounding_check` | `bool` | `True` | Toggle citation verification against sources |

---

## 4. How to Add a New Guardrail

### Adding an Input Guardrail
Decorate your check function with `@register_input_guardrail` in `guardrails/input_guardrails.py`:

```python
from guardrails.pipeline import register_input_guardrail
from guardrails.types import GuardrailResult

@register_input_guardrail
def my_custom_input_check(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    # 1. Inspect text or context
    if "forbidden_term" in text.lower():
        # Block with reason code and polite user message
        return GuardrailResult.block(
            reason_code="custom_forbidden_term",
            user_message="I cannot process queries containing this restricted terminology.",
        )
    # 2. Or modify text
    if "sanitize_me" in text:
        cleaned = text.replace("sanitize_me", "[CLEANED]")
        return GuardrailResult.modify(sanitized_text=cleaned, reason_code="sanitized")

    # 3. Or allow
    return GuardrailResult.allow(text)
```

### Adding an Output Guardrail
Decorate your output check function with `@register_output_guardrail` in `guardrails/output_guardrails.py`:

```python
from guardrails.pipeline import register_output_guardrail
from guardrails.types import GuardrailResult

@register_output_guardrail
def my_custom_output_check(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    if "unwanted_phrase" in answer:
        sanitized = answer.replace("unwanted_phrase", "suitable_phrase")
        return GuardrailResult.modify(sanitized_text=sanitized, reason_code="phrase_replaced")
    return GuardrailResult.allow(answer)
```

---

## 5. Audit Logging & SQLite Schema

Every guardrail trigger (`block`, `modify`, `warn`) is recorded in `data/lexai.db` in table `guardrail_events`:

```sql
CREATE TABLE guardrail_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    timestamp       TEXT NOT NULL,
    layer           TEXT NOT NULL,      -- 'input' or 'output'
    reason_code     TEXT NOT NULL,      -- e.g. 'prompt_injection', 'emergency_suicide'
    action          TEXT NOT NULL,      -- 'block', 'modify', 'warn'
    masked_snippet  TEXT                -- truncated and digit-masked snippet (never raw PII)
);
```

- **Data Minimization:** Raw sensitive input is never stored in `guardrail_events`. Digit-masking (`mask_snippet`) redacts all numbers to `*` and truncates to 32 characters.

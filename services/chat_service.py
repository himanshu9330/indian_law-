"""
LexAI Chat Service Layer
Coordinates conversation management, RAG retrieval, LLM fallback,
safety-critical emergency routing, structured legal advice persistence,
and multi-layered input/output guardrails.
"""

import json
import logging
import re
from typing import Any, Optional

from core.llm import call_llm
from core.pipeline import run_query
from guardrails import GUARDRAILS_CONFIG, run_input_guardrails, run_output_guardrails
from services.conversations import (
    add_message,
    create_conversation,
    get_conversation,
    get_messages,
)

logger = logging.getLogger("lexai.chat_service")


class ChatResponse(dict):
    """Container allowing both dict access (res['reply']) and tuple unpacking (reply, meta, cid)."""

    def __iter__(self):
        yield self["reply"]
        yield self["metadata"]
        yield self["conversation_id"]


def generate_followups(question: str, answer: str) -> list[str]:
    """
    Generate exactly 3 short, specific follow-up questions as a JSON array using the model.
    Parses safely. If it fails or the output is invalid, returns an empty list.
    Never returns fake or default suggestions.
    """
    if not question or not answer:
        return []

    prompt = f"""Given this user question about Indian law and the assistant's legal answer, suggest exactly 3 short, specific follow-up questions the user might ask next.
Return ONLY a valid JSON array of 3 strings, e.g. ["question 1", "question 2", "question 3"].
Do not include any explanation, markdown formatting, or preamble.

USER QUESTION:
{question}

ASSISTANT ANSWER:
{answer[:1000]}
"""
    try:
        raw_response = call_llm(
            prompt=prompt,
            system_prompt="You are a legal assistant that outputs ONLY a strict JSON array of 3 strings.",
            temperature=0.3,
        )

        if not raw_response or not raw_response.strip():
            return []

        cleaned = raw_response.strip()
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Extract JSON array substring if surrounded by extra text
        match = re.search(r"\[\s*\".*?\"\s*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        data = json.loads(cleaned)
        if isinstance(data, list):
            valid_questions = [
                str(q).strip().strip('"\'')
                for q in data
                if isinstance(q, (str, int, float)) and len(str(q).strip()) > 5
            ]
            if valid_questions:
                return valid_questions[:3]
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")

    return []


def send_message(conversation_id: Optional[str], user_text: str) -> ChatResponse:
    """
    Execute complete consultation cycle:
    1. Validates inputs & sets up conversation.
    2. Runs input guardrails (validation, injection, emergency routing, scope, harm, PII).
    3. If blocked: records refusal and returns early without calling LLM.
    4. If emergency (e.g. self-harm, domestic violence, active arrest):
       - Prepends calm supportive message with dynamic helpline numbers at TOP of response.
       - Suicide/self-harm: leads with support and keeps legal information brief.
    5. Runs RAG + LLM + Fallback with sanitized text.
    6. Saves assistant reply with metadata and returns response.
    """
    # 1. Validation & Conversation Initialization
    if not user_text or not user_text.strip():
        raise ValueError("User message cannot be empty or whitespace.")

    clean_text = user_text.strip()

    if conversation_id is not None:
        conv_meta = get_conversation(conversation_id)
        if not conv_meta:
            raise ValueError(f"Unknown conversation ID: {conversation_id}")
    else:
        # Title = first user question trimmed to about 40 characters
        title = clean_text[:40].strip()
        if len(clean_text) > 40:
            title = title.rstrip(".,;:?!- ") + "..."
        conversation_id = create_conversation(title=title)

    # 2. Run Input Guardrails
    guard_ctx = {"conversation_id": conversation_id}
    input_guard = run_input_guardrails(clean_text, context=guard_ctx)

    if not input_guard.allowed:
        logger.info(f"Message blocked by input guardrail: {input_guard.reason_code}")
        # Save user message marked as blocked so it won't be used as LLM context
        add_message(
            conversation_id=conversation_id,
            role="user",
            content=clean_text,
            metadata={
                "blocked": True,
                "reason_code": input_guard.reason_code,
                "action": input_guard.action,
            },
        )
        # Save blocked refusal as normal assistant reply
        refusal_reply = input_guard.user_message or GUARDRAILS_CONFIG.get_refusal(input_guard.reason_code)
        blocked_meta = {
            "risk": "blocked",
            "confidence": 0.0,
            "fallback_used": False,
            "statutes": [],
            "steps": [],
            "rights": [],
            "sources": [],
            "follow_ups": [],
            "answer_source": "guardrail_blocked",
            "reason_code": input_guard.reason_code,
            "guardrail_action": input_guard.action,
        }
        add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=refusal_reply,
            metadata=blocked_meta,
        )
        return ChatResponse({
            "reply": refusal_reply,
            "metadata": blocked_meta,
            "conversation_id": conversation_id,
        })

    # 3. Determine sanitized text for LLM (masked PII / stripped control characters)
    text_for_llm = input_guard.sanitized_text or clean_text

    # Save User Message to database
    add_message(
        conversation_id=conversation_id,
        role="user",
        content=clean_text,
        metadata={
            "pii_masked": guard_ctx.get("pii_masked", False),
            "emergency_type": guard_ctx.get("emergency_type"),
        },
    )

    emergency_type = guard_ctx.get("emergency_type")
    emergency_banner = guard_ctx.get("emergency_banner")

    # 4. Handle Suicide / Self-Harm with brief supportive legal summary
    if emergency_type == "suicide":
        suicide_legal_advice = (
            "### Legal Context & Citizen Rights (Mental Healthcare Act, 2017)\n"
            "- **Decriminalization of Suicide (Section 115)**: Under Section 115 of the Mental Healthcare Act, 2017, "
            "any person who attempts to commit suicide is presumed to be under severe stress and is NOT liable to prosecution or punishment under the penal code.\n"
            "- **Government Statutory Duty**: The Central and State Governments have an explicit legal duty to provide care, treatment, and rehabilitation.\n"
            "- **Right to Confidential Medical Support**: Every person has a legally protected right to access mental healthcare services in complete confidentiality and without fear of police inquiry.\n\n"
            "Please call the Tele-MANAS helpline (`14416`) or Emergency (`112`) right away. You are not alone and support is always here."
        )
        raw_reply = f"{emergency_banner}\n\n---\n\n{suicide_legal_advice}"
        sections = {
            "legal_summary": suicide_legal_advice,
            "relevant_laws": [{"section": "Section 115, Mental Healthcare Act, 2017", "description": "Presumption of severe stress; non-prosecution for suicide attempts"}],
            "action_roadmap": ["Contact Tele-MANAS helpline (14416) or emergency (112)", "Reach out to a medical professional or trusted confidant"],
            "important_rights": ["Right to access confidential healthcare without stigma", "Protection from prosecution under Section 115"],
            "lawyer_needed": "Not Needed",
            "lawyer_reason": "Immediate healthcare support is paramount.",
        }
        risk = "high"
        confidence = 1.0
        fallback_used = False
        statutes = sections["relevant_laws"]
        steps = sections["action_roadmap"]
        rights = sections["important_rights"]
        sources = []
        follow_ups = []
        answer_source = "emergency_router"
        fallback_reason = None
        latency = 0.05
    else:
        # 5. Build recent conversation context (Max N messages from config)
        past_msgs = get_messages(conversation_id) if conversation_id else []
        valid_history = [
            m for m in past_msgs[:-1]
            if not (m.get("metadata") or {}).get("blocked")
        ]
        recent_history = valid_history[-GUARDRAILS_CONFIG.max_context_messages:]
        history_lines = []
        for m in recent_history:
            role = "User" if m.get("role") == "user" else "Assistant"
            snip = str(m.get("content", ""))[:300].strip().replace("\n", " ")
            history_lines.append(f"{role}: {snip}")
        history_context = "\n".join(history_lines) if history_lines else ""

        # 6. Call Existing RAG + LLM + Fallback + Risk pipeline
        try:
            runner = run_query(query=text_for_llm, history_context=history_context)
            final_result = None
            for event in runner:
                if event.get("node") == "complete":
                    final_result = event.get("result")

            if not final_result:
                raise RuntimeError("Pipeline failed to produce a final legal response.")
        except Exception as e:
            logger.error(f"Legal pipeline execution failed: {e}")
            final_result = {
                "raw_answer": f"An error occurred while analyzing this legal matter: {str(e)}. Please try rephrasing your question.",
                "answer_sections": {
                    "legal_summary": "Service encountered a processing error.",
                    "relevant_laws": [],
                    "action_roadmap": [],
                    "important_rights": [],
                    "lawyer_needed": "Required",
                    "lawyer_reason": "Technical processing exception.",
                },
                "risk": "medium",
                "confidence_score": 0.0,
                "answer_source": "error_handler",
                "fallback_reason": str(e),
                "retrieved_chunks": [],
            }

        sections = final_result.get("answer_sections", {}) or {}
        raw_reply = final_result.get("raw_answer") or sections.get("legal_summary") or "Advice generated."

        # If an emergency was detected (domestic violence, arrest, sexual assault, child abuse),
        # prepend the calm supportive helpline message to the TOP of the response
        if emergency_banner:
            raw_reply = f"{emergency_banner}\n\n---\n\n{raw_reply}"

        # If PII was masked, attach privacy notice
        if guard_ctx.get("pii_masked"):
            raw_reply = f"{GUARDRAILS_CONFIG.pii_notice}\n\n{raw_reply}"

        risk = "high" if emergency_type else final_result.get("risk", "low")
        confidence = final_result.get("confidence_score", 0.0)
        fallback_used = (
            final_result.get("answer_source") in ("huggingface_fallback", "openrouter_qwen")
            or bool(final_result.get("fallback_reason"))
        )

        statutes = sections.get("relevant_laws", [])
        steps = sections.get("action_roadmap", [])
        rights = sections.get("important_rights", [])
        sources = final_result.get("retrieved_chunks", []) or final_result.get("sources", [])
        answer_source = final_result.get("answer_source", "RAG")
        fallback_reason = final_result.get("fallback_reason")
        latency = final_result.get("latency")

        # Generate dynamic follow-ups
        follow_ups = generate_followups(question=text_for_llm, answer=raw_reply)

    # 6. Run Output Guardrails (PII, Prompt Leak, Overconfidence, Grounding, Confidence, Disclaimer)
    output_ctx = {
        "conversation_id": conversation_id,
        "confidence_score": confidence if (confidence is not None and confidence > 0.0) else (final_result.get("confidence_score") if 'final_result' in locals() and final_result else None),
        "retrieved_chunks": sources,
        "user_query": text_for_llm,
    }
    out_res = run_output_guardrails(raw_reply, context=output_ctx)
    if not out_res.allowed or out_res.action == "block":
        raw_reply = out_res.user_message or "This response could not be displayed due to safety guidelines."
    elif out_res.sanitized_text:
        raw_reply = out_res.sanitized_text

    # 7. Assemble metadata
    metadata = {
        "risk": risk,
        "confidence": confidence,
        "fallback_used": fallback_used,
        "statutes": statutes,
        "steps": steps,
        "rights": rights,
        "sources": sources,
        "follow_ups": follow_ups,
        "answer_source": answer_source,
        "fallback_reason": fallback_reason,
        "lawyer_needed": sections.get("lawyer_needed", "Optional"),
        "lawyer_reason": sections.get("lawyer_reason", ""),
        "latency": latency,
        "pii_masked": guard_ctx.get("pii_masked", False),
        "emergency_type": emergency_type,
        "emergency_routed": bool(emergency_type),
    }

    # 8. Save Assistant Reply with metadata
    add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=raw_reply,
        metadata=metadata,
    )

    return ChatResponse({
        "reply": raw_reply,
        "metadata": metadata,
        "conversation_id": conversation_id,
    })

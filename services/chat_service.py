"""
LexAI Chat Service Layer
Coordinates conversation management, RAG retrieval, LLM fallback,
structured legal advice persistence, and dynamic follow-up inquiry generation.
"""

import json
import logging
import re
from typing import Any, Optional

from core.llm import call_llm
from core.pipeline import run_query
from services.conversations import (
    add_message,
    create_conversation,
    get_conversation,
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

    # Return empty list on failure — never return fake/default questions
    return []


def send_message(conversation_id: Optional[str], user_text: str) -> ChatResponse:
    """
    Execute complete consultation cycle:
    1. Validates inputs.
    2. If conversation_id is None, creates a conversation (title = first user question trimmed to ~40 chars).
    3. Saves the user message.
    4. Calls the EXISTING RAG + LLM + Fallback + Risk pipeline.
    5. Generates follow-ups and saves the AI reply with full metadata in database.
    6. Returns reply, metadata, and conversation_id.
    """
    # 1. Validation
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

    # 2. Save User Message
    add_message(
        conversation_id=conversation_id,
        role="user",
        content=clean_text,
    )

    # 3. Call Existing RAG + LLM + Fallback + Risk pipeline
    try:
        runner = run_query(query=clean_text)
        final_result = None
        for event in runner:
            if event.get("node") == "complete":
                final_result = event.get("result")

        if not final_result:
            raise RuntimeError("Pipeline failed to produce a final legal response.")
    except Exception as e:
        logger.error(f"Legal pipeline execution failed: {e}")
        # Provide resilient legal error reply if pipeline crashes
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

    risk = final_result.get("risk", "low")
    confidence = final_result.get("confidence_score", 0.0)
    fallback_used = (
        final_result.get("answer_source") in ("huggingface_fallback", "openrouter_qwen")
        or bool(final_result.get("fallback_reason"))
    )

    statutes = sections.get("relevant_laws", [])
    steps = sections.get("action_roadmap", [])
    rights = sections.get("important_rights", [])
    sources = final_result.get("retrieved_chunks", []) or final_result.get("sources", [])

    # 4. Generate dynamic follow-ups
    follow_ups = generate_followups(question=clean_text, answer=raw_reply)

    # 5. Assemble metadata
    metadata = {
        "risk": risk,
        "confidence": confidence,
        "fallback_used": fallback_used,
        "statutes": statutes,
        "steps": steps,
        "rights": rights,
        "sources": sources,
        "follow_ups": follow_ups,
        "answer_source": final_result.get("answer_source", "RAG"),
        "fallback_reason": final_result.get("fallback_reason"),
        "lawyer_needed": sections.get("lawyer_needed", "Optional"),
        "lawyer_reason": sections.get("lawyer_reason", ""),
        "latency": final_result.get("latency"),
    }

    # 6. Save Assistant Reply with metadata
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

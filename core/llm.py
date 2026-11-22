"""
Groq LLM integration and helper utilities for LexAI.
Supports both non-streaming and streaming inference, title generation,
and follow-up generation. Gracefully handles missing keys and provides
automatic fallback across active models (e.g. qwen/qwen3.8-27b, openai/gpt-oss-120b).
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Generator, Optional

try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    Groq = None
    _HAS_GROQ = False

from core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)
from core.prompts import (
    FOLLOWUP_GENERATOR,
    TITLE_GENERATOR,
)

# Ordered fallback models to try if the requested model returns 404 / model_not_found
FALLBACK_MODELS = [
    GROQ_MODEL,
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def _clean_key(key: Optional[str]) -> str:
    """Sanitize API key by stripping quotes and whitespace."""
    k = (key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")).strip().strip("\"' ")
    return k


def get_groq_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Return an initialized Groq client or None if API key is not set."""
    if not _HAS_GROQ or Groq is None:
        return None
    key = _clean_key(api_key)
    if not key or key in ("", "your_groq_api_key_here"):
        return None
    try:
        return Groq(api_key=key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None


def _call_groq_rest(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    api_key: str,
) -> str:
    """Direct REST API fallback for Groq chat completions using urllib."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "LexAI/1.0",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()


def _call_openrouter_llm(
    prompt: str,
    system_prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Optional[str]:
    """Helper to query OpenRouter (Qwen) when Groq key is absent or fails."""
    key = (api_key or os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY or "").strip().strip("\"' ")
    if not key or key in ("", "your_openrouter_api_key_here"):
        return None
    m = model or os.getenv("OPENROUTER_MODEL") or OPENROUTER_MODEL or "qwen/qwen-2.5-72b-instruct"
    try:
        from core.hf_fallback import _call_openrouter_api
        ans = _call_openrouter_api(
            prompt=prompt,
            system_prompt=system_prompt,
            model=m,
            api_key=key,
            temperature=temperature,
        )
        if ans and ans.strip():
            return ans.strip()
    except Exception as e:
        print(f"OpenRouter call error in llm.py: {e}")
    return None


def call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful legal AI assistant specializing in Indian law.",
    model: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None,
) -> str:
    """
    Standard LLM call using Groq with automatic model fallback,
    seamlessly routing to OpenRouter (Qwen) if Groq key is missing or encounters errors.
    """
    key = _clean_key(api_key)

    # 1. Attempt Groq if key is configured
    if key and key not in ("", "your_groq_api_key_here"):
        client = get_groq_client(key)
        primary_model = model or GROQ_MODEL
        models_to_try = [primary_model]
        for fm in FALLBACK_MODELS:
            if fm not in models_to_try:
                models_to_try.append(fm)

        # Attempt calling via SDK
        if client:
            for m in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                    )
                    content = response.choices[0].message.content or ""
                    if content.strip():
                        return content.strip()
                except Exception as e:
                    err_str = str(e)
                    if "model_not_found" in err_str or "does not exist" in err_str:
                        continue
                    if "401" in err_str or "invalid_api_key" in err_str:
                        break
                    continue

        # REST fallback
        for m in models_to_try:
            try:
                res = _call_groq_rest(prompt, system_prompt, m, temperature, key)
                if res.strip():
                    return res.strip()
            except Exception:
                continue

    # 2. Seamless fallback to OpenRouter (Qwen)
    or_res = _call_openrouter_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        model=openrouter_model,
        api_key=openrouter_api_key,
        temperature=temperature,
    )
    if or_res and or_res.strip():
        return or_res.strip()

    # 3. Graceful offline/demo response
    return (
        "I am your Indian Legal Assistant. To generate live responses, "
        "please ensure your OpenRouter API key or Groq API key is configured in the Settings tab or in your .env file."
    )


def stream_llm(
    prompt: str,
    system_prompt: str = "You are a helpful legal AI assistant specializing in Indian law.",
    model: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream tokens from Groq LLM.
    Yields chunks of text as they arrive.
    """
    key = _clean_key(api_key)
    client = get_groq_client(key)

    if not client or not key:
        demo_msg = (
            "### Disclaimer\n"
            "This information is provided for educational and general guidance only and should not be considered "
            "as formal legal advice. Please consult a qualified legal professional for your specific situation.\n\n"
            "### Lawyer Needed\n"
            "**Optional**\n"
            "Consulting an advocate is recommended to evaluate the facts and review documentary evidence.\n\n"
            "### Legal Summary\n"
            "Under Indian law, citizens and residents are protected by statutory frameworks and constitutional "
            "guarantees. In civil and criminal matters, timely documentation and appropriate forum selection are vital.\n\n"
            "### Relevant Laws\n"
            "- **Bharatiya Nyaya Sanhita (BNS) / IPC**: Substantive criminal provisions\n"
            "- **Bharatiya Nagarik Suraksha Sanhita (BNSS) / CrPC**: Procedure for reporting and inquiry\n"
            "- **Constitution of India**: Fundamental rights protection\n\n"
            "### Action Roadmap\n"
            "1. Gather and document all relevant communications, notices, and receipts.\n"
            "2. Send a formal legal notice or approach the concerned jurisdictional authority.\n"
            "3. Seek consultation with an advocate specializing in this subject if needed.\n\n"
            "### Important Rights\n"
            "- Right to fair hearing and due process\n"
            "- Right to receive acknowledgment for written complaints\n\n"
            "### Final Note\n"
            "Keep written records of every correspondence."
        )
        for word in demo_msg.split(" "):
            yield word + " "
        return

    models_to_try = [model or GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != (model or GROQ_MODEL)]

    for m in models_to_try:
        try:
            stream = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            return
        except Exception as e:
            if "model_not_found" in str(e) or "does not exist" in str(e):
                continue
            yield f"\n\n[Error streaming response: {str(e)}]"
            return

    yield "\n\n[Error: Unable to stream response from available models]"


def generate_chat_title(
    query: str,
    api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None,
) -> str:
    """Generate a 3-6 word descriptive chat title based on the first query."""
    prompt = TITLE_GENERATOR.format(query=query)
    title = call_llm(
        prompt=prompt,
        system_prompt="You generate concise 3 to 6 word titles for legal conversations.",
        temperature=0.2,
        api_key=api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )
    cleaned = title.strip().strip('"\'#*').replace("\n", " ")
    if len(cleaned) > 50 or "Demo Mode" in cleaned or "Error" in cleaned or not cleaned:
        words = query.strip().split()
        return " ".join(words[:5]) + ("..." if len(words) > 5 else "")
    return cleaned


def generate_followup_questions(
    query: str,
    summary: str,
    api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None,
) -> list[str]:
    """Generate 3 contextual follow-up questions."""
    prompt = FOLLOWUP_GENERATOR.format(query=query, summary=summary[:400])
    raw = call_llm(
        prompt=prompt,
        system_prompt="You suggest 3 short, relevant legal follow-up questions.",
        temperature=0.4,
        api_key=api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )
    questions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and any(line.startswith(prefix) for prefix in ("1.", "2.", "3.", "-", "*")):
            cleaned = line.lstrip("1234567890.-*• ").strip()
            if cleaned and len(cleaned) > 5 and "Demo Mode" not in cleaned:
                questions.append(cleaned)
    if not questions:
        questions = [
            "What documents or evidence do I need to collect?",
            "What is the limitation period or deadline to take action?",
            "What are the expected court fees or costs involved?",
        ]
    return questions[:3]

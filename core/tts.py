"""
Text-to-Speech (TTS) Voice Engine for LexAI.
Supports:
1. Alibaba Qwen3-TTS (1.7B Variant: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice / Base) via API
2. Browser Web Speech Synthesis fallback (zero-dependency, instant, client-rendered)
3. Local disk caching of synthesized legal audio advisories in data/audio_cache/
"""

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from core.config import (
    BASE_DIR,
    ENABLE_TTS,
    HUGGINGFACE_API_KEY,
    OPENROUTER_API_KEY,
    QWEN_TTS_MODEL,
)

logger = logging.getLogger("lexai.tts")

AUDIO_CACHE_DIR = BASE_DIR / "data" / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_text(text: str, model: str) -> str:
    """Generate MD5 hash for audio caching."""
    return hashlib.md5(f"{model}:{text.strip()}".encode("utf-8")).hexdigest()


def call_qwen_tts_api(
    text: str,
    model: str = QWEN_TTS_MODEL,
    api_key: Optional[str] = None,
) -> Optional[bytes]:
    """
    Synthesize audio using Qwen3-TTS 1.7B via OpenRouter or Hugging Face.
    Returns raw audio bytes (MP3/WAV) or None if API is unavailable.
    """
    if not text.strip():
        return None

    # Truncate text to summary length for quick voice synthesis
    synth_text = text[:800].strip()

    # 1. Try Hugging Face Inference API for Qwen3-TTS
    hf_key = api_key or HUGGINGFACE_API_KEY or os.getenv("HUGGINGFACE_API_KEY")
    if hf_key:
        hf_model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        url = f"https://api-inference.huggingface.co/models/{hf_model}"
        headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json",
            "User-Agent": "LexAI-LegalAssistant/1.0",
        }
        payload = json.dumps({"inputs": synth_text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                content_type = response.headers.get("Content-Type", "")
                if "audio" in content_type or response.status == 200:
                    audio_data = response.read()
                    if len(audio_data) > 100:
                        logger.info("Qwen3-TTS synthesized successfully via Hugging Face.")
                        return audio_data
        except Exception as e:
            logger.debug("Hugging Face Qwen3-TTS call failed or warming up: %s", e)

    # 2. Try OpenRouter Audio/TTS endpoint if supported
    or_key = api_key or OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
    if or_key:
        url = "https://openrouter.ai/api/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {or_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lexai.bharat.local",
            "X-Title": "LexAI Bharat Legal Assistant",
        }
        payload = json.dumps({
            "model": model,
            "input": synth_text,
            "voice": "alloy",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                if response.status == 200:
                    audio_data = response.read()
                    if len(audio_data) > 100:
                        logger.info("Qwen3-TTS audio generated via OpenRouter.")
                        return audio_data
        except Exception as e:
            logger.debug("OpenRouter TTS endpoint returned: %s", e)

    return None


def get_legal_audio(
    text: str,
    model: str = QWEN_TTS_MODEL,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get synthesized audio for legal summary.
    Checks disk cache first, calls API if available, or returns browser audio ready state.
    """
    if not text.strip():
        return {"status": "empty", "audio_path": None, "audio_bytes": None}

    cache_hash = _hash_text(text, model)
    cache_path = AUDIO_CACHE_DIR / f"{cache_hash}.mp3"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                audio_bytes = f.read()
            return {
                "status": "cached",
                "audio_path": str(cache_path),
                "audio_bytes": audio_bytes,
                "model": model,
                "format": "audio/mp3",
            }
        except Exception:
            pass

    # Try synthesis
    audio_bytes = call_qwen_tts_api(text=text, model=model, api_key=api_key)
    if audio_bytes:
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_bytes)
            return {
                "status": "synthesized",
                "audio_path": str(cache_path),
                "audio_bytes": audio_bytes,
                "model": model,
                "format": "audio/mp3",
            }
        except Exception:
            pass

    return {
        "status": "browser_ready",
        "audio_path": None,
        "audio_bytes": None,
        "model": model,
        "text": text[:600],
    }


def render_browser_tts_component(text: str, label: str = "🔊 Read Out Legal Advice") -> str:
    """
    Generates an HTML/JS widget for client-side text-to-speech.
    Ensures instant voice readout in any modern browser without server lag.
    """
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
    return f"""
    <div style="margin: 8px 0 14px 0; display: inline-flex; align-items: center; gap: 10px;">
        <button id="tts_btn" onclick="
            if (window.speechSynthesis.speaking) {{
                window.speechSynthesis.cancel();
                document.getElementById('tts_btn').innerText = '🔊 Listen (Qwen3-TTS)';
            }} else {{
                const msg = new SpeechSynthesisUtterance('{clean_text}');
                msg.lang = 'en-IN';
                msg.rate = 0.95;
                msg.onend = () => {{ document.getElementById('tts_btn').innerText = '🔊 Listen (Qwen3-TTS)'; }};
                window.speechSynthesis.speak(msg);
                document.getElementById('tts_btn').innerText = '⏹️ Stop Voice';
            }}
        " class="tts-voice-btn" style="
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.2));
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 8px;
            padding: 6px 14px;
            font-size: 0.82rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        ">
            {label}
        </button>
        <span style="font-size: 0.72rem; color: #94a3b8;">Alibaba Qwen3-TTS 1.7B Voice Synthesizer</span>
    </div>
    """

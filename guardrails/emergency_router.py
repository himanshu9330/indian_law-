"""
LexAI Emergency Safety Router
Detects active, safety-critical crises and prepends supportive guidance with
relevant national helpline numbers dynamically retrieved from HELPLINES.
Does NOT block the user — ensures calm crisis routing followed by legal advice.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.config import HELPLINES
from core.llm import call_llm
from guardrails.logger import log_guardrail_event
from guardrails.pipeline import register_input_guardrail
from guardrails.types import GuardrailResult

logger = logging.getLogger("lexai.guardrails.emergency")

# ── Dynamic Helpline Lookup from core.config.HELPLINES ────────────────────────
def get_helplines_for_crisis(crisis_type: str) -> List[Dict[str, str]]:
    """
    Dynamically retrieve relevant helplines from HELPLINES without hardcoding numbers.
    """
    mapping = {
        "domestic_violence": ["Women Helpline", "Police Emergency", "NALSA Free Legal Aid"],
        "suicide": ["Tele-MANAS Mental Health Helpline", "Police Emergency"],
        "child_abuse": ["Child Helpline", "Police Emergency"],
        "arrest_detention": ["NALSA Free Legal Aid", "Police Emergency"],
        "sexual_assault": ["Women Helpline", "Police Emergency", "NALSA Free Legal Aid"],
        "immediate_danger": ["Police Emergency", "NALSA Free Legal Aid"],
    }
    target_names = mapping.get(crisis_type, ["Police Emergency", "NALSA Free Legal Aid"])
    selected = [h for h in HELPLINES if any(t.lower() in h["name"].lower() for t in target_names)]
    if not selected:
        selected = [h for h in HELPLINES if "police" in h["name"].lower()]
    return selected


# ── Active Emergency Patterns ────────────────────────────────────────────────
EMERGENCY_PATTERNS: Dict[str, List[str]] = {
    "domestic_violence": [
        r"(?i)\b(?:husband|partner|father|in-laws|wife|spouse|boyfriend|girlfriend|someone)\s+(?:is\s+)?(?:physically\s+)?(?:beating|hitting|strangling|attacking|abusing|hurting)\b",
        r"(?i)\b(?:being\s+beaten|locked\s+in|locked\s+the\s+door|domestic\s+violence\s+(?:right\s+now|at\s+this\s+moment))\b",
        r"(?i)\bhelp\s+me\s+my\s+(?:husband|spouse|in-laws|partner)\s+(?:is\s+hurting|is\s+beating|threatens\s+to\s+kill|is\s+physically\s+beating)\b",
    ],
    "suicide": [
        r"(?i)\b(?:want\s+to|going\s+to)\s+(?:end\s+my\s+life|kill\s+myself|commit\s+suicide)\b",
        r"(?i)\b(?:don't\s+want\s+to\s+live|feel\s+like\s+dying|ending\s+it\s+all|harm\s+myself)\b",
        r"(?i)\bthinking\s+of\s+suicide\b",
        r"(?i)\b(?:hopeless|depressed)\s+and\s+want\s+to\s+(?:end\s+my\s+life|die|kill\s+myself)\b",
    ],
    "child_abuse": [
        r"(?i)\b(?:child|baby|minor|kid|daughter|son)\s+is\s+being\s+(?:abused|beaten|tortured|trafficked|molested)\b",
        r"(?i)\b(?:abusing|hitting|beating|starving|harming)\s+(?:a|the|my|their)?\s*(?:\d+\s*(?:year|yr)s?[\s-]*(?:old\s+)?)?(?:child|kid|baby|minor|daughter|son)\b",
        r"(?i)\bchild\s+trafficking\s+in\s+progress\b",
    ],
    "arrest_detention": [
        r"(?i)\b(?:police|cops)\s+(?:have\s+barged|barged\s+in|are\s+arresting|have\s+arrested|detained|at\s+my\s+door|taking\s+away)\b",
        r"(?i)\bbeing\s+arrested\s+right\s+now\b",
        r"(?i)\bi\s+am\s+(?:in\s+police\s+custody|being\s+held\s+at\s+police\s+station|detained\s+illegally)\b",
        r"(?i)\barresting\s+me\s+(?:right\s+now|currently|without\s+(?:a\s+)?warrant)\b",
    ],
    "sexual_assault": [
        r"(?i)\b(?:just\s+been|was\s+just|currently\s+being)\s+(?:sexually\s+assaulted|raped|molested)\b",
        r"(?i)\bsomeone\s+(?:just\s+assaulted|sexually\s+assaulted|attempted\s+to\s+rape)\s+me\b",
    ],
    "immediate_danger": [
        r"(?i)\b(?:someone\s+is\s+trying\s+to\s+kill\s+me|life\s+is\s+in\s+danger\s+right\s+now|threatened\s+with\s+a\s+gun\s+right\s+now)\b",
    ],
}

# ── False-Positive Filter: Purely Educational / Past / Statutory Queries ──────
BENIGN_INQUIRY_PATTERNS: List[str] = [
    r"(?i)^\s*(?:what\s+is|what\s+are|explain|define|details\s+of|provisions\s+of|statutory\s+rules|punishment\s+for|penalty\s+for|how\s+does|how\s+is)\b",
    r"(?i)\b(?:under\s+the\s+act|under\s+bns|under\s+ipc|under\s+bnss|under\s+crpc|under\s+the\s+constitution|under\s+pocso|pocso\s+act)\b",
    r"(?i)\b(?:case\s+law|legal\s+definition|difference\s+between|historical\s+context|section\s+\d+)\b",
    r"(?i)^\s*can\s+police\s+arrest\s+someone\s*(?:on\s+a\s+sunday|at\s+night)?\b",
]


def detect_emergency(text: str) -> Optional[str]:
    """
    Evaluates text for active crisis. Returns emergency type or None.
    Avoids false positives on statutory/educational questions.
    """
    clean = text.strip()

    # Step 1: Check if this is an educational or general statutory query
    # E.g. "What is the punishment for murder under BNS?" must NOT trigger emergency
    has_urgent_first_person = bool(re.search(r"(?i)\b(?:right\s+now|at\s+this\s+moment|help\s+me|currently|hurting\s+me)\b", clean))
    for benign_pat in BENIGN_INQUIRY_PATTERNS:
        if re.search(benign_pat, clean) and not has_urgent_first_person:
            return None

    # Step 2: Pattern match against emergency categories
    for crisis_type, patterns in EMERGENCY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean):
                return crisis_type

    # Step 3: Fast LLM classifier for ambiguous inputs with high emotion
    if len(clean.split()) > 4 and any(w in clean.lower() for w in ["help", "save", "scared", "afraid", "danger", "police", "arrest"]):
        try:
            prompt = f"""You are an emergency crisis classifier for an Indian legal assistance bot.
Determine if the user is in an ACTIVE, IMMEDIATE PERSONAL CRISIS happening right now.
Categories:
- "domestic_violence" (immediate violence at home)
- "suicide" (self-harm, wanting to die)
- "child_abuse" (child in immediate harm)
- "arrest_detention" (police currently detaining or arresting someone right now)
- "sexual_assault" (recent sexual crime)
- "immediate_danger" (life threatened right now)
- "none" (general, educational, or theoretical questions, e.g. 'punishment for murder', 'can police arrest on Sunday')

Respond ONLY with valid JSON: {{"is_emergency": true/false, "emergency_type": "none" or category}}

USER QUERY:
{clean}
"""
            raw_eval = call_llm(
                prompt=prompt,
                system_prompt="You are a strict JSON classifier. Output only valid JSON.",
                temperature=0.0,
            )
            if raw_eval and raw_eval.strip():
                match = re.search(r"\{.*?\}", raw_eval.strip(), re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    if data.get("is_emergency") and data.get("emergency_type") in EMERGENCY_PATTERNS:
                        return data.get("emergency_type")
        except Exception as e:
            logger.info(f"Emergency LLM classifier safely skipped: {e}")

    return None


def build_emergency_banner(crisis_type: str) -> str:
    """
    Construct a calm, supportive message with numbers dynamically read from HELPLINES.
    """
    helplines = get_helplines_for_crisis(crisis_type)
    lines = []

    if crisis_type == "suicide":
        lines.append("> 🫂 **You Are Not Alone — Immediate Help is Available**")
        lines.append("> If you are experiencing overwhelming distress or thoughts of self-harm, please connect with trained counselors who can support you right now. Confidential, 24/7 help is available:")
    elif crisis_type in ("domestic_violence", "sexual_assault"):
        lines.append("> 🚨 **Immediate Safety & Crisis Support**")
        lines.append("> If you are in immediate danger or facing physical violence, please prioritize your safety. Contact official 24/7 emergency helplines right away:")
    elif crisis_type == "child_abuse":
        lines.append("> 🚨 **Child Protection & Emergency Intervention**")
        lines.append("> To report child abuse, distress, or imminent danger to a minor, official national emergency response is available 24/7:")
    elif crisis_type == "arrest_detention":
        lines.append("> ⚖️ **Urgent Legal Assistance & Arrest Protection**")
        lines.append("> If you or a family member is currently being detained or arrested, you have immediate legal protections under Articles 21 & 22 of the Constitution and Section 479/BNSS:")
    else:
        lines.append("> 🚨 **Immediate Emergency Assistance**")
        lines.append("> If you are in an emergency or facing imminent threat, please contact authorities immediately:")

    for h in helplines:
        lines.append(f"> - {h.get('icon', '📞')} **{h['name']}:** `{h['number']}` ({h.get('desc', 'Toll-free 24/7')})")

    lines.append("> *Your physical safety is paramount. Here is the relevant legal guidance for your situation:*")
    return "\n".join(lines)


# ── Guardrail Integration ────────────────────────────────────────────────────
def emergency_routing_guardrail(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Safety-critical emergency routing:
    - Detects domestic violence, suicide/self-harm, child abuse, active arrest, sexual assault.
    - Does NOT block. Sets banner in context and logs trigger to guardrail_events.
    """
    crisis_type = detect_emergency(text)
    if crisis_type:
        reason_code = f"emergency_{crisis_type}"
        banner = build_emergency_banner(crisis_type)

        if context is not None:
            context["emergency_type"] = crisis_type
            context["emergency_banner"] = banner

        return GuardrailResult.warn(
            reason_code=reason_code,
            user_message=banner,
            text=text,
        )

    return GuardrailResult.allow(text)

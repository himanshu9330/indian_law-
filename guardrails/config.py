"""
LexAI Guardrails Configuration
Centralized toggles, numerical thresholds, rate limits, regex patterns,
and refusal messages. All settings can be tuned here without modifying chat logic.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern


# ── Prompt Injection & Jailbreak Patterns ─────────────────────────────────────
INJECTION_PATTERNS: List[str] = [
    # Instruction overrides
    r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules|commands|directives)\b",
    r"(?i)\b(?:stop\s+being|you\s+are\s+no\s+longer)\s+(?:a\s+)?(?:legal\s+assistant|lexai)\b",
    r"(?i)\bnew\s+instructions?\s*:",
    r"(?i)\b(?:reset|clear)\s+(?:your\s+)?(?:system|context|instructions)\b",
    
    # System prompt extractions
    r"(?i)\b(?:reveal|show|print|display|tell\s+me|what\s+is)\s+(?:your\s+)?(?:(?:base|initial|original|core|system)\s+)?(?:system\s+prompt|prompt|instructions|system\s+instructions|base\s+prompt)\b",
    r"(?i)\boutput\s+(?:everything\s+)?above\b",
    
    # Jailbreak personas & modes
    r"(?i)\b(?:act\s+as\s+dan|dan\s+mode|jailbreak|do\s+anything\s+now|developer\s+mode|unrestricted\s+mode|god\s+mode)\b",
    r"(?i)\bpretend\s+(?:you\s+have\s+no\s+(?:rules|restrictions|filters)|you\s+can\s+do\s+anything)\b",
    r"(?i)\byou\s+are\s+now\s+(?:an?\s+unrestricted|free\s+from\s+all\s+rules)\b",
    
    # Role-override & tag attacks
    r"(?i)(?:<system>|\[system\]|###\s*system|\[inst\]|<inst>|<<sys>>)",
    r"(?i)(?:<!--\s*ignore.*?-->|<!\[cdata\[.*?ignore.*?\]\]>)",
]


# ── Scope Checking: Heuristic Indicators ─────────────────────────────────────
# Clear off-topic patterns (cooking, programming homework, pop culture/gossip, etc.)
OFF_TOPIC_PATTERNS: List[str] = [
    # Recipes & cooking
    r"(?i)\b(?:recipe\s+for|how\s+to\s+(?:cook|bake|fry|grill|make\s+a\s+cake)|ingredients\s+for|delicious\s+(?:curry|pasta|soup|pizza|biryani))\b",
    # Software development / coding homework
    r"(?i)\b(?:write\s+(?:a\s+)?(?:python|java|c\+\+|javascript|sql|html|react)\s+(?:code|script|program|app)|implement\s+(?:a\s+)?(?:quicksort|binary\s+search|linked\s+list|tree)|fix\s+this\s+(?:bug|syntax\s+error|nullpointerexception)|git\s+commit|docker-compose)\b",
    # Pop culture / celebrity gossip
    r"(?i)\b(?:celebrity\s+gossip|who\s+is\s+dating|box\s+office\s+collection|who\s+won\s+(?:ipl|world\s+cup|big\s+boss)|song\s+lyrics\s+for)\b",
    # School math / physics homework
    r"(?i)\b(?:calculate\s+the\s+integral|solve\s+(?:for\s+x|the\s+equation)|derivation\s+of\s+newton|thermodynamics\s+formula)\b",
]

# Legal anchor keywords that immediately validate legal context (checked with whole-word boundary)
LEGAL_KEYWORDS: List[str] = [
    "law", "act", "section", "ipc", "crpc", "bns", "bnss", "bsa", "cpc",
    "court", "police", "fir", "bail", "advocate", "lawyer", "judge", "magistrate",
    "arrest", "custody", "warrant", "summons", "charge", "chargesheet",
    "rights", "legal", "statute", "constitution", "article", "complaint",
    "consumer", "labour", "employment", "tenant", "rent", "landlord", "eviction",
    "contract", "agreement", "cheating", "fraud", "theft", "assault", "harassment",
    "posh", "dowry", "maintenance", "divorce", "custody", "will", "property",
    "tribunal", "high court", "supreme court", "pil", "writ", "jurisdiction",
]


# ── Harmful Intent Patterns vs Neutral Legal Explanation ─────────────────────
# Questions asking to learn or explain criminal law (MUST PASS)
BENIGN_LEGAL_EXPLANATION_PATTERNS: List[str] = [
    r"(?i)\b(?:explain|what\s+is|what\s+are|define|details\s+of)\s+(?:bns|ipc|crpc|bnss|bsa|section|act)\b",
    r"(?i)\bwhat\s+is\s+the\s+punishment\s+for\b",
    r"(?i)\bwhat\s+(?:are\s+the\s+penalties|penalty\s+for|charges\s+for)\b",
    r"(?i)\b(?:what\s+constitutes|definition\s+of|elements\s+of)\b",
    r"(?i)\bwhat\s+are\s+my\s+rights\b",
    r"(?i)\bis\s+(?:it|this)\s+(?:illegal|an?\s+offence|a\s+crime)\b",
    r"(?i)\bhow\s+is\s+(?:bail|anticipatory\s+bail|maintenance|custody)\s+(?:granted|decided|applied)\b",
    r"(?i)\bhow\s+to\s+(?:file|register)\s+(?:an?\s+fir|a\s+complaint|a\s+case|an\s+rti|a\s+petition)\b",
]

# Actionable intent to commit crime, evade law enforcement, forge, or destroy evidence (MUST BLOCK)
HARMFUL_INTENT_PATTERNS: List[str] = [
    # Committing crime / robbery / fraud / violence
    r"(?i)\bhow\s+(?:can\s+i|to|do\s+i)\s+(?:commit|execute|carry\s+out)\s+(?:murder|theft|robbery|fraud|extortion|rape|crime)\b",
    r"(?i)\bhow\s+(?:can\s+i|to|do\s+i)\s+(?:rob|burgle|loot|steal|defraud)\b",
    r"(?i)\b(?:how\s+to|help\s+me)\s+(?:poison|kill|murder|assault|harm|kidnap)\s+(?:someone|my|a\s+person|anybody)\b",
    r"(?i)\bhow\s+(?:can\s+i|to)\s+bribe\s+(?:a\s+)?(?:police|judge|officer|official|government)\b",
    r"(?i)\bhow\s+to\s+launder\s+(?:black\s+)?money\b",
    
    # Evading police, arrest, or investigation
    r"(?i)\bhow\s+(?:can\s+i|to)\s+(?:evade|escape|flee|hide\s+from)\s+(?:police|arrest|investigation|custody|warrant)\b",
    r"(?i)\bhow\s+to\s+avoid\s+getting\s+caught\s+by\s+police\b",
    r"(?i)\bhow\s+to\s+hide\s+(?:my\s+location|from\s+police\s+tracking)\b",
    
    # Evidence destruction / tampering
    r"(?i)\bhow\s+(?:can\s+i|to|do\s+i)\s+(?:destroy|tamper\s+with|hide|wipe|dispose\s+of)\b.*?\b(?:evidence|weapon|murder\s+weapon|fingerprints|cbi\s+data|records|documents)\b",
    r"(?i)\bhow\s+to\s+make\s+a\s+(?:body|weapon|evidence)\s+disappear\b",
    
    # Forgery & counterfeit
    r"(?i)\bhow\s+(?:can\s+i|to|do\s+i)\s+(?:forge|fake|falsify|counterfeit)\b",
    r"(?i)\b(?:how\s+to|help\s+me)\s+forge\b",
    
    # Threat, extortion, harassment
    r"(?i)\bhow\s+(?:can\s+i|to)\s+(?:blackmail|extort|threaten|harass)\s+(?:someone|my|a\s+person)\b",
]


# ── PII Regex Patterns (Aadhaar, PAN, Phone, Email, Bank A/C) ─────────────────
# 1. Aadhaar: 12 digits (first digit 2-9, optional spaces or dashes)
AADHAAR_REGEX: str = r"\b[2-9]\d{3}[\s-]?(?:\d{4})[\s-]?(?:\d{4})\b"

# 2. PAN: 5 uppercase letters, 4 digits, 1 uppercase letter
PAN_REGEX: str = r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]{1}\b"

# 3. Indian Phone: 10 digits starting with 6, 7, 8, or 9 (with optional +91, 91, or 0)
PHONE_REGEX: str = r"(?:\+91[\s-]?)?(?:\b0)?[6-9]\d{4}[\s-]?\d{5}\b"

# 4. Email Address: standard email format
EMAIL_REGEX: str = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

# 5. Bank Account Number: 9 to 18 digits preceded by account context keywords
BANK_ACC_REGEX: str = r"(?i)\b(?:a/c|account|acc|bank\s*a/c)(?:\s*(?:no|number|#))?[\s:]*([0-9]{9,18})\b"


# ── Overconfidence & Guarantee Patterns (Output Guardrail) ────────────────────
OVERCONFIDENCE_PATTERNS: List[str] = [
    r"(?i)\byou\s+will\s+(?:definitely|certainly|surely)\s+win\b",
    r"(?i)\bthis\s+is\s+100%\s+legal\b",
    r"(?i)\b(?:100%\s+guaranteed|guaranteed\s+to\s+win|cannot\s+lose\s+this\s+case)\b",
    r"(?i)\bthe\s+court\s+will\s+definitely\s+(?:rule\s+in\s+your\s+favor|acquit\s+you|grant\s+bail)\b",
    r"(?i)\byou\s+are\s+guaranteed\s+an?\s+acquittal\b",
]

# ── System Prompt Leak Patterns (Output Guardrail) ───────────────────────────
PROMPT_LEAK_PATTERNS: List[str] = [
    r"(?i)\byou\s+are\s+(?:a\s+strict\s+indian\s+legal\s+assistant|lexai,\s+a\s+dedicated\s+indian\s+legal\s+information\s+assistant)\b",
    r"(?i)\bCRITICAL\s+RULES\s+\(MUST\s+FOLLOW\)\b",
    r"(?i)\bHARDENED\s+ETHICAL\s+&\s+SAFETY\s+PRINCIPLES\b",
    r"(?i)\b###\s*Disclaimer\s*\n\s*Write\s+a\s+standard\s+legal\s+disclaimer\b",
    r"(?i)\bState\s+one\s+of:\s*\*\*Required\*\*\s*/\s*\*\*Optional\*\*\s*/\s*\*\*Not\s+Needed\*\*\b",
    r"(?i)\bProvide\s+a\s+clear,\s+\{detail_level\}\s+explanation\b",
]

# ── Foundational Valid Indian Statutes & Sections ───────────────────────────
# Canonical valid references recognized even if not in a specific RAG chunk
KNOWN_VALID_STATUTES: Dict[str, List[str]] = {
    "bns": ["103", "106", "115", "303", "305", "316", "318", "356", "75", "64", "65", "69", "111", "187"],
    "ipc": ["302", "304", "304a", "307", "376", "378", "379", "405", "406", "420", "498a", "499", "500", "506", "323", "324", "34", "120b"],
    "bnss": ["35", "173", "479", "482", "187", "126"],
    "crpc": ["41", "154", "436a", "437", "438", "482", "144", "125"],
    "constitution": ["14", "19", "20", "21", "22", "32", "226", "300a", "39a"],
    "pocso": ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    "mha": ["115"],
}


@dataclass
class GuardrailsConfig:
    # Master switches
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True

    # Layer switches (Input)
    enable_validation_check: bool = True
    enable_injection_check: bool = True
    enable_scope_check: bool = True
    enable_harmful_intent_check: bool = True
    enable_pii_masking: bool = True

    # Layer switches (Output)
    enable_prompt_leak_check: bool = True
    enable_pii_output_masking: bool = True
    enable_overconfidence_check: bool = True
    enable_citation_grounding_check: bool = True
    enable_retrieval_confidence_check: bool = True
    enable_output_disclaimer: bool = True

    # Rate Limiting & Abuse Protection
    enable_rate_limiting: bool = True
    max_messages_per_minute: int = 10
    max_messages_per_day: int = 100
    max_messages_per_conversation: int = 50
    repeated_block_threshold: int = 5
    repeated_block_window_seconds: int = 600  # 10 minutes
    block_cooldown_seconds: int = 60  # 1 minute
    max_context_messages: int = 10  # recent N messages sent to LLM context

    # Numeric constraints & thresholds
    max_message_length: int = 4000
    min_message_length: int = 1
    rag_confidence_threshold: float = 0.5

    # Pattern lists
    injection_patterns: List[str] = field(default_factory=lambda: INJECTION_PATTERNS)
    off_topic_patterns: List[str] = field(default_factory=lambda: OFF_TOPIC_PATTERNS)
    harmful_intent_patterns: List[str] = field(default_factory=lambda: HARMFUL_INTENT_PATTERNS)
    benign_legal_patterns: List[str] = field(default_factory=lambda: BENIGN_LEGAL_EXPLANATION_PATTERNS)
    overconfidence_patterns: List[str] = field(default_factory=lambda: OVERCONFIDENCE_PATTERNS)
    prompt_leak_patterns: List[str] = field(default_factory=lambda: PROMPT_LEAK_PATTERNS)

    # Output messages & disclaimers
    output_disclaimer_text: str = "*This is general legal information, not a substitute for advice from a qualified advocate.*"
    safe_prompt_leak_message: str = (
        "I am LexAI, an Indian legal information assistant. I am here to help you understand Indian laws, "
        "citizen rights, and court procedures. Please feel free to ask your legal question directly."
    )
    unverified_citation_note: str = "Please verify this section reference"
    low_confidence_note: str = "> ℹ️ *Note: This answer is based on general legal knowledge and should be verified.*"

    # Refusals
    refusals: Dict[str, str] = field(default_factory=lambda: {
        "empty_input": "Your message cannot be empty. Please enter your legal question.",
        "message_too_long": (
            "Your inquiry exceeds the maximum allowed length (4,000 characters). "
            "Please summarize your question and submit again."
        ),
        "prompt_injection": (
            "I cannot process this request because it contains instruction overrides or prompt injection patterns. "
            "Please state your Indian legal question directly."
        ),
        "out_of_scope": (
            "LexAI is a dedicated Indian legal assistant. I can only assist with questions regarding Indian laws, "
            "statutory codes (such as BNS, IPC, CrPC, BNSS), citizen rights, legal notices, and court procedures. "
            "Please feel free to ask any legal question."
        ),
        "crime_intent": (
            "I cannot provide instructions, assistance, or strategies for committing crimes, evading police, "
            "destroying evidence, or forging documents. "
            "If you are facing a legal dispute or criminal allegation, the lawful course of action is to "
            "consult an advocate, apply for legal remedies (such as anticipatory bail under Section 482 of BNSS), "
            "or seek free legal aid through NALSA (Toll-Free 15100)."
        ),
        "crisis_detected": (
            "If you or someone you know is in immediate danger or facing violence, please contact official 24/7 helplines right away:\n\n"
            "- 🚔 **Police Emergency:** `112`\n"
            "- 👩 **Women Helpline:** `181`\n"
            "- 👶 **Childline:** `1098`\n"
            "- 🧠 **Mental Health / Tele-MANAS:** `14416` / `1800-891-4416`\n"
            "- ⚖️ **NALSA Free Legal Aid:** `15100`"
        ),
        "default_refusal": (
            "I cannot fulfill this request as it does not comply with our legal advisory safety guidelines."
        ),
        "rate_limit_minute": "You're sending messages too fast, please wait a moment before sending another question.",
        "rate_limit_day": "You have reached your daily consultation limit. Please try again tomorrow.",
        "rate_limit_conversation": "This consultation has reached its maximum message limit. Please start a new consultation.",
        "repeated_blocks_cooldown": "Multiple requests were flagged by safety guardrails in a short period. Please wait 1 minute before trying again.",
    })

    pii_notice: str = (
        "> ℹ️ *Note: Sensitive personal details (Aadhaar, PAN, phone number, email, or bank account) in your message "
        "were automatically masked for your privacy before processing.*"
    )

    def get_refusal(self, reason_code: str) -> str:
        """Fetch refusal message by reason code with safe fallback."""
        return self.refusals.get(reason_code, self.refusals["default_refusal"])


# Global singleton instance
GUARDRAILS_CONFIG = GuardrailsConfig()

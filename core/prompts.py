"""
All LLM prompt templates used in the LexAI pipeline.
Each prompt uses {placeholders} for str.format() substitution.
"""

# ── 1. Query Refinement ──────────────────────────────────────────────────
QUERY_REFINER = """You are an Indian Legal Research Expert.
Convert the user SITUATION into a precise legal search query.

CRITICAL RULES:
- The query MUST be focused on the Indian Legal System.
- Append keywords like "Indian Law", "IPC", "BNS", or "High Court of India" if applicable.
- Do NOT include any US-specific terminology (like "Miranda" or "Amendment 5").

USER SITUATION:
{query}

OUTPUT:
Return ONLY the refined legal search query string — no explanation, no preamble."""


# ── 2. Risk Classification ───────────────────────────────────────────────
RISK_CLASSIFIER = """You are a Legal Risk Assessment Expert.
Analyze the following user situation and classify the risk level.

RISK DEFINITIONS:
- LOW: General informational queries about laws, basic rights, or minor civil matters with no immediate threat.
- MEDIUM: Legal notices received, contract disputes, non-violent civil disagreements, or situations where a minor offence may have occurred but there is no immediate danger.
- HIGH: Immediate threat of arrest, physical violence, serious criminal charges (e.g. assault, theft, harassment), or urgent legal deadlines that could lead to loss of liberty or safety.

USER QUERY:
{query}

Respond in EXACTLY this format (two lines only):
RISK: LOW or MEDIUM or HIGH
REASON: One sentence explaining why."""


# ── 3. Legal Expert (Structured Output) ──────────────────────────────────
LEGAL_EXPERT = """You are a STRICT Indian legal assistant.
Your job is to provide ONLY India-specific legal guidance.

═══════════════════════════════════════════════
CRITICAL RULES (MUST FOLLOW):

1. ONLY use Indian laws:
   - Indian Penal Code (IPC) / Bharatiya Nyaya Sanhita (BNS)
   - Code of Criminal Procedure (CrPC) / Bharatiya Nagarik Suraksha Sanhita (BNSS)
   - Constitution of India
   - Labour laws, Consumer Protection Act, POSH Act, etc.

2. NEVER mention foreign laws (US, UK, etc.)

3. If unsure about the correct Indian law — give general guidance WITHOUT naming wrong laws.

4. DO NOT hallucinate sections or mix unrelated laws.
═══════════════════════════════════════════════

CONTEXT (Retrieved Legal Documents / Web Info):
{context}

USER QUERY:
{query}

RISK LEVEL: {risk}

═══════════════════════════════════════════════
{style_instruction}
═══════════════════════════════════════════════

You MUST structure your response using EXACTLY these section headers (use ### markdown headers).
Every section is REQUIRED. Do not skip any.

### Disclaimer
Write a standard legal disclaimer that this is general legal information, not legal advice.

### Lawyer Needed
State one of: **Required** / **Optional** / **Not Needed**
Then on a new line explain why in 1-2 sentences.

### Legal Summary
Provide a clear, {detail_level} explanation of the law relevant to this query.

### Relevant Laws
List the specific Indian law sections as bullet points in this format:
- **Section X of [Act Name]**: Brief description of what it covers

If you are not sure of the exact section, skip this section or say "Consult a lawyer for the specific applicable sections."

### Action Roadmap
Provide a numbered step-by-step action plan:
1. First step
2. Second step
(etc.)

### Important Rights
List the user's key rights as bullet points:
- Right 1
- Right 2

### Final Note
A brief closing note with encouragement and a reminder to consult a lawyer for serious matters."""


# ── 4. Roadmap Generator ─────────────────────────────────────────────────
ROADMAP_GENERATOR = """Based on the following legal context and the user's situation, generate a clear, step-by-step legal roadmap.

User Query: {query}

Legal Context:
{context}

Instructions:
- Provide a numbered list of actions (Step 1, Step 2, etc.).
- Keep it practical (e.g., mention filing FIRs, court procedures, or seeking specific counsel).
- Use a professional yet supportive tone.
- If the context is missing specific details, give general best-practice advice for this legal domain.
- Each step should be one concise sentence."""


# ── 5. Chat Title Generator ──────────────────────────────────────────────
TITLE_GENERATOR = """Generate a short, descriptive title (3-6 words) for a legal chat conversation that starts with this query.
The title should capture the legal topic. Do NOT use quotes.

Query: {query}

Title:"""


# ── 6. Follow-up Suggestions ─────────────────────────────────────────────
FOLLOWUP_GENERATOR = """Based on this legal Q&A, suggest exactly 3 natural follow-up questions the user might ask next.
Each question should explore a different angle (procedure, rights, consequences, exceptions, etc.).
Keep them concise (under 15 words each). Output ONLY the 3 questions, one per line, numbered 1-3.

User asked: {query}

Summary of answer: {summary}

Follow-up questions:"""


# ── 7. General (Non-Law) Response ─────────────────────────────────────────
GENERAL_RESPONSE = """You are a helpful AI assistant. Answer the following general question clearly and concisely.
If the user seems to have a legal question, gently suggest they rephrase it with legal context.

Question: {query}

Answer:"""


# ── 8. Document Summary ──────────────────────────────────────────────────
DOCUMENT_SUMMARY = """You are a legal document analyst specializing in Indian law.
Analyze the following document text and provide:

1. **Document Type**: What kind of legal document is this? (e.g., rent agreement, legal notice, FIR copy, court order, etc.)
2. **Key Parties**: Who are the parties involved?
3. **Summary**: A plain-language summary of the document (3-5 sentences).
4. **Important Clauses**: List any clauses that are particularly important or could be risky for the reader.
5. **Red Flags**: Highlight any concerning or unusual terms that the reader should be aware of.
6. **Recommendations**: What should the reader do next?

DOCUMENT TEXT:
{document_text}

Provide your analysis in a clear, structured format using the headers above."""


# ── 9. Legal Draft Templates ─────────────────────────────────────────────
DRAFT_TEMPLATES: dict[str, dict] = {
    "RTI Application": {
        "description": "Right to Information application to a public authority",
        "fields": ["applicant_name", "address", "authority_name", "information_sought", "period"],
        "field_labels": {
            "applicant_name": "Your Full Name",
            "address": "Your Full Address",
            "authority_name": "Name of Public Authority / Department",
            "information_sought": "Information you are seeking (be specific)",
            "period": "Time period of information (e.g., 2020-2024)",
        },
        "prompt": """Draft a formal RTI Application under the Right to Information Act, 2005.

Applicant: {applicant_name}
Address: {address}
To: Public Information Officer, {authority_name}
Information Sought: {information_sought}
Period: {period}

Format it as a proper formal application with:
- Subject line
- Reference to RTI Act, 2005
- Clear numbered points of information sought
- Fee mention (₹10 postal order / demand draft)
- Declaration that the applicant is an Indian citizen
- Date and signature placeholders"""
    },
    "Police Complaint": {
        "description": "Written complaint to the police station",
        "fields": ["complainant_name", "address", "police_station", "incident_details", "date_of_incident", "accused_details"],
        "field_labels": {
            "complainant_name": "Your Full Name",
            "address": "Your Address",
            "police_station": "Name of Police Station",
            "incident_details": "Describe the incident in detail",
            "date_of_incident": "Date and time of incident",
            "accused_details": "Details of the accused (if known)",
        },
        "prompt": """Draft a formal written complaint to be submitted at a police station for FIR registration.

Complainant: {complainant_name}
Address: {address}
Police Station: {police_station}
Date of Incident: {date_of_incident}
Incident Details: {incident_details}
Accused Details: {accused_details}

Format as a proper police complaint with:
- Subject line mentioning relevant offences
- Factual narration of the incident
- Request to register FIR under appropriate sections
- Prayer clause
- Date and signature placeholders"""
    },
    "Consumer Complaint": {
        "description": "Complaint to Consumer Forum / Commission",
        "fields": ["complainant_name", "address", "opposite_party", "product_service", "complaint_details", "relief_sought", "amount"],
        "field_labels": {
            "complainant_name": "Your Full Name",
            "address": "Your Full Address",
            "opposite_party": "Company / Seller name and address",
            "product_service": "Product or service involved",
            "complaint_details": "Describe your complaint in detail",
            "relief_sought": "What relief / action do you want?",
            "amount": "Amount involved (₹)",
        },
        "prompt": """Draft a formal Consumer Complaint under the Consumer Protection Act, 2019.

Complainant: {complainant_name}
Address: {address}
Opposite Party: {opposite_party}
Product/Service: {product_service}
Complaint: {complaint_details}
Relief Sought: {relief_sought}
Amount: ₹{amount}

Format as a proper consumer complaint with:
- Case title (Complainant vs Opposite Party)
- Jurisdiction mention based on amount
- Factual narration with dates
- Deficiency in service / defective goods description
- Prayer clause with specific relief
- Verification and signature placeholders"""
    },
    "Legal Notice": {
        "description": "Formal legal notice before filing a case",
        "fields": ["sender_name", "sender_address", "recipient_name", "recipient_address", "subject", "facts", "demand", "deadline_days"],
        "field_labels": {
            "sender_name": "Your Full Name",
            "sender_address": "Your Address",
            "recipient_name": "Recipient's Name",
            "recipient_address": "Recipient's Address",
            "subject": "Subject of the notice",
            "facts": "Facts and background of the dispute",
            "demand": "What you are demanding",
            "deadline_days": "Response deadline (in days, e.g., 15)",
        },
        "prompt": """Draft a formal Legal Notice under Section 80 of CPC (or as appropriate).

From: {sender_name}, {sender_address}
To: {recipient_name}, {recipient_address}
Subject: {subject}
Facts: {facts}
Demand: {demand}
Deadline: {deadline_days} days

Format as a proper legal notice with:
- "LEGAL NOTICE" header
- Reference number and date
- "Without prejudice" header
- Clear factual narration
- Legal basis and applicable laws
- Specific demand with deadline
- Consequences of non-compliance
- Note: "This notice is sent through Registered Post AD / Email"
- Advocate / sender signature placeholder"""
    },
}


# ── Style instructions for answer detail level ────────────────────────────
STYLE_SIMPLE = "Explain everything in simple, everyday language. Avoid legal jargon. Use short sentences. Imagine you are explaining to someone with no legal background."
STYLE_DETAILED = "Provide a thorough, detailed legal analysis. Include specific law sections, precedents where possible, and nuanced explanations suitable for someone with some legal understanding."

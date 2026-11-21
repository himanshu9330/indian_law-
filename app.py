"""
LexAI — Premier AI Legal Assistant for Indian Law
Production-Grade Futuristic SaaS Interface
Rebuilt from scratch with Dark Glassmorphic Design System,
Comprehensive Telemetry Dashboard, and Multi-Stage RAG Pipeline.
"""

import os
import sys
from pathlib import Path

# Ensure application directory is in Python path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="LexAI — Bharat Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    PINECONE_API_KEY,
    QWEN_TTS_MODEL,
    RAG_THRESHOLD,
    RAG_TOP_K,
    SERPAPI_API_KEY,
    save_env_key,
)
from core.llm import call_llm, generate_chat_title
from core.pipeline import run_query
from core.prompts import DRAFT_TEMPLATES, DOCUMENT_SUMMARY
from db.database import Database
from ui.analytics import render_analytics_page
from ui.dashboard import render_dashboard_page
from ui.components import (
    load_css,
    render_bns_ipc_matrix,
    render_glossary_view,
    render_helplines_card,
    render_hero,
    render_structured_answer,
    render_top_navbar,
)
from utils.export import export_markdown, export_pdf
from utils.helpers import (
    format_timestamp,
    get_width_kw,
    group_chats_by_date,
    sanitize_input,
)


# ── Database Initialization ───────────────────────────────────────────────
@st.cache_resource
def get_db() -> Database:
    return Database()


db = get_db()


# ── Session State Setup ───────────────────────────────────────────────────
def init_session_state() -> None:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    if "user" not in st.session_state:
        st.session_state["user"] = None  # None = Guest Mode
    if "current_chat_id" not in st.session_state:
        chats = db.get_chats(user_id=None)
        if chats:
            st.session_state["current_chat_id"] = chats[0]["id"]
        else:
            new_id = db.create_chat(user_id=None, title="New Consultation")
            st.session_state["current_chat_id"] = new_id
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "🚀 Dashboard Overview"
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None
    if "answer_style" not in st.session_state:
        st.session_state["answer_style"] = "detailed"
    if "rag_threshold" not in st.session_state:
        st.session_state["rag_threshold"] = RAG_THRESHOLD
    if "rag_top_k" not in st.session_state:
        st.session_state["rag_top_k"] = RAG_TOP_K
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = GROQ_MODEL
    if "groq_key" not in st.session_state:
        st.session_state["groq_key"] = GROQ_API_KEY
    if "pinecone_key" not in st.session_state:
        st.session_state["pinecone_key"] = PINECONE_API_KEY
    if "serpapi_key" not in st.session_state:
        st.session_state["serpapi_key"] = SERPAPI_API_KEY
    if "openrouter_key" not in st.session_state:
        st.session_state["openrouter_key"] = OPENROUTER_API_KEY
    if "openrouter_model" not in st.session_state:
        st.session_state["openrouter_model"] = OPENROUTER_MODEL
    if "qwen_tts_model" not in st.session_state:
        st.session_state["qwen_tts_model"] = QWEN_TTS_MODEL
    if "hf_key" not in st.session_state:
        st.session_state["hf_key"] = HUGGINGFACE_API_KEY
    if "hf_model" not in st.session_state:
        st.session_state["hf_model"] = HUGGINGFACE_MODEL


init_session_state()
load_css()


# ── Sidebar Navigation & Consultation History ────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # SaaS Brand Card
        st.markdown(
            """
            <div class="sidebar-brand-card">
                <div class="brand-icon-box">⚖️</div>
                <div>
                    <h2 class="brand-title">LexAI</h2>
                    <p class="brand-subtitle">Bharat Legal SaaS</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user = st.session_state.get("user")
        user_id = user["id"] if user else None

        # User Authentication status
        with st.expander(f"👤 {user['username'] if user else 'Guest Session'}", expanded=False):
            if user:
                st.caption(f"Signed in as **{user['username']}**")
                if st.button("Log Out", key="logout_btn", **get_width_kw()):
                    st.session_state["user"] = None
                    st.rerun()
            else:
                auth_tab1, auth_tab2 = st.tabs(["Sign In", "Register"])
                with auth_tab1:
                    l_user = st.text_input("Username", key="login_u")
                    l_pass = st.text_input("Password", type="password", key="login_p")
                    if st.button("Sign In", key="login_btn", **get_width_kw()):
                        u_record = db.authenticate_user(l_user, l_pass)
                        if u_record:
                            st.session_state["user"] = u_record
                            st.toast(f"Welcome back, {l_user}!", icon="👋")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                with auth_tab2:
                    r_user = st.text_input("New Username", key="reg_u")
                    r_pass = st.text_input("New Password", type="password", key="reg_p")
                    if st.button("Create Account", key="reg_btn", **get_width_kw()):
                        if r_user and r_pass:
                            try:
                                uid = db.create_user(r_user, r_pass)
                                st.session_state["user"] = {"id": uid, "username": r_user}
                                st.success("Account registered!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Registration error: {e}")

        # New Consultation Action
        if st.button("➕ New Consultation", key="new_chat_btn", type="primary", **get_width_kw()):
            new_id = db.create_chat(user_id=user_id, title="New Consultation")
            st.session_state["current_chat_id"] = new_id
            st.session_state["nav_page"] = "💬 Legal Chat"
            st.rerun()

        # Primary Navigation
        nav_options = [
            "🚀 Dashboard Overview",
            "💬 Legal Chat",
            "⚖️ BNS ⇄ IPC Matrix",
            "📝 Legal Draft Studio",
            "📄 Document Analyzer",
            "🔖 Saved Case Files",
            "📊 Analytics Telemetry",
            "📖 Statutory Glossary",
            "🚨 Emergency Helplines",
            "⚙️ Settings & Vault",
        ]

        current_idx = nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0
        st.session_state["nav_page"] = st.radio(
            "Navigation",
            options=nav_options,
            index=current_idx,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

        # Past Conversations Drawer
        st.markdown(
            """
            <div class="chat-history-header">
                <span>📁 Past Case Files</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        search_query = st.text_input("Search chats...", key="chat_search", placeholder="Filter topics...", label_visibility="collapsed")
        all_chats = db.search_chats(search_query, user_id=user_id) if search_query else db.get_chats(user_id=user_id)

        grouped = group_chats_by_date(all_chats)
        current_cid = st.session_state.get("current_chat_id")

        for group_label, chats_in_group in grouped.items():
            st.caption(f"**{group_label}**")
            for c in chats_in_group:
                cid = c["id"]
                title = c.get("title") or "Consultation"
                is_active = (cid == current_cid)

                col_btn, col_pin, col_del = st.columns([0.72, 0.14, 0.14])
                with col_btn:
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(
                        f"{'📌 ' if c.get('pinned') else ''}{title[:20]}",
                        key=f"chat_select_{cid}",
                        type=btn_type,
                        **get_width_kw(),
                    ):
                        st.session_state["current_chat_id"] = cid
                        st.session_state["nav_page"] = "💬 Legal Chat"
                        st.rerun()

                with col_pin:
                    pinned = bool(c.get("pinned", 0))
                    if st.button("📍" if pinned else "📌", key=f"pin_{cid}", help="Pin or unpin consultation"):
                        db.pin_chat(cid, not pinned)
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"del_{cid}", help="Delete consultation"):
                        db.delete_chat(cid)
                        if st.session_state["current_chat_id"] == cid:
                            remaining = db.get_chats(user_id=user_id)
                            st.session_state["current_chat_id"] = remaining[0]["id"] if remaining else None
                        st.rerun()

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

        # Privacy Guarantee Card
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.03); padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div class="radar-dot"></div>
                    <div style="font-size: 0.74rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px;">
                        Data Sovereignty
                    </div>
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 6px; line-height: 1.45;">
                    Private local SQLite persistence. Zero telemetry or external sharing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


render_sidebar()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 💬 LEGAL CHAT
# ═══════════════════════════════════════════════════════════════════════════
def render_chat_page() -> None:
    current_cid = st.session_state.get("current_chat_id")
    if not current_cid:
        current_cid = db.create_chat(
            user_id=st.session_state["user"]["id"] if st.session_state.get("user") else None,
            title="New Consultation",
        )
        st.session_state["current_chat_id"] = current_cid

    chat_info = db.get_chat(current_cid)
    chat_title = chat_info.get("title", "Legal Consultation") if chat_info else "Legal Consultation"

    # Executive Top Navbar Bar
    top_col1, top_col2, top_col3, top_col4 = st.columns([0.48, 0.20, 0.16, 0.16])
    with top_col1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.35rem; font-weight: 900; color: #ffffff;">⚖️ {chat_title}</span>
                <span class="live-status-pill"><span class="radar-dot"></span> LIVE DAG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col2:
        st.session_state["answer_style"] = st.selectbox(
            "Answer Style",
            options=["detailed", "simple"],
            format_func=lambda x: "Detailed Legal" if x == "detailed" else "Simple (ELI5)",
            index=0 if st.session_state["answer_style"] == "detailed" else 1,
            label_visibility="collapsed",
        )

    with top_col3:
        with st.popover("🚨 SOS Helpline"):
            st.markdown("#### 🚨 24/7 National Crisis Numbers")
            st.markdown("- **Police Emergency**: `112`")
            st.markdown("- **Women in Distress**: `181`")
            st.markdown("- **Cyber Financial Fraud**: `1930`")
            st.markdown("- **Free Legal Aid (NALSA)**: `15100`")
            st.markdown("- **National Consumer Helpline**: `1915`")

    with top_col4:
        with st.popover("⚙️ Case Actions"):
            st.markdown("##### Consultation Settings")
            new_title = st.text_input("Rename Consultation", value=chat_title)
            if st.button("Save Title", key="update_title_btn", **get_width_kw()):
                if new_title.strip():
                    db.update_chat_title(current_cid, new_title.strip())
                    st.rerun()

            st.markdown("---")
            curr_msgs = db.get_messages(current_cid)
            if curr_msgs:
                full_md = export_markdown(chat_title, curr_msgs)
                st.download_button(
                    "📄 Export as Markdown",
                    data=full_md,
                    file_name=f"{chat_title.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    **get_width_kw(fn=st.download_button),
                )
                full_pdf = export_pdf(chat_title, curr_msgs)
                st.download_button(
                    "📑 Export as PDF Report",
                    data=full_pdf,
                    file_name=f"{chat_title.lower().replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    **get_width_kw(fn=st.download_button),
                )

    messages = db.get_messages(current_cid)

    # Empty consultation state: Show executive hero with scenario chips
    if not messages:
        render_hero(on_select_query=lambda q: None)

    # Render conversation messages
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="⚖️"):
                render_structured_answer(
                    message=msg,
                    db=db,
                    on_followup=lambda q: None,
                )

    # Prompt input handling
    prompt_input = st.chat_input("Ask any Indian law question (e.g. tenant eviction, cyber fraud, anticipatory bail, BNS 2024)...")
    prompt_to_run = None

    if st.session_state.get("pending_query"):
        prompt_to_run = st.session_state["pending_query"]
        st.session_state["pending_query"] = None
    elif prompt_input:
        prompt_to_run = prompt_input

    if prompt_to_run:
        cleaned_prompt = sanitize_input(prompt_to_run)

        # 1. Add User Message
        with st.chat_message("user", avatar="👤"):
            st.markdown(cleaned_prompt)

        user_msg_id = db.add_message(
            chat_id=current_cid,
            role="user",
            content=cleaned_prompt,
        )

        # Auto-title on first message
        if len(messages) == 0:
            auto_title = generate_chat_title(cleaned_prompt, api_key=st.session_state.get("groq_key"))
            db.update_chat_title(current_cid, auto_title)

        # 2. Run Multi-Stage Pipeline DAG
        with st.chat_message("assistant", avatar="⚖️"):
            status_container = st.container()
            events_log = []

            with status_container:
                with st.status("🧠 Executing LexAI Multi-Stage Legal Reasoning Pipeline...", expanded=True) as status_box:
                    runner = run_query(
                        query=cleaned_prompt,
                        style=st.session_state.get("answer_style", "detailed"),
                        rag_threshold=st.session_state.get("rag_threshold", 0.5),
                        top_k=st.session_state.get("rag_top_k", 3),
                        groq_api_key=st.session_state.get("groq_key"),
                        pinecone_api_key=st.session_state.get("pinecone_key"),
                        serpapi_api_key=st.session_state.get("serpapi_key"),
                        openrouter_api_key=st.session_state.get("openrouter_key"),
                        openrouter_model=st.session_state.get("openrouter_model"),
                        hf_api_key=st.session_state.get("hf_key"),
                        hf_model=st.session_state.get("hf_model"),
                    )

                    final_result = None
                    for event in runner:
                        node = event.get("node")
                        events_log.append(event)
                        st.write(f"• **{event.get('title', node)}**: {event.get('message', '')}")
                        if node == "complete":
                            final_result = event.get("result")

                    status_box.update(label="🎉 Multi-Stage Legal Analysis Complete", state="complete", expanded=False)

            if final_result:
                db.add_message(
                    chat_id=current_cid,
                    role="assistant",
                    content=final_result.get("raw_answer", ""),
                    metadata=final_result,
                )
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 🔖 SAVED CASE FILES
# ═══════════════════════════════════════════════════════════════════════════
def render_saved_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>🔖 Bookmarked Case Files & Strategic Legal Opinions</h2>
            <p>Access all answers you have saved across your research consultations under Indian Law.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.get("user")
    bookmarks = db.get_bookmarks(user_id=user["id"] if user else None)

    if not bookmarks:
        st.info("No saved cases yet. Click '⭐ Bookmark' on any assistant response in the chat to file it here.")
        return

    for bm in bookmarks:
        with st.container(border=True):
            b_col1, b_col2 = st.columns([0.85, 0.15])
            with b_col1:
                st.markdown(f"#### ⚖️ Case File: *{bm.get('chat_title', 'Consultation')}*")
                st.caption(f"Filed on {format_timestamp(bm.get('created_at'))}")
            with b_col2:
                if st.button("🗑️ Remove", key=f"del_bm_{bm['id']}", **get_width_kw()):
                    db.remove_bookmark(bm["id"])
                    st.toast("Bookmark removed.")
                    st.rerun()

            msg_record = {
                "id": bm["message_id"],
                "content": bm.get("content", ""),
                "metadata": bm.get("metadata", {}),
            }
            render_structured_answer(msg_record, db=db)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 📝 LEGAL DRAFT STUDIO
# ═══════════════════════════════════════════════════════════════════════════
def render_drafts_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>📝 Indian Legal Document & Formal Notice Studio</h2>
            <p>Select a standard statutory template, enter facts, and generate a legally compliant formal draft.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_names = list(DRAFT_TEMPLATES.keys())
    selected_template_name = st.selectbox("Select Statutory Document Template", options=template_names)
    template = DRAFT_TEMPLATES[selected_template_name]

    st.info(f"ℹ️ **{selected_template_name}**: {template['description']}")

    field_values = {}
    with st.form(key="draft_form"):
        for field in template["fields"]:
            label = template["field_labels"].get(field, field.replace("_", " ").title())
            if "details" in field or "sought" in field or "facts" in field:
                field_values[field] = st.text_area(label, height=100)
            else:
                field_values[field] = st.text_input(label)

        submitted = st.form_submit_button("Generate Formal Legal Draft", type="primary", **get_width_kw())

    if submitted:
        if not any(field_values.values()):
            st.warning("Please enter required facts to draft the statutory notice.")
            return

        with st.spinner("⚖️ Drafting formal Indian legal application compliant with Bar Council rules..."):
            prompt = template["prompt"].format(**field_values)
            draft_text = call_llm(
                prompt=prompt,
                system_prompt="You are a legal document drafting expert enrolled with the Bar Council of India.",
                temperature=0.2,
                api_key=st.session_state.get("groq_key"),
            )

        st.success("Draft Generated Successfully!")
        st.markdown("### 📄 Formal Draft Document Preview")
        st.text_area("Draft Text", value=draft_text, height=360)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download as Text (.txt)",
                data=draft_text,
                file_name=f"{selected_template_name.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                **get_width_kw(fn=st.download_button),
            )
        with col_d2:
            st.download_button(
                label="📥 Download as Markdown (.md)",
                data=draft_text,
                file_name=f"{selected_template_name.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                **get_width_kw(fn=st.download_button),
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: 📄 DOCUMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════
def render_document_analyzer_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>📄 Contract & Legal Document Clause Analyzer</h2>
            <p>Upload a rental agreement, FIR copy, or legal notice to extract hidden liabilities, red flags, and citizen rights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload legal document (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        doc_text = ""
        try:
            if uploaded_file.name.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(uploaded_file)
                pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                doc_text = "\n".join(pages)
            else:
                doc_text = uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error parsing document: {e}")
            return

        if not doc_text.strip():
            st.warning("Could not extract readable text from this document.")
            return

        st.caption(f"Loaded {len(doc_text)} characters from {uploaded_file.name}")
        with st.expander("Preview Extracted Raw Text"):
            st.text(doc_text[:1200] + ("..." if len(doc_text) > 1200 else ""))

        if st.button("🔍 Analyze Clauses & Legal Red Flags", type="primary", **get_width_kw()):
            with st.spinner("Analyzing clauses under Indian contract and procedural law..."):
                prompt = DOCUMENT_SUMMARY.format(document_text=doc_text[:12000])
                analysis = call_llm(
                    prompt=prompt,
                    system_prompt="You are a senior Indian advocate reviewing documents for risk analysis.",
                    temperature=0.2,
                    api_key=st.session_state.get("groq_key"),
                )

            st.markdown("### 📋 Executive Document Risk Analysis Report")
            st.markdown(analysis)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ⚙️ SETTINGS & VAULT
# ═══════════════════════════════════════════════════════════════════════════
def render_settings_page() -> None:
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>⚙️ Platform Settings & API Credentials Vault</h2>
            <p>Customize inference models, retrieval thresholds, API keys, and local SQLite data retention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_model, tab_keys, tab_privacy = st.tabs(["🤖 LLM & Vector RAG", "🔑 API Keys Vault", "🛡️ Data Sovereignty"])

    with tab_model:
        st.markdown("##### Primary LLM & Pinecone Vector Search Parameters")
        model_options = [
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "mixtral-8x7b-32768",
        ]
        st.session_state["selected_model"] = st.selectbox(
            "Active Primary LLM Model (Groq)",
            options=model_options,
            index=model_options.index(st.session_state.get("selected_model", GROQ_MODEL)) if st.session_state.get("selected_model") in model_options else 0,
        )

        st.session_state["rag_threshold"] = st.slider(
            "Pinecone Retrieval Confidence Threshold",
            min_value=0.2,
            max_value=0.9,
            value=float(st.session_state.get("rag_threshold", 0.5)),
            step=0.05,
            help="If vector cosine similarity score falls below this value, the pipeline automatically routes to the Indian Law Fallback Model.",
        )

        st.session_state["rag_top_k"] = st.slider(
            "Knowledge Base Chunks (Top-K)",
            min_value=1,
            max_value=8,
            value=int(st.session_state.get("rag_top_k", 3)),
            help="Number of statutory chunks to retrieve from the nearest cluster.",
        )

        st.markdown("---")
        st.markdown("##### ⚡ Low-Confidence Indian Legal Fallback Model (OpenRouter / Qwen)")
        or_models = [
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-32b-instruct",
            "qwen/qwen-2.5-7b-instruct:free",
            "qwen/qwen3.8-27b",
        ]
        curr_or_model = st.session_state.get("openrouter_model", OPENROUTER_MODEL)
        st.session_state["openrouter_model"] = st.selectbox(
            "Fallback Indian Law Reasoning Model",
            options=or_models,
            index=or_models.index(curr_or_model) if curr_or_model in or_models else 0,
            help="Used whenever Pinecone is unavailable, missing, or confidence score falls below threshold.",
        )

        st.markdown("##### 🔊 Voice & Speech Generation Model (Qwen3-TTS)")
        st.session_state["qwen_tts_model"] = st.text_input(
            "Text-to-Speech Engine",
            value=st.session_state.get("qwen_tts_model", QWEN_TTS_MODEL),
            help="Alibaba Qwen3-TTS 1.7B parameter speech model for reading out legal summaries to citizens.",
        )

    with tab_keys:
        st.markdown("##### API Keys (Stored in memory and local .env)")
        st.caption("All credentials remain exclusively in your local environment. Zero external telemetry.")

        k_or = st.text_input("OpenRouter API Key (for Qwen Fallback & Reasoning)", value=st.session_state.get("openrouter_key", ""), type="password")
        k_hf = st.text_input("Hugging Face API Key / Token", value=st.session_state.get("hf_key", ""), type="password")
        k_groq = st.text_input("Groq API Key", value=st.session_state.get("groq_key", ""), type="password")
        k_pine = st.text_input("Pinecone API Key", value=st.session_state.get("pinecone_key", ""), type="password")
        k_serp = st.text_input("SerpAPI Key", value=st.session_state.get("serpapi_key", ""), type="password")

        if st.button("Save API Keys to Vault", **get_width_kw()):
            or_k = k_or.strip().strip("\"' ")
            hf_k = k_hf.strip().strip("\"' ")
            g_key = k_groq.strip().strip("\"' ")
            p_key = k_pine.strip().strip("\"' ")
            s_key = k_serp.strip().strip("\"' ")

            st.session_state["openrouter_key"] = or_k
            st.session_state["hf_key"] = hf_k
            st.session_state["groq_key"] = g_key
            st.session_state["pinecone_key"] = p_key
            st.session_state["serpapi_key"] = s_key

            save_env_key("OPENROUTER_API_KEY", or_k)
            save_env_key("HUGGINGFACE_API_KEY", hf_k)
            save_env_key("GROQ_API_KEY", g_key)
            save_env_key("PINECONE_API_KEY", p_key)
            save_env_key("SERPAPI_API_KEY", s_key)
            save_env_key("OPENROUTER_MODEL", st.session_state.get("openrouter_model", "qwen/qwen-2.5-72b-instruct"))
            save_env_key("QWEN_TTS_MODEL", st.session_state.get("qwen_tts_model", "qwen/qwen3-tts-1.7b"))

            st.success("API keys & model preferences updated and securely saved to local .env file!")

    with tab_privacy:
        st.markdown("##### Local SQLite Data Retention & Sovereignty")
        st.write(f"- **Database Path**: `{db.db_path}`")
        st.write(f"- **Current Storage Footprint**: `{db.get_db_size()}`")
        st.write("- **Telemetry Status**: Zero external analytics tracking. 100% private to this device.")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("⚠️ **Danger Zone**")
            st.caption("Permanently purge all consultation history, bookmarks, and user progress from the local SQLite database.")
            if st.button("🗑️ Delete All Local Data", type="primary", **get_width_kw()):
                user = st.session_state.get("user")
                db.clear_all_data(user_id=user["id"] if user else None)
                st.session_state["current_chat_id"] = None
                st.success("All conversations and local records have been purged.")
                st.rerun()


# ── Route to Active Navigation Page ──────────────────────────────────────
page = st.session_state.get("nav_page", "🚀 Dashboard Overview")

if page == "🚀 Dashboard Overview":
    render_dashboard_page(db=db, user_id=st.session_state["user"]["id"] if st.session_state.get("user") else None)
elif page == "💬 Legal Chat":
    render_chat_page()
elif page == "⚖️ BNS ⇄ IPC Matrix":
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>⚖️ Bharatiya Nyaya Sanhita (BNS) ⇄ IPC & BNSS ⇄ CrPC Matrix</h2>
            <p>Interactive statute conversion and transition guide for India's 2024 criminal law overhaul.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_bns_ipc_matrix(in_expander=False)
elif page == "📝 Legal Draft Studio":
    render_drafts_page()
elif page == "📄 Document Analyzer":
    render_document_analyzer_page()
elif page == "🔖 Saved Case Files":
    render_saved_page()
elif page == "📊 Analytics Telemetry":
    render_analytics_page(db=db, user_id=st.session_state["user"]["id"] if st.session_state.get("user") else None)
elif page == "📖 Statutory Glossary":
    render_glossary_view()
elif page == "🚨 Emergency Helplines":
    render_helplines_card()
elif page == "⚙️ Settings & Vault":
    render_settings_page()

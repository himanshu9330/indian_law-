"""
LexAI Legal Tools Pages
Clean, modular interfaces for Matrix, Draft Studio, Document Analyzer,
Statutory Glossary, Emergency Helplines, and Settings.
"""

import os
import streamlit as st
from core.prompts import DRAFT_TEMPLATES
from core.services import (
    get_bns_ipc_mappings,
    get_statutory_glossary,
    get_emergency_helplines,
    generate_legal_draft,
    analyze_legal_document,
)


def render_matrix_page():
    st.subheader("⚖️ BNS 2024 vs IPC 1860 & BNSS vs CrPC Matrix")
    st.caption("Cross-reference provisions between new 2024 criminal codes and legacy statutes.")

    search_term = st.text_input(
        "Search offence, section or topic...",
        placeholder="e.g. Cheating, Murder, FIR, Theft, 420, 302, 154, Defamation",
        key="tool_matrix_search",
    )

    items = get_bns_ipc_mappings(search_term)
    if not items:
        st.info("No matching sections found. Try a different term (e.g. 'cheating', 'arrest', '420').")
        return

    for item in items:
        with st.container(border=True):
            c1, c2 = st.columns([0.65, 0.35], vertical_alignment="center")
            with c1:
                st.markdown(f"**{item['topic']}**")
                st.caption(item["desc"])
                st.markdown(f"<span style='color: #10a37f; font-size: 0.8rem; font-weight: 600;'>Classification: {item['severity']}</span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"`{item['old_law']}` ➔ `{item['new_law']}`")


def render_draft_page():
    st.subheader("📝 Legal Draft Studio")
    st.caption("Generate formal notices, complaints, and RTI applications adhering to Indian legal standards.")

    template_names = list(DRAFT_TEMPLATES.keys())
    selected_template = st.selectbox("Select document template", template_names, key="tool_draft_type")
    template_cfg = DRAFT_TEMPLATES[selected_template]

    st.info(template_cfg.get("description", "Legal draft template"))

    form_data = {}
    with st.form(key=f"form_draft_{selected_template}"):
        for field in template_cfg.get("fields", []):
            label = template_cfg.get("field_labels", {}).get(field, field.replace("_", " ").title())
            if "details" in field or "facts" in field or "information" in field:
                form_data[field] = st.text_area(label, key=f"field_{field}")
            else:
                form_data[field] = st.text_input(label, key=f"field_{field}")

        submitted = st.form_submit_button("Generate Legal Draft", type="primary")

    if submitted:
        # Validate that required fields are filled
        empty_fields = [k for k, v in form_data.items() if not v.strip()]
        if empty_fields:
            st.warning("Please fill in all requested fields before generating the draft.")
        else:
            with st.spinner("Drafting legal document..."):
                try:
                    draft = generate_legal_draft(selected_template, form_data)
                    st.success("Draft generated successfully!")
                    st.text_area("Generated Document Draft", value=draft, height=350, key="draft_result_output")
                    st.download_button(
                        label="📥 Download Draft (.txt)",
                        data=draft,
                        file_name=f"{selected_template.lower().replace(' ', '_')}.txt",
                        mime="text/plain",
                        key="btn_download_draft",
                    )
                except Exception as e:
                    st.error(f"Draft generation failed: {e}")


def render_analyzer_page():
    st.subheader("📄 Legal Document Analyzer")
    st.caption("Analyze contracts, FIR copies, rent agreements, or legal notices for risks and obligations.")

    uploaded_file = st.file_uploader("Upload legal document (.txt)", type=["txt"], key="tool_doc_upload")
    doc_text = ""
    if uploaded_file is not None:
        try:
            doc_text = uploaded_file.read().decode("utf-8")
        except Exception:
            doc_text = ""

    pasted_text = st.text_area(
        "Or paste document text here:",
        value=doc_text,
        height=200,
        placeholder="Paste legal agreement or notice clauses...",
        key="tool_doc_paste",
    )

    if st.button("Analyze Document", type="primary", key="btn_run_analyzer"):
        text_to_analyze = pasted_text.strip()
        if not text_to_analyze:
            st.warning("Please provide or upload document text to analyze.")
        else:
            with st.spinner("Auditing clauses under Indian Law..."):
                try:
                    analysis = analyze_legal_document(text_to_analyze)
                    st.markdown("### Analysis Report")
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")


def render_glossary_page():
    st.subheader("📖 Indian Statutory & Legal Glossary")
    st.caption("Plain-language explanations of court terms, criminal procedure, and statutory phrases.")

    search_term = st.text_input(
        "Search legal terms or acronyms...",
        placeholder="e.g. FIR, Bail, PIL, Anticipatory, Caveat, BNS",
        key="tool_glossary_search",
    )

    glossary = get_statutory_glossary(search_term)
    if not glossary:
        st.info("No matching terms found. Try a different search.")
        return

    col1, col2 = st.columns(2)
    for idx, (term, definition) in enumerate(glossary.items()):
        col = col1 if idx % 2 == 0 else col2
        with col:
            with st.container(border=True):
                st.markdown(f"**⚖️ {term}**")
                st.caption(definition)


def render_helplines_page():
    st.subheader("🚨 Emergency Legal & Crisis Helplines (India)")
    st.caption("Official 24/7 national toll-free helplines for immediate legal assistance and crime reporting.")

    helplines = get_emergency_helplines()
    col1, col2 = st.columns(2)
    for idx, h in enumerate(helplines):
        col = col1 if idx % 2 == 0 else col2
        with col:
            with st.container(border=True):
                st.markdown(f"### {h.get('icon', '📞')} {h['name']}")
                st.markdown(f"**Toll-Free Dial:** `{h['number']}`")
                st.caption(h.get("desc", ""))


def render_settings_page():
    st.subheader("⚙️ Settings & System Information")
    st.caption("Overview of configured models, knowledge bases, and local storage.")

    with st.container(border=True):
        st.markdown("**LLM & Knowledge Engines**")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        pinecone_key = os.getenv("PINECONE_API_KEY", "")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("- **Primary LLM:** Google Gemini")
            st.markdown(f"- **Gemini Status:** {'✅ Configured' if gemini_key else '⚠️ Not set'}")
            st.markdown("- **Indian Law Fallback:** HuggingFace / Local Rule Engine (Active)")
        with c2:
            st.markdown("- **Vector RAG:** Pinecone (Indian Legal Corpus)")
            st.markdown(f"- **Pinecone Status:** {'✅ Configured' if pinecone_key else '⚠️ Offline / Fallback ready'}")
            st.markdown("- **Database:** SQLite (`data/lexai.db`)")

"""
UI Component library for LexAI.
Renders high-end structured legal answers, risk badges, interactive roadmaps,
starter chips, emergency helplines, pipeline progress, and feedback bars.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

from core.config import (
    BNS_IPC_MAPPING,
    HELPLINES,
    LEGAL_GLOSSARY,
    STARTER_QUESTIONS,
)
from utils.export import export_markdown, export_pdf
from utils.helpers import (
    format_confidence,
    get_width_kw,
    risk_description,
    risk_emoji,
)


def load_css() -> None:
    """Inject custom modern dark styles into Streamlit."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


def render_top_navbar(title: str, subtitle: str = "Indian Law Intelligence", right_component: Optional[Callable] = None) -> None:
    """Renders a sleek SaaS top navbar with breadcrumbs and live system indicator."""
    st.markdown(
        f"""
        <div class="lexai-top-navbar">
            <div class="top-nav-left">
                <div class="top-page-badge">
                    <span>{title}</span>
                </div>
                <div class="top-meta-crumb">
                    {subtitle}
                </div>
            </div>
            <div class="live-status-pill">
                <span class="radar-dot"></span>
                <span>GROQ INFERENCE ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bns_ipc_matrix(in_expander: bool = True) -> None:
    """Renders an interactive BNS 2024 vs IPC 1860 & BNSS vs CrPC conversion widget."""
    def _content():
        st.markdown("Search or cross-reference the new 2024 criminal codes with legacy Indian Penal Code & CrPC provisions:")
        filter_text = st.text_input(
            "🔍 Search offence, section, or keyword...",
            placeholder="e.g. Cheating, Murder, FIR, Theft, 420, 302, 154, Defamation",
            key="bns_search_field",
        )

        filtered = [
            item for item in BNS_IPC_MAPPING
            if not filter_text or (
                filter_text.lower() in item["topic"].lower()
                or filter_text.lower() in item["old_law"].lower()
                or filter_text.lower() in item["new_law"].lower()
                or filter_text.lower() in item["desc"].lower()
            )
        ]

        if not filtered:
            st.info("No matching sections found. Try a different term (e.g., 'cheating', 'arrest', '420').")
            return

        for item in filtered:
            st.markdown(
                f"""
                <div class="matrix-row-item">
                    <div style="flex: 1; padding-right: 18px;">
                        <div style="font-weight: 800; font-size: 1.05rem; color: #ffffff; margin-bottom: 4px;">{item['topic']}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">{item['desc']}</div>
                        <div style="font-size: 0.74rem; color: #34d399; font-weight: 700; margin-top: 6px;">• Classification: {item['severity']}</div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="matrix-law-old">{item['old_law']}</span>
                        <span class="matrix-arrow">➔</span>
                        <span class="matrix-law-new">{item['new_law']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if in_expander:
        with st.expander("⚖️ Quick Statute Cross-Reference: Bharatiya Nyaya Sanhita (BNS) ⇄ IPC / BNSS ⇄ CrPC", expanded=False):
            _content()
    else:
        _content()


def render_hero(on_select_query: Optional[Callable[[str], None]] = None) -> None:
    """Render executive hero landing screen with colorful cards, KPIs, and 4-step pipeline visual."""

    # 1. Executive Hero Card
    st.markdown(
        """
        <div class="saas-hero-card">
            <div class="hero-pill-tag">
                <span class="radar-dot"></span>
                <span>BHARAT LEGAL INTELLIGENCE • 2026 EDITION</span>
            </div>
            <div class="hero-heading">
                Your Rights, Your Procedure. <br><span class="gradient-text">Decoded with Precision.</span>
            </div>
            <div class="hero-subheading">
                Context-aware legal analysis engineered specifically for Indian Law — cross-referencing
                the Bharatiya Nyaya Sanhita (BNS), BNSS / CrPC, Consumer Protection Act, and Labour Codes.
            </div>
            <div class="hero-feature-tags">
                <span class="feature-tag-chip">⚡ Groq High-Speed Inference</span>
                <span class="feature-tag-chip">📚 16,000+ Pinecone Legal Chunks</span>
                <span class="feature-tag-chip">🌐 SerpAPI Live Web Fallback</span>
                <span class="feature-tag-chip">🔒 100% Private Local Storage</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. KPI Summary Row
    st.markdown(
        """
        <div class="kpi-row-grid">
            <div class="kpi-metric-card indigo">
                <div class="kpi-card-header">
                    <div class="kpi-card-icon indigo">📚</div>
                    <span class="kpi-card-trend up">Verified</span>
                </div>
                <div class="kpi-card-value">16,240+</div>
                <div class="kpi-card-label">Statutory Vector Chunks</div>
            </div>
            <div class="kpi-metric-card emerald">
                <div class="kpi-card-header">
                    <div class="kpi-card-icon emerald">⚖️</div>
                    <span class="kpi-card-trend up">Active</span>
                </div>
                <div class="kpi-card-value">BNS 2024</div>
                <div class="kpi-card-label">Criminal Codes In Effect</div>
            </div>
            <div class="kpi-metric-card amber">
                <div class="kpi-card-header">
                    <div class="kpi-card-icon amber">⚡</div>
                    <span class="kpi-card-trend neutral">&lt; 1.5s</span>
                </div>
                <div class="kpi-card-value">Groq LLM</div>
                <div class="kpi-card-label">Ultra-Fast Reasoning</div>
            </div>
            <div class="kpi-metric-card rose">
                <div class="kpi-card-header">
                    <div class="kpi-card-icon rose">🛡️</div>
                    <span class="kpi-card-trend up">Zero Cloud DB</span>
                </div>
                <div class="kpi-card-value">100% Local</div>
                <div class="kpi-card-label">Private SQLite Telemetry</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. How It Works Pipeline Visual
    st.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin: 1.8rem 0 0.8rem 0;">
            🧠 Directed Acyclic Graph (DAG) Multi-Stage Workflow
        </div>
        <div class="dag-flow-grid">
            <div class="dag-node-box">
                <div class="dag-node-badge" style="background: linear-gradient(135deg, #6366f1, #4f46e5);">01</div>
                <div class="dag-node-title">Classify & Route</div>
                <div class="dag-node-desc">Filters Indian legal queries vs. general conversation, checking domain scope.</div>
            </div>
            <div class="dag-node-box">
                <div class="dag-node-badge" style="background: linear-gradient(135deg, #8b5cf6, #7c3aed);">02</div>
                <div class="dag-node-title">Refine & Risk Score</div>
                <div class="dag-node-desc">Extracts statutes & assigns Low / Medium / High legal risk tag.</div>
            </div>
            <div class="dag-node-box">
                <div class="dag-node-badge" style="background: linear-gradient(135deg, #f59e0b, #d97706);">03</div>
                <div class="dag-node-title">Cluster RAG Search</div>
                <div class="dag-node-desc">KMeans cluster vector search over Pinecone with live SerpAPI web fallback.</div>
            </div>
            <div class="dag-node-box">
                <div class="dag-node-badge" style="background: linear-gradient(135deg, #10b981, #059669);">04</div>
                <div class="dag-node-title">Action Roadmap</div>
                <div class="dag-node-desc">Synthesizes structured rights, statutory citations, and lawyer recommendation.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Interactive BNS vs IPC Navigator Expander
    render_bns_ipc_matrix(in_expander=True)

    # 5. Frequently Explored Scenarios
    st.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin: 2rem 0 1rem 0;">
            💡 Frequently Explored Legal Scenarios (One-Click Inquiries)
        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = list(STARTER_QUESTIONS.items())
    cols = st.columns(2)

    for idx, (cat_name, questions) in enumerate(categories):
        col = cols[idx % 2]
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: #ffffff;">{cat_name}</span>
                        <span style="font-size: 0.72rem; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.12); padding: 2px 8px; border-radius: 6px;">Curated Scenarios</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for q in questions:
                    if st.button(f"👉 {q}", key=f"starter_{idx}_{q[:24]}", **get_width_kw()):
                        if on_select_query:
                            on_select_query(q)
                        st.session_state["pending_query"] = q
                        st.rerun()


def render_pipeline_status(status_events: list[dict]) -> None:
    """Render real-time node-by-node execution pipeline in st.status."""
    with st.status("🧠 Orchestrating Indian Legal Analysis DAG...", expanded=True) as status_box:
        for ev in status_events:
            title = ev.get("title", ev.get("node", ""))
            msg = ev.get("message", "")
            if ev.get("status") == "done":
                st.write(f"✅ **{title}** — {msg}")
            else:
                st.write(f"⏳ **{title}** — {msg}")
        status_box.update(label="🎉 Multi-Stage Legal Workflow Completed", state="complete", expanded=False)


def render_pipeline_expander(metadata: dict) -> None:
    """Renders a 'How I Answered This' transparent DAG execution inspection."""
    refined = metadata.get("refined_query", "")
    cluster = metadata.get("matched_cluster")
    score = metadata.get("confidence_score")
    source = metadata.get("answer_source") or metadata.get("source", "RAG")
    latency = metadata.get("latency")
    fallback_reason = metadata.get("fallback_reason")

    cluster_text = f"Cluster #{cluster}" if cluster is not None else "Direct semantic match"
    score_text = f"{score:.2f}" if score is not None else "N/A"
    latency_text = f"{latency}s" if latency else "fast"

    with st.expander("🔍 Pipeline DAG Transparency & Retrieval Inspector", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            if source == "huggingface_fallback" or fallback_reason:
                st.metric("Source Engine", "⚡ Indian Law Fallback (Qwen/HF)")
            elif source in ("pinecone_rag", "RAG"):
                st.metric("Source Engine", "🌲 Pinecone Vector DB")
            else:
                st.metric("Source Engine", "🌐 Live Web Search")
        with c2:
            st.metric("Retrieval Confidence", score_text)
        with c3:
            st.metric("Total Latency", latency_text)

        st.markdown(f"**Refined Indian Law Query**: `{refined}`")
        if fallback_reason:
            st.warning(f"⚠️ **Fallback Triggered**: {fallback_reason}")
        elif cluster is not None:
            st.caption(f"Scoped to nearest semantic document cluster: **{cluster_text}**")


def render_structured_answer(
    message: dict,
    db: Any,
    on_followup: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Renders an executive, modern structured legal response card:
      - Top Segmented Meta Panel (Risk pill, Source pill, Confidence bar)
      - Lawyer Advisory Banner
      - Tabbed Deep Dive (Summary, Relevant Statutes, Interactive Checklist Roadmap, Rights, Sources)
      - Action Toolbar (Feedback, Bookmark, Markdown, PDF)
      - Follow-up recommendation chips
      - Disclaimer notice
    """
    msg_id = message.get("id")
    content = message.get("content", "")
    meta = message.get("metadata", {}) or {}
    sections = meta.get("answer_sections", {}) or {}

    risk = meta.get("risk", "low").lower()
    source = meta.get("source", "RAG")
    confidence = meta.get("confidence_score", 0.75)
    lawyer_needed = meta.get("lawyer_needed", sections.get("lawyer_needed", "Optional"))
    lawyer_reason = meta.get("lawyer_reason", sections.get("lawyer_reason", ""))

    summary = sections.get("legal_summary") or content
    relevant_laws = sections.get("relevant_laws", [])
    roadmap_steps = sections.get("action_roadmap", [])
    rights = sections.get("important_rights", [])
    final_note = sections.get("final_note", "")
    disclaimer = sections.get("disclaimer", "Information is for general guidance only under Indian law.")

    # 1. Top Segmented Meta Panel (Compact Horizontal Status Bar)
    conf_pct = int((confidence if confidence is not None else 0.75) * 100)

    if source in ("huggingface_fallback", "openrouter_qwen") or meta.get("fallback_reason"):
        source_label = "⚡ Indian Law Fallback (Qwen/HF)"
    elif source in ("pinecone_rag", "RAG"):
        source_label = "📚 Pinecone Legal VectorDB"
    else:
        source_label = "🌐 Live Indian Web Search"

    st.markdown(
        f"""
        <div class="response-status-bar">
            <div class="response-status-left">
                <span class="risk-halo-pill {risk}">
                    {risk_emoji(risk)} {risk.upper()} RISK
                </span>
                <span class="source-halo-pill">
                    {source_label}
                </span>
            </div>
            <div class="response-status-right">
                <div class="confidence-fill-box">
                    <span>Similarity: <b>{conf_pct}%</b></span>
                    <div class="confidence-rail">
                        <div class="confidence-slider" style="width: {conf_pct}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Compact Advocate Consultation Advisory
    lawyer_cls = lawyer_needed.lower().replace(" ", "-")
    lawyer_icon = "🚨" if ("require" in lawyer_cls or "yes" in lawyer_cls) else ("⚖️" if "optional" in lawyer_cls else "✅")
    st.markdown(
        f"""
        <div class="premium-lawyer-banner {lawyer_cls}">
            <div class="lawyer-badge-large-icon">{lawyer_icon}</div>
            <div class="lawyer-banner-content">
                <span class="lawyer-banner-main">Advocate Consultation: <b>{lawyer_needed}</b></span>
                <span class="lawyer-banner-sub">— {lawyer_reason or 'Formal legal consultation is advised before executing statutory filings.'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Tabs: Deep-Dive Presentation
    tabs = st.tabs(["📋 Legal Summary", "📜 Applicable Statutes", "🗺️ Action Roadmap", "🛡️ Citizen Rights", "📑 Raw Sources"])

    with tabs[0]:
        from core.tts import render_browser_tts_component
        st.markdown(render_browser_tts_component(summary, label="🔊 Listen to Legal Advice (Qwen3-TTS 1.7B)"), unsafe_allow_html=True)
        st.markdown(summary)
        if final_note:
            st.info(f"💡 **Practical Advisory**: {final_note}")

    with tabs[1]:
        if relevant_laws:
            st.markdown("##### 🏛️ Applicable Statutory Provisions")
            for item in relevant_laws:
                sec = item.get("section", "")
                desc = item.get("description", "")
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px;">
                        <span class="statute-code-chip">§ {sec}</span>
                        <span style="color: #cbd5e1; font-size: 0.92rem;">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("General procedural doctrines of Indian Law apply to this scenario.")

    with tabs[2]:
        st.markdown("##### 🗺️ Step-by-Step Action Plan")
        st.caption("Check off items as you proceed; your progress is automatically saved to local SQLite.")

        if roadmap_steps:
            progress = db.get_roadmap_progress(msg_id) if (db and msg_id) else {}
            completed_count = sum(1 for v in progress.values() if v)
            total_count = len(roadmap_steps)
            progress_pct = completed_count / total_count if total_count > 0 else 0.0

            st.progress(progress_pct, text=f"Action Progress: {completed_count}/{total_count} steps completed")

            for step_idx, step_text in enumerate(roadmap_steps, 1):
                col_check, col_text = st.columns([0.08, 0.92])
                with col_check:
                    is_done = progress.get(step_idx, False)
                    checked = st.checkbox(
                        label=f"Step {step_idx}",
                        value=is_done,
                        key=f"step_{msg_id}_{step_idx}",
                        label_visibility="collapsed",
                    )
                    if checked != is_done and db and msg_id:
                        db.update_roadmap_step(msg_id, step_idx, checked)
                with col_text:
                    st.markdown(f"**Step {step_idx}:** {step_text}")
        else:
            st.markdown("1. File formal written grievance with jurisdictional authority.\n2. Retain date-stamped acknowledgments.")

    with tabs[3]:
        if rights:
            st.markdown("##### 🛡️ Citizen Legal Rights Under Indian Law")
            for r in rights:
                st.markdown(f"• **{r}**")
        else:
            st.markdown("Guaranteed fundamental rights under Part III of the Constitution of India apply.")

    with tabs[4]:
        chunks = meta.get("retrieved_chunks", [])
        web_res = meta.get("web_results", [])

        if chunks:
            st.markdown("**Retrieved Knowledge Base Chunks (Pinecone):**")
            for i, chunk in enumerate(chunks, 1):
                with st.expander(f"Chunk {i} — Relevance Score: {chunk.get('score', 'N/A')}"):
                    st.text(chunk.get("text", ""))

        if web_res:
            st.markdown("**Live Web Context Sources (SerpAPI):**")
            for w in web_res:
                st.markdown(f"- [{w.get('title')}]({w.get('link')}) — *{w.get('snippet')}*")

        if not chunks and not web_res:
            st.caption("Statutory context synthesized from general legal knowledge base.")

    # 4. Pipeline Transparency Expander
    render_pipeline_expander(meta)

    # 5. Action Toolbar
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 14px 0 10px 0;'>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([0.22, 0.22, 0.22, 0.17, 0.17])

    with col1:
        if st.button("👍 Helpful", key=f"thumb_up_{msg_id}", **get_width_kw()):
            if db and msg_id:
                db.add_feedback(msg_id, 1)
            st.toast("Feedback recorded! Thank you.", icon="🙏")

    with col2:
        if st.button("👎 Inaccurate", key=f"thumb_down_{msg_id}", **get_width_kw()):
            if db and msg_id:
                db.add_feedback(msg_id, -1)
            st.toast("Feedback recorded for review.", icon="📝")

    with col3:
        is_bm = db.is_bookmarked(msg_id) if (db and msg_id) else False
        btn_label = "🔖 Saved" if is_bm else "⭐ Bookmark"
        if st.button(btn_label, key=f"bm_{msg_id}", **get_width_kw()):
            if db and msg_id:
                if not is_bm:
                    db.add_bookmark(msg_id)
                    st.toast("Saved to Case Bookmarks!", icon="⭐")
                    st.rerun()

    with col4:
        md_data = export_markdown("Legal Consultation", [message])
        st.download_button(
            label="📄 MD",
            data=md_data,
            file_name="lexai_consultation.md",
            mime="text/markdown",
            key=f"dl_md_{msg_id}",
            **get_width_kw(fn=st.download_button),
        )

    with col5:
        pdf_bytes = export_pdf("Legal Consultation", [message])
        st.download_button(
            label="📑 PDF",
            data=pdf_bytes,
            file_name="lexai_consultation.pdf",
            mime="application/pdf",
            key=f"dl_pdf_{msg_id}",
            **get_width_kw(fn=st.download_button),
        )

    # 6. Follow-up Suggestions
    followups = meta.get("followups", [])
    if followups:
        st.markdown("##### 🔍 Explore Related Questions:")
        f_cols = st.columns(len(followups))
        for f_idx, f_q in enumerate(followups):
            with f_cols[f_idx]:
                if st.button(f_q, key=f"fup_{msg_id}_{f_idx}", **get_width_kw()):
                    if on_followup:
                        on_followup(f_q)
                    st.session_state["pending_query"] = f_q
                    st.rerun()

    # 7. Compliance Disclaimer
    st.markdown(
        f"""
        <div class="legal-disclaimer-box">
            ⚖️ <b>Legal Disclaimer</b>: {disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_helplines_card() -> None:
    """Render Emergency Helplines quick card with dial pill badges."""
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>🚨 Emergency Legal & Crisis Helplines (India)</h2>
            <p>Official 24/7 national toll-free helplines for immediate legal assistance, crime reporting, and intervention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for h in HELPLINES:
        st.markdown(
            f"""
            <div class="sos-helpline-card">
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff;">{h['icon']} {h['name']}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 3px;">{h['desc']}</div>
                </div>
                <div class="sos-dial-badge">{h['number']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_glossary_view() -> None:
    """Render interactive legal jargon glossary with pill badges."""
    st.markdown(
        """
        <div class="page-title-banner">
            <h2>📖 Indian Legal Terms Glossary</h2>
            <p>Plain-language explanations for court terms, criminal procedure, and statutory phrases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input("🔍 Search term or abbreviation...", placeholder="e.g. FIR, BNS, Bail, PIL, Caveat, Anticipatory")
    filtered = {
        k: v for k, v in LEGAL_GLOSSARY.items()
        if not search or search.lower() in k.lower() or search.lower() in v.lower()
    }

    cols = st.columns(2)
    for i, (term, definition) in enumerate(filtered.items()):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**⚖️ {term}**")
                st.markdown(f"<span style='font-size: 0.92rem; color: #cbd5e1;'>{definition}</span>", unsafe_allow_html=True)

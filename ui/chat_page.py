"""
LexAI Chat Page Module
Handles message rendering, metadata captions, expandable statutory sections,
dynamic follow-up chips, and sticky chat input with pending prompt dispatch.
"""

import streamlit as st
from services.chat_service import send_message
from services.conversations import get_messages


EXAMPLE_PROMPTS = [
    "Can my employer fire me without notice?",
    "What are my rights if I am arrested?",
    "How do I file a consumer complaint?",
    "Explain BNS Section 103.",
]


def _on_pill_change(pill_key: str):
    """Callback for st.pills to route follow-up questions to pending_prompt."""
    selected = st.session_state.get(pill_key)
    if selected:
        st.session_state["pending_prompt"] = selected
        st.session_state[pill_key] = None


def render_chat_page():
    conv_id = st.session_state.get("current_conversation_id")
    messages = get_messages(conv_id) if conv_id else []

    # Find index of the latest assistant message
    last_asst_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_asst_idx = idx
            break

    # 1. Render Messages or Empty State
    if not messages and not st.session_state.get("pending_prompt"):
        st.markdown(
            """
            <div style="text-align: center; margin-top: 4.5rem; margin-bottom: 2rem;">
                <div style="font-size: 3.8rem; margin-bottom: 0.6rem; line-height: 1;">⚖️</div>
                <h2 style="font-weight: 600; font-size: 1.65rem; margin: 0; color: #ececf1;">
                    How can I help you with Indian law?
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Clickable Example Prompts (2x2 grid)
        col1, col2 = st.columns(2)
        for idx, prompt_text in enumerate(EXAMPLE_PROMPTS):
            col = col1 if idx % 2 == 0 else col2
            with col:
                if st.button(
                    prompt_text,
                    key=f"starter_chip_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["pending_prompt"] = prompt_text
                    st.rerun()
    else:
        # Render existing messages chronologically
        for i, msg in enumerate(messages):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

            with st.chat_message(role):
                # Markdown, lists, tables, and code blocks render via st.markdown
                st.markdown(content)

                # Metadata and expandable sections for assistant replies
                if role == "assistant":
                    meta = msg.get("metadata") or {}

                    # Muted caption row: Risk level • Model used • Retrieval confidence
                    caption_items = []

                    risk = meta.get("risk")
                    if risk:
                        caption_items.append(f"Risk: {str(risk).capitalize()}")

                    if meta.get("fallback_used"):
                        caption_items.append("Model: Indian Law Fallback")
                    else:
                        caption_items.append("Model: Gemini")

                    # Confidence: hide if missing or <= 0 (never show 0%)
                    conf = meta.get("confidence")
                    if conf is None:
                        conf = meta.get("confidence_score")
                    if conf is not None:
                        try:
                            conf_val = float(conf)
                            if conf_val > 0.0:
                                caption_items.append(f"Confidence: {int(conf_val * 100)}%")
                        except (ValueError, TypeError):
                            pass

                    if caption_items:
                        st.caption(" • ".join(caption_items))

                    # Expanders (collapsed by default: expanded=False)
                    # 1. Applicable Statutes
                    statutes = meta.get("statutes") or meta.get("relevant_laws")
                    if statutes and isinstance(statutes, list) and len(statutes) > 0:
                        with st.expander("⚖️ Applicable Statutes", expanded=False):
                            for s in statutes:
                                if isinstance(s, dict):
                                    sec = s.get("section") or s.get("title") or ""
                                    desc = s.get("description") or s.get("summary") or ""
                                    if sec and desc:
                                        st.markdown(f"- **{sec}**: {desc}")
                                    elif sec:
                                        st.markdown(f"- **{sec}**")
                                    elif desc:
                                        st.markdown(f"- {desc}")
                                else:
                                    st.markdown(f"- {s}")

                    # 2. Action Steps
                    steps = meta.get("steps") or meta.get("action_roadmap")
                    if steps and isinstance(steps, list) and len(steps) > 0:
                        with st.expander("📋 Action Steps", expanded=False):
                            for s_idx, step in enumerate(steps, 1):
                                st.markdown(f"{s_idx}. {step}")

                    # 3. Your Rights
                    rights = meta.get("rights") or meta.get("important_rights")
                    if rights and isinstance(rights, list) and len(rights) > 0:
                        with st.expander("🛡️ Your Rights", expanded=False):
                            for r in rights:
                                st.markdown(f"- {r}")

                    # 4. Sources
                    sources = meta.get("sources") or meta.get("retrieved_chunks")
                    if sources and isinstance(sources, list) and len(sources) > 0:
                        with st.expander("📚 Sources", expanded=False):
                            for src in sources:
                                if isinstance(src, dict):
                                    src_title = src.get("title") or src.get("source") or src.get("statute") or "Legal Authority"
                                    src_text = src.get("text") or src.get("content") or ""
                                    st.markdown(f"**{src_title}**")
                                    if src_text:
                                        st.caption(src_text[:280] + ("..." if len(src_text) > 280 else ""))
                                else:
                                    st.markdown(f"- {src}")

                    # Follow-ups: Render ONLY under the LATEST AI reply
                    if i == last_asst_idx:
                        follow_ups = meta.get("follow_ups") or []
                        valid_fus = [
                            str(q).strip() for q in follow_ups
                            if isinstance(q, (str, int, float)) and str(q).strip()
                        ][:3]

                        if valid_fus:
                            st.markdown(
                                '<div style="font-size: 0.78rem; font-weight: 600; color: #8e8ea0; margin-top: 0.65rem; margin-bottom: 0.25rem;">SUGGESTED FOLLOW-UPS</div>',
                                unsafe_allow_html=True,
                            )
                            if hasattr(st, "pills"):
                                pill_key = f"fu_pills_{msg.get('id', 'latest')}"
                                st.pills(
                                    "Suggested follow-ups",
                                    options=valid_fus,
                                    selection_mode="single",
                                    key=pill_key,
                                    label_visibility="collapsed",
                                    on_change=_on_pill_change,
                                    args=(pill_key,),
                                )
                            else:
                                for fu_idx, fu_q in enumerate(valid_fus):
                                    if st.button(fu_q, key=f"fu_btn_{msg.get('id')}_{fu_idx}", use_container_width=True):
                                        st.session_state["pending_prompt"] = fu_q
                                        st.rerun()

        # Spacer at bottom to guarantee the last message is never hidden behind chat input
        st.markdown('<div style="height: 3.5rem;"></div>', unsafe_allow_html=True)

    # 2. Chat Input Widget
    user_typed = st.chat_input("Message LexAI (Indian Law Assistant)...", key="main_chat_input_field")

    # If user submitted via chat_input, route into pending_prompt
    if user_typed:
        st.session_state["pending_prompt"] = user_typed

    # 3. Single Unified Message Sending Path
    if st.session_state.get("pending_prompt"):
        prompt_to_send = st.session_state.pop("pending_prompt").strip()
        if prompt_to_send:
            # Render user message optimistically
            with st.chat_message("user"):
                st.markdown(prompt_to_send)

            # Process consultation through RAG/LLM/Fallback pipeline with spinner
            with st.spinner("Analyzing Indian statutes & precedent..."):
                try:
                    response = send_message(conv_id, prompt_to_send)
                    st.session_state["current_conversation_id"] = response["conversation_id"]
                    st.query_params["c"] = response["conversation_id"]
                except Exception as e:
                    st.error(f"Consultation error: {e}")

            # Rerun to load updated message history and dynamic follow-up chips
            st.rerun()

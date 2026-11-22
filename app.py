"""
LexAI - Indian Legal Assistant
ChatGPT-style Streamlit Application
Responsive layout supporting desktop (1440px), tablet (768px), and mobile (375px).
"""

from datetime import datetime, timezone, timedelta
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="LexAI - Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Responsive Custom CSS
CUSTOM_CSS = """
<style>
/* Base container centering at ~760px max width */
.block-container {
    max-width: 760px;
    padding-top: 1.5rem;
    padding-bottom: 7.5rem;
    margin: 0 auto;
}

/* Responsive adjustments for tablet (768px) and mobile (375px) */
@media (max-width: 768px) {
    .block-container {
        max-width: 100% !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
        padding-bottom: 7rem !important;
    }
    div[data-testid="stChatInput"] {
        max-width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
}

/* Chat input pinned and centered within 760px */
div[data-testid="stChatInput"] {
    max-width: 760px;
    margin: 0 auto;
}

/* Prevent horizontal overflow on code blocks and tables while keeping them scrollable */
pre, code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
div[data-testid="stTable"], table {
    display: block;
    overflow-x: auto;
    max-width: 100%;
}

/* Compact sidebar layout */
section[data-testid="stSidebar"] {
    padding-top: 1rem;
}

/* Sidebar action and history buttons */
div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    border-radius: 6px;
    text-align: left;
    font-size: 0.85rem;
    padding: 0.35rem 0.55rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Sidebar options popover */
div[data-testid="stSidebar"] div[data-testid="stPopover"] > button {
    padding: 0.35rem 0.45rem;
    border-radius: 6px;
}

/* Category section labels */
.group-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8e8ea0;
    margin-top: 0.85rem;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Pills & follow-up wrapping */
div[data-testid="stPills"] {
    flex-wrap: wrap !important;
}
div[data-testid="stPills"] button {
    white-space: normal !important;
    text-align: left !important;
    height: auto !important;
    line-height: 1.35 !important;
    padding: 0.45rem 0.85rem !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Synchronize conversation ID with query params for seamless browser refresh
if "current_conversation_id" not in st.session_state:
    param_cid = st.query_params.get("c")
    st.session_state["current_conversation_id"] = param_cid if param_cid else None

# 4. Imports from Services & UI Pages
from services.conversations import (
    list_conversations,
    rename_conversation,
    delete_conversation,
)
from ui.chat_page import render_chat_page
from ui.tools_pages import (
    render_matrix_page,
    render_draft_page,
    render_analyzer_page,
    render_glossary_page,
    render_helplines_page,
    render_settings_page,
)

# 5. Multi-Page Navigation Configuration
page_chat = st.Page(render_chat_page, title="Chat", icon="💬", default=True)
page_matrix = st.Page(render_matrix_page, title="BNS-IPC Matrix", icon="⚖️", url_path="matrix")
page_draft = st.Page(render_draft_page, title="Legal Draft Studio", icon="📝", url_path="draft-studio")
page_analyzer = st.Page(render_analyzer_page, title="Document Analyzer", icon="📄", url_path="document-analyzer")
page_glossary = st.Page(render_glossary_page, title="Statutory Glossary", icon="📖", url_path="statutory-glossary")
page_helplines = st.Page(render_helplines_page, title="Emergency Helplines", icon="🚨", url_path="helplines")
page_settings = st.Page(render_settings_page, title="Settings", icon="⚙️", url_path="settings")

all_pages = [
    page_chat,
    page_matrix,
    page_draft,
    page_analyzer,
    page_glossary,
    page_helplines,
    page_settings,
]
current_page = st.navigation(all_pages, position="hidden")

# 6. Sidebar Construction
with st.sidebar:
    # Logo and App Name
    st.markdown("### ⚖️ **LexAI**")

    # "+ New Chat" Action Button
    if st.button("➕ New Chat", key="btn_new_chat", use_container_width=True, type="primary"):
        st.session_state["current_conversation_id"] = None
        st.query_params.clear()
        st.switch_page(page_chat)

    # Search Box to filter conversation titles
    search_query = st.text_input(
        "Search conversations",
        placeholder="Search conversations...",
        key="search_conversations_input",
        label_visibility="collapsed",
    )

    # Conversation History Grouping
    st.markdown('<div class="group-header">HISTORY</div>', unsafe_allow_html=True)

    conversations = list_conversations(limit=100)
    q = (search_query or "").strip().lower()
    if q:
        conversations = [c for c in conversations if q in c.get("title", "").lower()]

    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    yesterday = today - timedelta(days=1)
    seven_days_ago = today - timedelta(days=7)

    groups = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Older": [],
    }

    for c in conversations:
        ts_str = c.get("updated_at") or c.get("created_at") or ""
        try:
            clean_ts = ts_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
            c_date = dt.date()
        except Exception:
            c_date = today

        if c_date == today:
            groups["Today"].append(c)
        elif c_date == yesterday:
            groups["Yesterday"].append(c)
        elif seven_days_ago <= c_date < yesterday:
            groups["Previous 7 Days"].append(c)
        else:
            groups["Older"].append(c)

    has_any = any(len(lst) > 0 for lst in groups.values())
    if not has_any:
        st.caption("No conversations yet.")
    else:
        for group_name, conv_list in groups.items():
            if not conv_list:
                continue
            st.markdown(f'<div class="group-header">{group_name}</div>', unsafe_allow_html=True)
            for conv in conv_list:
                c1, c2 = st.columns([0.82, 0.18], vertical_alignment="center")
                with c1:
                    title = conv.get("title") or "New Consultation"
                    short_title = title[:22] + "…" if len(title) > 23 else title
                    is_active = (
                        st.session_state.get("current_conversation_id") == conv["id"]
                        and current_page == page_chat
                    )
                    if st.button(
                        short_title,
                        key=f"conv_btn_{conv['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                        help=title,
                    ):
                        st.session_state["current_conversation_id"] = conv["id"]
                        st.query_params["c"] = conv["id"]
                        st.switch_page(page_chat)
                with c2:
                    with st.popover("⋯", help="Options"):
                        new_title = st.text_input(
                            "Rename",
                            value=conv.get("title", ""),
                            key=f"rename_input_{conv['id']}",
                        )
                        if st.button("Save", key=f"rename_save_{conv['id']}"):
                            if new_title.strip():
                                rename_conversation(conv["id"], new_title.strip())
                                st.rerun()
                        st.markdown("---")
                        confirm = st.checkbox("Confirm delete", key=f"del_confirm_{conv['id']}")
                        if st.button(
                            "🗑️ Delete",
                            key=f"del_execute_{conv['id']}",
                            disabled=not confirm,
                            type="primary",
                        ):
                            delete_conversation(conv["id"])
                            if st.session_state.get("current_conversation_id") == conv["id"]:
                                st.session_state["current_conversation_id"] = None
                                st.query_params.clear()
                            st.rerun()

    # Compact Navigation at Bottom of Sidebar
    st.markdown('<div class="group-header">LEGAL TOOLS</div>', unsafe_allow_html=True)
    if current_page != page_chat:
        if st.button("💬 Chat Assistant", key="nav_btn_return_chat", use_container_width=True, type="secondary"):
            st.switch_page(page_chat)

    tools = [
        ("⚖️ BNS-IPC Matrix", page_matrix),
        ("📝 Legal Draft Studio", page_draft),
        ("📄 Document Analyzer", page_analyzer),
        ("📖 Statutory Glossary", page_glossary),
        ("🚨 Emergency Helplines", page_helplines),
        ("⚙️ Settings", page_settings),
    ]
    for label, target_page in tools:
        is_active = (current_page == target_page)
        if st.button(
            label,
            key=f"nav_tool_{target_page.url_path}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.switch_page(target_page)

# 7. Run Current Page Router
current_page.run()

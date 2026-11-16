import os
from pathlib import Path
from time import monotonic
from uuid import uuid4

import streamlit as st

import alexander


st.set_page_config(page_title="Alexander")
st.title("Alexander")
st.caption("Passphrase-protected personal assistant")

SESSION_IDLE_TIMEOUT = 30 * 60


if len(os.environ.get("ALEXANDER_PASSWORD", "")) < alexander.MIN_PASSWORD_LENGTH:
    st.error(
        "Set ALEXANDER_PASSWORD to a passphrase of at least "
        f"{alexander.MIN_PASSWORD_LENGTH} characters before using Alexander."
    )
    st.stop()

if st.session_state.get("authenticated", False):
    last_activity = st.session_state.get("last_authenticated_at", 0.0)
    if monotonic() - last_activity >= SESSION_IDLE_TIMEOUT:
        st.session_state.authenticated = False

if not st.session_state.get("authenticated", False):
    with st.form("unlock-form"):
        phrase = st.text_input("Owner passphrase", type="password", max_chars=1024)
        submitted = st.form_submit_button("Unlock")
    if submitted:
        client_key = getattr(st.context, "ip_address", None) or "streamlit"
        authenticated, retry_in = alexander.authenticate_password(phrase, client_key)
        if authenticated:
            st.session_state.authenticated = True
            st.session_state.last_authenticated_at = monotonic()
            st.rerun()
        if retry_in:
            st.error(f"Too many attempts. Try again in {retry_in} seconds.")
        else:
            st.error("Passphrase rejected.")
    st.stop()

st.session_state.last_authenticated_at = monotonic()


def deny_web_tool(message: str) -> bool:
    alexander.log.warning("Streamlit denied risky tool: %s", message)
    return False


if "assistant" not in st.session_state:
    alexander.ensure_dirs()
    memory_dir = Path(".alexander_sessions")
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory = alexander.Memory(path=str(memory_dir / f"{uuid4().hex}.json"))
    knowledge = alexander.Knowledge()
    knowledge.sync()
    st.session_state.assistant = {
        "brain": alexander.Brain(confirm_fn=deny_web_tool),
        "team": alexander.build_team(knowledge, confirm_fn=deny_web_tool),
        "memory": memory,
        "knowledge": knowledge,
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Conversation")
    if st.button("Lock"):
        st.session_state.authenticated = False
        st.session_state.pop("last_authenticated_at", None)
        st.rerun()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.assistant["brain"].reset()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Type to Alexander..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant = st.session_state.assistant
                reply = alexander.handle(
                    prompt,
                    assistant["brain"],
                    assistant["team"],
                    memory=assistant["memory"],
                    knowledge=assistant["knowledge"],
                )
            except Exception:
                alexander.log.exception("Streamlit chat request failed")
                reply = "Sorry, something went wrong while handling that request."
                st.error(reply)
            else:
                st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

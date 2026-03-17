import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Clinical Document Intelligence",
    page_icon="🏥",
    layout="wide",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Clinical RAG")
    st.caption("Powered by GPT-4o + ChromaDB")
    st.divider()

    # Document ingestion
    st.subheader("Upload Document")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, HTML",
        type=["pdf", "docx", "txt", "html", "htm"],
    )
    if uploaded and st.button("Ingest Document"):
        with st.spinner("Ingesting…"):
            resp = requests.post(
                f"{API_BASE}/api/v1/documents/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if resp.ok:
            data = resp.json()
            st.success(f"Ingested {data['chunks_ingested']} chunks from {data['filename']}")
        else:
            st.error(f"Ingestion failed: {resp.text}")

    st.divider()

    # Ingested sources
    st.subheader("Knowledge Base")
    if st.button("Refresh Sources"):
        resp = requests.get(f"{API_BASE}/api/v1/documents/sources")
        if resp.ok:
            sources = resp.json()["sources"]
            if sources:
                for s in sources:
                    st.markdown(f"- {s}")
            else:
                st.info("No documents ingested yet.")

# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("Clinical Document Intelligence System")
st.caption("Ask questions about clinical guidelines, drug protocols, and WHO/NHS documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** — Page {src.get('page', 'N/A')}\n\n"
                        f"> {src['excerpt']}"
                    )

if prompt := st.chat_input("Ask a clinical question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching clinical documents…"):
            resp = requests.post(
                f"{API_BASE}/api/v1/query/",
                json={"question": prompt},
            )
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            if data["sources"]:
                with st.expander(f"Sources ({len(data['sources'])})"):
                    for src in data["sources"]:
                        st.markdown(
                            f"**{src['source']}** — Page {src.get('page', 'N/A')}\n\n"
                            f"> {src['excerpt']}"
                        )
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
            })
        else:
            err = "Failed to get a response. Check the API is running."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

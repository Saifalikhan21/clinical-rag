"""LangGraph RAG pipeline with grounded citations."""
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.core.llm import get_llm
from src.core.vectorstore import get_retriever
from src.rag.pipeline import SYSTEM_PROMPT, HUMAN_PROMPT


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    sources: list[dict]


def retrieve(state: RAGState) -> RAGState:
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def generate(state: RAGState) -> RAGState:
    llm = get_llm()
    docs = state["documents"]

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, "
        f"Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=HUMAN_PROMPT.format(question=state["question"])),
    ]

    response = llm.invoke(messages)

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "N/A"),
            "chunk_index": d.metadata.get("chunk_index"),
            "excerpt": d.page_content[:200] + "..." if len(d.page_content) > 200 else d.page_content,
        }
        for d in docs
    ]

    return {**state, "answer": response.content, "sources": sources}


def no_documents_found(state: RAGState) -> RAGState:
    return {
        **state,
        "answer": (
            "I could not find relevant information in the clinical document library "
            "to answer your question. Please verify the documents have been ingested "
            "or rephrase your query."
        ),
        "sources": [],
    }


def route_after_retrieval(state: RAGState) -> str:
    return "generate" if state["documents"] else "no_documents_found"


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_documents_found", no_documents_found)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieval)
    graph.add_edge("generate", END)
    graph.add_edge("no_documents_found", END)

    return graph.compile()


# Module-level compiled graph (reused across requests)
rag_graph = build_rag_graph()


def query(question: str) -> dict:
    """Run the RAG pipeline and return answer + sources."""
    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "answer": "",
        "sources": [],
    }
    result = rag_graph.invoke(initial_state)
    return {"answer": result["answer"], "sources": result["sources"]}

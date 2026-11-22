"""
LangGraph orchestration pipeline for LexAI.
Preserves the exact DAG logic from agent.ipynb while exposing a generator
run_query() that yields real-time node progress events, and returns
a structured dict with parsed sections and metadata.
"""

import time
from typing import Any, Generator, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
    _HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = "__end__"
    _HAS_LANGGRAPH = False

from core.config import (
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    LAW_KEYWORDS,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    RAG_CONFIDENCE_THRESHOLD,
    RAG_THRESHOLD,
    RAG_TOP_K,
)
import core.rag
from core.hf_fallback import generate_indian_legal_fallback
from core.llm import (
    call_llm,
    generate_followup_questions,
)
from core.prompts import (
    GENERAL_RESPONSE,
    LEGAL_EXPERT,
    QUERY_REFINER,
    RISK_CLASSIFIER,
    ROADMAP_GENERATOR,
    STYLE_DETAILED,
    STYLE_SIMPLE,
)
from core.rag import search_rag
from core.search import search_web
from utils.helpers import parse_risk_response, parse_structured_answer


# ── TypedDict for LangGraph State ─────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    domain: str
    refined_query: str
    risk: str
    risk_reason: str
    context: str
    score: float
    confidence_score: float
    matched_cluster: Optional[int]
    retrieved_chunks: list[dict]
    retrieved_documents_count: int
    web_results: list[dict]
    source: str
    answer_source: str
    fallback_reason: Optional[str]
    response_status: str
    roadmap: str
    final_answer: str
    structured_sections: dict[str, Any]
    style: str
    groq_api_key: Optional[str]
    pinecone_api_key: Optional[str]
    serpapi_api_key: Optional[str]
    openrouter_api_key: Optional[str]
    openrouter_model: Optional[str]
    hf_api_key: Optional[str]
    hf_model: Optional[str]


# ── Individual Nodes (Preserving exact logic from agent.ipynb) ────────────

def domain_classifier_node(state: GraphState) -> GraphState:
    """Classifies if query is law-related or general."""
    query = state["query"].lower()
    is_legal = any(k in query for k in LAW_KEYWORDS)
    if not is_legal:
        procedural_tokens = [
            "document", "documents", "evidence", "proof", "fee", "fees", "cost", "costs",
            "limitation", "period", "deadline", "procedure", "process", "step", "steps",
            "action", "notice", "complaint", "file", "filing", "claim", "sue", "remedy",
            "relief", "section", "act", "clause", "provision", "jurisdiction", "rights",
            "what should i do", "what can i do", "how to proceed", "next step",
            "police", "court", "lawyer", "advocate", "appeal", "statute", "tribunal"
        ]
        if any(pt in query for pt in procedural_tokens):
            is_legal = True
    domain = "law" if is_legal else "general"
    return {**state, "domain": domain}


def domain_router(state: GraphState) -> str:
    return "law" if state.get("domain") == "law" else "general"


def general_llm_node(state: GraphState) -> GraphState:
    """Direct LLM response for non-law queries."""
    prompt = GENERAL_RESPONSE.format(query=state["query"])
    answer = call_llm(
        prompt,
        api_key=state.get("groq_api_key"),
        openrouter_api_key=state.get("openrouter_api_key"),
        openrouter_model=state.get("openrouter_model"),
    )
    if not answer or "Demo Mode" in answer or "Groq API key is not configured" in answer:
        answer = f"I am your Indian Legal Assistant (powered by Qwen). For your inquiry regarding '{state['query']}': Please feel free to ask any question regarding Indian criminal or civil law, procedures, or citizen rights."

    return {
        **state,
        "final_answer": answer,
        "source": "GENERAL",
        "structured_sections": {
            "legal_summary": answer,
            "disclaimer": "This is a general inquiry response, not formal legal advice.",
            "lawyer_needed": "Not Needed",
            "lawyer_reason": "General question.",
            "relevant_laws": [],
            "action_roadmap": [],
            "important_rights": [],
            "final_note": "Feel free to ask any Indian law questions if you need legal guidance.",
        },
    }


def refine_query_node(state: GraphState) -> GraphState:
    """Converts user situation into a precise Indian legal search query."""
    prompt = QUERY_REFINER.format(query=state["query"])
    refined = call_llm(
        prompt=prompt,
        system_prompt="You are an Indian Legal Research Expert. Output only the refined query string.",
        temperature=0.2,
        api_key=state.get("groq_api_key"),
        openrouter_api_key=state.get("openrouter_api_key"),
        openrouter_model=state.get("openrouter_model"),
    )
    # Fallback to original query if LLM fails or repeats preamble
    refined = refined.strip().strip('"\'')
    if not refined or "Demo Mode" in refined:
        refined = f"{state['query']} under Indian Law"
    return {**state, "refined_query": refined}


def risk_classifier_node(state: GraphState) -> GraphState:
    """Classifies risk level: LOW / MEDIUM / HIGH with reasoning."""
    prompt = RISK_CLASSIFIER.format(query=state["query"])
    raw_response = call_llm(
        prompt=prompt,
        system_prompt="You are a Legal Risk Assessment Expert.",
        temperature=0.2,
        api_key=state.get("groq_api_key"),
        openrouter_api_key=state.get("openrouter_api_key"),
        openrouter_model=state.get("openrouter_model"),
    )
    level, reason = parse_risk_response(raw_response)
    return {**state, "risk": level, "risk_reason": reason}


def rag_node(state: GraphState) -> GraphState:
    """Retrieves top legal chunks from Pinecone and evaluates confidence."""
    search_query = state.get("refined_query") or state["query"]
    rag_result = search_rag(
        query=search_query,
        top_k=RAG_TOP_K,
        api_key=state.get("pinecone_api_key"),
    )

    score = rag_result.get("confidence_score", rag_result.get("score", 0.0))
    status = rag_result.get("status", "unknown")
    chunks = rag_result.get("chunks", [])
    fallback_reason = None

    if status == "key_missing":
        fallback_reason = "Pinecone API key is not configured"
    elif status == "connection_error":
        fallback_reason = "Pinecone vector index is unreachable or connection failed"
    elif status == "no_documents_found" or not rag_result.get("context"):
        fallback_reason = "No relevant statutory documents found in Pinecone"
    elif score < RAG_CONFIDENCE_THRESHOLD:
        fallback_reason = f"Retrieval confidence ({score:.2f}) falls below threshold ({RAG_CONFIDENCE_THRESHOLD:.2f})"

    source = "pinecone_rag" if not fallback_reason else "huggingface_fallback"

    return {
        **state,
        "context": rag_result.get("context", ""),
        "score": score,
        "confidence_score": score,
        "matched_cluster": rag_result.get("cluster_id"),
        "retrieved_chunks": chunks,
        "retrieved_documents_count": len(chunks),
        "source": source,
        "answer_source": source,
        "fallback_reason": fallback_reason,
        "response_status": "success" if not fallback_reason else "fallback_triggered",
    }


def rag_confidence_router(state: GraphState) -> str:
    """Routes to Hugging Face / OpenRouter fallback if confidence is low, else roadmap."""
    if state.get("fallback_reason") or state.get("confidence_score", 0.0) < RAG_CONFIDENCE_THRESHOLD:
        return "hf_fallback"
    return "roadmap"


def hf_fallback_node(state: GraphState) -> GraphState:
    """Answers using the specialized Indian Legal Fallback Model."""
    reason = state.get("fallback_reason") or "Pinecone retrieval confidence below threshold"
    fallback_res = generate_indian_legal_fallback(
        query=state["query"],
        refined_query=state.get("refined_query"),
        risk=state.get("risk", "medium"),
        style=state.get("style", "detailed"),
        fallback_reason=reason,
        openrouter_api_key=state.get("openrouter_api_key"),
        openrouter_model=state.get("openrouter_model"),
        hf_api_key=state.get("hf_api_key"),
        hf_model=state.get("hf_model"),
    )
    return {
        **state,
        "final_answer": fallback_res["raw_answer"],
        "structured_sections": fallback_res["structured_sections"],
        "answer_source": fallback_res["answer_source"],
        "response_status": fallback_res["response_status"],
    }


def rag_router(state: GraphState) -> str:
    """Routes to SerpAPI if RAG score is below threshold; otherwise to roadmap."""
    score = state.get("score", 0.0)
    threshold = RAG_THRESHOLD
    if score < threshold or not state.get("context"):
        return "serp"
    return "roadmap"


def serp_node(state: GraphState) -> GraphState:
    """Fetches live web context restricted to Indian Law."""
    refined = state.get("refined_query") or state["query"]
    search_query = f"{refined} under Indian Law BNS IPC"
    serp_result = search_web(
        query=search_query,
        api_key=state.get("serpapi_api_key"),
    )

    existing = state.get("context", "")
    web_text = serp_result.get("web_context", "")
    new_context = f"{existing}\n\n=== INDIAN WEB CONTEXT ===\n{web_text}".strip()

    return {
        **state,
        "context": new_context,
        "web_results": serp_result.get("results", []),
        "source": "WEB",
    }


def roadmap_node(state: GraphState) -> GraphState:
    """Generates a practical legal roadmap based on context."""
    context = state.get("context") or "No specific statute context found."
    prompt = ROADMAP_GENERATOR.format(query=state["query"], context=context)
    roadmap = call_llm(
        prompt=prompt,
        system_prompt="You are a legal procedure advisor for Indian citizens.",
        temperature=0.3,
        api_key=state.get("groq_api_key"),
        openrouter_api_key=state.get("openrouter_api_key"),
        openrouter_model=state.get("openrouter_model"),
    )
    return {**state, "roadmap": roadmap}


def legal_expert_node(state: GraphState) -> GraphState:
    """Synthesizes all context into structured Indian legal advice."""
    style = state.get("style", "detailed")
    style_instruction = STYLE_SIMPLE if style == "simple" else STYLE_DETAILED
    detail_level = "simple, plain-language" if style == "simple" else "thorough and well-reasoned"

    context = state.get("context") or "No direct statute context found."
    risk_level = (state.get("risk") or "low").upper()

    prompt = LEGAL_EXPERT.format(
        context=context,
        query=state["query"],
        risk=risk_level,
        style_instruction=style_instruction,
        detail_level=detail_level,
    )

    final_answer = call_llm(
        prompt=prompt,
        system_prompt="You are a STRICT Indian legal assistant. Follow the exact section headers requested.",
        temperature=0.3,
        api_key=state.get("groq_api_key"),
        openrouter_api_key=state.get("openrouter_api_key"),
        openrouter_model=state.get("openrouter_model"),
    )

    structured = parse_structured_answer(final_answer)
    return {
        **state,
        "final_answer": final_answer,
        "structured_sections": structured,
    }


# ── Build LangGraph DAG (Backward Compatible) ──────────────────────────────

def build_graph() -> Any:
    """Compiles and returns the LangGraph workflow DAG with Fallback Routing."""
    if not _HAS_LANGGRAPH or StateGraph is None:
        return None
    graph = StateGraph(GraphState)

    graph.add_node("domain", domain_classifier_node)
    graph.add_node("general", general_llm_node)
    graph.add_node("refine", refine_query_node)
    graph.add_node("risk", risk_classifier_node)
    graph.add_node("rag", rag_node)
    graph.add_node("serp", serp_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("final", legal_expert_node)
    graph.add_node("hf_fallback", hf_fallback_node)

    graph.set_entry_point("domain")

    graph.add_conditional_edges(
        "domain",
        domain_router,
        {"law": "refine", "general": "general"},
    )
    graph.add_edge("refine", "risk")
    graph.add_edge("risk", "rag")

    graph.add_conditional_edges(
        "rag",
        rag_confidence_router,
        {"roadmap": "roadmap", "hf_fallback": "hf_fallback"},
    )

    graph.add_edge("roadmap", "final")
    graph.add_edge("final", END)
    graph.add_edge("hf_fallback", END)
    graph.add_edge("general", END)

    return graph.compile()


# Global compiled app handle for direct invocation
langgraph_app = build_graph()


# ── Interactive Generator Pipeline for Streamlit ──────────────────────────

def run_query(
    query: str,
    style: str = "detailed",
    rag_threshold: float = RAG_CONFIDENCE_THRESHOLD,
    top_k: int = RAG_TOP_K,
    groq_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    serpapi_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None,
    hf_api_key: Optional[str] = None,
    hf_model: Optional[str] = None,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """
    Step-by-step pipeline execution yielding progress events.

    Yields:
      {"node": str, "status": "running"|"done", "data": dict}

    Final return/yield:
      {"node": "complete", "status": "done", "result": dict}
    """
    start_time = time.time()

    state: GraphState = {
        "query": query,
        "domain": "",
        "refined_query": "",
        "risk": "low",
        "risk_reason": "",
        "context": "",
        "score": 0.0,
        "confidence_score": 0.0,
        "matched_cluster": None,
        "retrieved_chunks": [],
        "retrieved_documents_count": 0,
        "web_results": [],
        "source": "pinecone_rag",
        "answer_source": "pinecone_rag",
        "fallback_reason": None,
        "response_status": "in_progress",
        "roadmap": "",
        "final_answer": "",
        "structured_sections": {},
        "style": style,
        "groq_api_key": groq_api_key,
        "pinecone_api_key": pinecone_api_key,
        "serpapi_api_key": serpapi_api_key,
        "openrouter_api_key": openrouter_api_key or OPENROUTER_API_KEY,
        "openrouter_model": openrouter_model or OPENROUTER_MODEL,
        "hf_api_key": hf_api_key or HUGGINGFACE_API_KEY,
        "hf_model": hf_model or HUGGINGFACE_MODEL,
    }

    # 1. Domain classification
    yield {
        "node": "domain",
        "status": "running",
        "title": "Classifying Domain",
        "message": "Checking if query pertains to Indian law...",
    }
    t0 = time.time()
    state = domain_classifier_node(state)
    domain_time = round(time.time() - t0, 2)
    yield {
        "node": "domain",
        "status": "done",
        "title": f"Domain: {state['domain'].upper()}",
        "message": f"Identified as {state['domain']} domain ({domain_time}s)",
        "data": {"domain": state["domain"], "duration": domain_time},
    }

    if state["domain"] != "law":
        # General response branch
        yield {
            "node": "general",
            "status": "running",
            "title": "Generating Answer",
            "message": "Providing direct response for general inquiry...",
        }
        t0 = time.time()
        state = general_llm_node(state)
        gen_time = round(time.time() - t0, 2)

        followups = generate_followup_questions(
            query=query,
            summary=state["final_answer"],
            api_key=groq_api_key,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model,
        )

        total_time = round(time.time() - start_time, 2)
        final_result = {
            "query": query,
            "refined_query": query,
            "domain": "general",
            "risk": "low",
            "risk_reason": "Non-legal matter",
            "source": "general_llm",
            "answer_source": "general_llm",
            "confidence_score": 1.0,
            "fallback_reason": None,
            "retrieved_documents_count": 0,
            "response_status": "success",
            "matched_cluster": None,
            "retrieved_chunks": [],
            "web_results": [],
            "answer_sections": state["structured_sections"],
            "raw_answer": state["final_answer"],
            "lawyer_needed": "Not Needed",
            "lawyer_reason": "General question.",
            "followups": followups,
            "latency": total_time,
        }
        yield {
            "node": "complete",
            "status": "done",
            "title": "Response Ready",
            "message": f"Completed in {total_time}s",
            "result": final_result,
        }
        return final_result

    # 2. Query refinement
    yield {
        "node": "refine",
        "status": "running",
        "title": "Refining Legal Query",
        "message": "Mapping situation to Indian statutes (IPC/BNS, CrPC/BNSS)...",
    }
    t0 = time.time()
    state = refine_query_node(state)
    refine_time = round(time.time() - t0, 2)
    yield {
        "node": "refine",
        "status": "done",
        "title": "Query Refined",
        "message": f"Search terms: '{state['refined_query'][:60]}...' ({refine_time}s)",
        "data": {"refined_query": state["refined_query"], "duration": refine_time},
    }

    # 3. Risk classification
    yield {
        "node": "risk",
        "status": "running",
        "title": "Assessing Legal Risk",
        "message": "Evaluating urgency, rights violation, or imminent arrest danger...",
    }
    t0 = time.time()
    state = risk_classifier_node(state)
    risk_time = round(time.time() - t0, 2)
    yield {
        "node": "risk",
        "status": "done",
        "title": f"Risk Level: {state['risk'].upper()}",
        "message": f"{state['risk_reason']} ({risk_time}s)",
        "data": {"risk": state["risk"], "reason": state["risk_reason"], "duration": risk_time},
    }

    # 4. RAG Node: Vector Retrieval
    yield {
        "node": "rag",
        "status": "running",
        "title": "Querying Knowledge Base (Pinecone)",
        "message": "Scanning Pinecone vector clusters for Indian statutory context...",
    }
    t0 = time.time()
    state = rag_node(state)
    rag_time = round(time.time() - t0, 2)
    cluster_str = f"Cluster {state['matched_cluster']}" if state['matched_cluster'] is not None else "No cluster"

    yield {
        "node": "rag",
        "status": "done",
        "title": f"Vector Search: Confidence {state['confidence_score']:.2f}",
        "message": f"Retrieved {len(state['retrieved_chunks'])} chunks ({cluster_str}) in {rag_time}s",
        "data": {
            "score": state["confidence_score"],
            "cluster_id": state["matched_cluster"],
            "chunks_count": len(state["retrieved_chunks"]),
            "duration": rag_time,
            "fallback_reason": state.get("fallback_reason"),
        },
    }

    # 5. Routing Check: Confidence & Availability Evaluation
    if state.get("fallback_reason") or state["confidence_score"] < rag_threshold:
        reason = state.get("fallback_reason") or f"Confidence ({state['confidence_score']:.2f}) < threshold ({rag_threshold:.2f})"
        yield {
            "node": "fallback",
            "status": "running",
            "title": "Indian Legal Fallback Model (Qwen / OpenRouter)",
            "message": f"Routing query to Indian Law Fallback Engine: {reason}...",
        }
        t0 = time.time()
        state = hf_fallback_node(state)
        fallback_time = round(time.time() - t0, 2)

        yield {
            "node": "fallback",
            "status": "done",
            "title": "Indian Law Synthesis Complete",
            "message": f"Specialized Indian legal advisory generated ({fallback_time}s)",
            "data": {"duration": fallback_time, "source": "huggingface_fallback"},
        }

    else:
        # High Confidence: Use Pinecone RAG + Procedural Roadmap + Legal Expert Node
        yield {
            "node": "roadmap",
            "status": "running",
            "title": "Building Action Roadmap",
            "message": "Formulating procedural legal roadmap from verified Pinecone context...",
        }
        t0 = time.time()
        state = roadmap_node(state)
        roadmap_time = round(time.time() - t0, 2)
        yield {
            "node": "roadmap",
            "status": "done",
            "title": "Roadmap Prepared",
            "message": f"Action plan formulated ({roadmap_time}s)",
            "data": {"duration": roadmap_time},
        }

        yield {
            "node": "final",
            "status": "running",
            "title": "Synthesizing Legal Advice",
            "message": "Generating structured response with statutory citations & rights...",
        }
        t0 = time.time()
        state = legal_expert_node(state)
        expert_time = round(time.time() - t0, 2)
        yield {
            "node": "final",
            "status": "done",
            "title": "Synthesis Complete",
            "message": f"Advice drafted ({expert_time}s)",
            "data": {"duration": expert_time},
        }

    # Follow-up questions
    sections = state.get("structured_sections", {})
    summary_text = sections.get("legal_summary", state["final_answer"])
    followups = generate_followup_questions(
        query=query,
        summary=summary_text,
        api_key=groq_api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )

    total_time = round(time.time() - start_time, 2)
    final_result = {
        "query": query,
        "refined_query": state.get("refined_query", query),
        "domain": state.get("domain", "law"),
        "risk": state.get("risk", "low"),
        "risk_reason": state.get("risk_reason", ""),
        "source": state.get("answer_source", "pinecone_rag"),
        "answer_source": state.get("answer_source", "pinecone_rag"),
        "confidence_score": round(state.get("confidence_score", 0.0), 2),
        "fallback_reason": state.get("fallback_reason"),
        "retrieved_documents_count": state.get("retrieved_documents_count", len(state.get("retrieved_chunks", []))),
        "response_status": state.get("response_status", "success"),
        "matched_cluster": state.get("matched_cluster"),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "web_results": state.get("web_results", []),
        "answer_sections": state.get("structured_sections", {}),
        "raw_answer": state.get("final_answer", ""),
        "lawyer_needed": sections.get("lawyer_needed", "Optional"),
        "lawyer_reason": sections.get("lawyer_reason", ""),
        "followups": followups,
        "latency": total_time,
    }

    yield {
        "node": "complete",
        "status": "done",
        "title": "Analysis Ready",
        "message": f"All nodes executed in {total_time}s",
        "result": final_result,
    }

    return final_result

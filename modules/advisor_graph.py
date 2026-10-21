import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from modules.advisor_tools import AcademicAdvisorTools
from modules.hybrid_retriever import HybridRetriever


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"


class RouteDecision(BaseModel):
    use_rag: bool = Field(description="Use unstructured university documents.")
    use_course_tool: bool = Field(description="Use structured course catalog data.")
    use_student_tool: bool = Field(description="Use structured student profile data.")


class AdvisorState(TypedDict, total=False):
    query: str
    chat_history: List
    routes: List[str]
    retrieved_docs: List
    course_data: Optional[Dict]
    student_data: Optional[Dict]
    evidence: List[Dict]
    answer: str


ROUTER_SYSTEM_PROMPT = """
You route academic advising questions.

Choose any combination of:
- RAG: university policy, admissions, registration rules, official documents, broad NED information.
- Course tool: course credits, course descriptions, prerequisites, courses that require another course.
- Student tool: completed courses, eligibility for a course, degree progress, remaining credits.

Return structured booleans only.
"""


ANSWER_SYSTEM_PROMPT = """
You are an academic advising assistant for NED University.

Answer only from the provided evidence. If evidence is missing, say what is missing.
Use a direct, student-friendly tone. Include citations for document evidence using the citation labels provided.
For structured course or student data, state that it comes from the academic database.
"""


def get_openai_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")


def get_groq_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


def has_llm_key() -> bool:
    return bool(get_openai_key() or get_groq_key())


def make_llm(temperature: float = 0.0):
    groq_key = get_groq_key()
    if groq_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
            api_key=groq_key,
        )

    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "temperature": temperature,
    }
    openai_key = get_openai_key()
    if openai_key:
        kwargs["api_key"] = openai_key
    return ChatOpenAI(**kwargs)


def heuristic_route(query: str) -> RouteDecision:
    q = query.lower()

    has_course_code = bool(
        re.search(r"\b(CS|SE|MATH|EE|HU|MG)[-\s]?\d{3}\b", query, flags=re.IGNORECASE)
    )
    course_keywords = [
        "course",
        "prerequisite",
        "pre requisite",
        "machine learning",
        "artificial intelligence",
        "data structures",
    ]
    student_keywords = [
        "student",
        "completed",
        "eligible",
        "can i",
        "can student",
        "take",
        "remaining",
        "degree",
        "progress",
        "credits still need",
        "credits remaining",
    ]
    rag_keywords = [
        "policy",
        "rule",
        "admission",
        "registration",
        "deadline",
        "fee",
        "prospectus",
        "university",
        "department",
        "document",
        "override",
    ]

    use_student = any(keyword in q for keyword in student_keywords)
    use_course = has_course_code or (
        any(keyword in q for keyword in course_keywords)
        and not ("degree" in q or "remaining" in q or "still need" in q)
    )
    use_rag = any(keyword in q for keyword in rag_keywords)

    if not (use_course or use_student or use_rag):
        use_rag = True

    return RouteDecision(
        use_rag=use_rag,
        use_course_tool=use_course,
        use_student_tool=use_student,
    )


def route_with_llm(query: str) -> RouteDecision:
    if not has_llm_key():
        return heuristic_route(query)

    try:
        llm = make_llm().with_structured_output(RouteDecision)
        return llm.invoke(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", query),
            ]
        )
    except Exception:
        return heuristic_route(query)


@lru_cache(maxsize=1)
def get_tools() -> AcademicAdvisorTools:
    return AcademicAdvisorTools()


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    use_pinecone = bool(os.getenv("PINECONE_API_KEY") and get_openai_key())
    use_reranker = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
    return HybridRetriever.from_pdf_dir(
        data_path=str(DATA_PATH),
        use_pinecone=use_pinecone,
        use_reranker=use_reranker,
    )


def citation_for_doc(doc) -> str:
    metadata = doc.metadata or {}
    source = metadata.get("source", "University document")
    page = metadata.get("page")
    if page and page != "unknown":
        return f"{source}, p. {page}"
    return source


def route_query(state: AdvisorState) -> AdvisorState:
    decision = route_with_llm(state["query"])
    routes = []
    if decision.use_rag:
        routes.append("rag_search")
    if decision.use_course_tool:
        routes.append("course_tool")
    if decision.use_student_tool:
        routes.append("student_tool")
    if not routes:
        routes.append("rag_search")
    return {"routes": routes}


def choose_routes(state: AdvisorState) -> List[str]:
    return state.get("routes", ["rag_search"])


def rag_search(state: AdvisorState) -> AdvisorState:
    try:
        retriever = get_hybrid_retriever()
        docs = retriever.retrieve(state["query"])
        return {"retrieved_docs": docs}
    except Exception as exc:
        return {
            "retrieved_docs": [],
            "evidence": [
                {
                    "type": "system",
                    "source": "retrieval",
                    "content": f"RAG retrieval failed: {exc}",
                }
            ],
        }


def course_tool(state: AdvisorState) -> AdvisorState:
    tools = get_tools()
    query = state["query"]
    course_code = tools.infer_course_code(query)

    if course_code:
        course_data = tools.get_course_info(course_code)
    else:
        course_data = {
            "found": True,
            "search_results": tools.search_courses(query),
            "message": "No exact course code was found, so course search was used.",
        }

    return {"course_data": course_data}


def student_tool(state: AdvisorState) -> AdvisorState:
    tools = get_tools()
    query = state["query"]
    student_id = tools.infer_student_id(query)
    course_code = tools.infer_course_code(query)

    student_data = {
        "profile": tools.get_student_profile(student_id),
        "degree_progress": tools.get_degree_progress(student_id),
    }
    if course_code:
        student_data["prerequisite_check"] = tools.check_prerequisites(student_id, course_code)

    return {"student_data": student_data}


def aggregate_context(state: AdvisorState) -> AdvisorState:
    evidence = list(state.get("evidence", []))

    for doc in state.get("retrieved_docs", []):
        evidence.append(
            {
                "type": "document",
                "source": doc.metadata.get("source", "University document"),
                "citation": citation_for_doc(doc),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
            }
        )

    if state.get("course_data"):
        evidence.append(
            {
                "type": "course_database",
                "source": "academic.db",
                "content": json.dumps(state["course_data"], indent=2),
            }
        )

    if state.get("student_data"):
        evidence.append(
            {
                "type": "student_database",
                "source": "academic.db",
                "content": json.dumps(state["student_data"], indent=2),
            }
        )

    return {"evidence": evidence}


def format_evidence(evidence: List[Dict]) -> str:
    formatted = []
    for index, item in enumerate(evidence, start=1):
        label = item.get("citation") or item.get("source") or f"Evidence {index}"
        formatted.append(f"[{index}] {label}\n{item.get('content', '')}")
    return "\n\n".join(formatted)


def fallback_answer(state: AdvisorState) -> str:
    course_data = state.get("course_data")
    student_data = state.get("student_data")
    document_evidence = [
        item for item in state.get("evidence", [])
        if item.get("type") == "document"
    ]

    if course_data and course_data.get("found") and course_data.get("course_code"):
        course_line = (
            f"{course_data['course_code']} is {course_data['course_name']} "
            f"({course_data['credits']} credits)."
        )
        prerequisites = course_data.get("prerequisites", [])
        if prerequisites:
            prereq_text = ", ".join(
                f"{item['prerequisite_code']} ({item['course_name']})"
                for item in prerequisites
            )
            course_line += f" Its prerequisites are: {prereq_text}."
        else:
            course_line += " It has no listed prerequisites."

        if student_data:
            check = student_data.get("prerequisite_check")
            if check:
                if check.get("eligible"):
                    course_line += (
                        f" Student {check['student_id']} has satisfied the listed "
                        "prerequisites, so they are eligible based on the academic database."
                    )
                else:
                    missing = ", ".join(check.get("missing_prerequisites", []))
                    course_line += (
                        f" Student {check['student_id']} is not eligible yet based on "
                        f"the academic database because they are missing: {missing}."
                    )

        if document_evidence:
            citations = ", ".join(
                item.get("citation", item.get("source", "document"))
                for item in document_evidence[:3]
            )
            course_line += f" Relevant document evidence found: {citations}."

        return course_line

    if student_data and student_data.get("degree_progress"):
        progress = student_data["degree_progress"]
        if progress.get("found"):
            categories = progress.get("categories", [])
            category_text = "; ".join(
                f"{item['category']}: {item['remaining_credits']} remaining"
                for item in categories
            )
            return (
                f"Student {progress['student_id']} is in {progress['program']}. "
                f"They have completed {progress['total_completed_credits']} credits "
                f"toward the tracked requirements and still need "
                f"{progress['total_remaining_credits']} credits. "
                f"Breakdown: {category_text}."
            )

    evidence = state.get("evidence", [])
    if not evidence:
        return "I do not have enough evidence to answer this yet."

    lines = ["I found relevant evidence, but I need the LLM step to produce a polished answer:"]
    for item in evidence[:4]:
        label = item.get("citation") or item.get("source")
        content = item.get("content", "")
        preview = content.replace("\n", " ")[:220]
        lines.append(f"- {label}: {preview}")
    return "\n".join(lines)


def generate_answer(state: AdvisorState) -> AdvisorState:
    if not has_llm_key():
        return {"answer": fallback_answer(state)}

    try:
        llm = make_llm(temperature=0.2)
        response = llm.invoke(
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                (
                    "human",
                    f"Question:\n{state['query']}\n\nEvidence:\n{format_evidence(state.get('evidence', []))}",
                ),
            ]
        )
        return {"answer": response.content}
    except Exception as exc:
        return {
            "answer": fallback_answer(state),
            "llm_error": str(exc),
        }


class SimpleAdvisorGraph:
    def invoke(self, state: AdvisorState) -> AdvisorState:
        state = {**state, **route_query(state)}
        if "rag_search" in state["routes"]:
            state = {**state, **rag_search(state)}
        if "course_tool" in state["routes"]:
            state = {**state, **course_tool(state)}
        if "student_tool" in state["routes"]:
            state = {**state, **student_tool(state)}
        state = {**state, **aggregate_context(state)}
        state = {**state, **generate_answer(state)}
        return state


@lru_cache(maxsize=1)
def get_advisor_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return SimpleAdvisorGraph()

    graph = StateGraph(AdvisorState)
    graph.add_node("route_query", route_query)
    graph.add_node("rag_search", rag_search)
    graph.add_node("course_tool", course_tool)
    graph.add_node("student_tool", student_tool)
    graph.add_node("aggregate_context", aggregate_context)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges(
        "route_query",
        choose_routes,
        {
            "rag_search": "rag_search",
            "course_tool": "course_tool",
            "student_tool": "student_tool",
        },
    )
    graph.add_edge("rag_search", "aggregate_context")
    graph.add_edge("course_tool", "aggregate_context")
    graph.add_edge("student_tool", "aggregate_context")
    graph.add_edge("aggregate_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def run_advisor(query: str, chat_history: Optional[List] = None) -> Dict:
    graph = get_advisor_graph()
    result = graph.invoke(
        {
            "query": query,
            "chat_history": chat_history or [],
            "retrieved_docs": [],
            "course_data": None,
            "student_data": None,
            "evidence": [],
        }
    )
    return result
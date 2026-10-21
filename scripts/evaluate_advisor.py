import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.advisor_graph import (
    aggregate_context,
    course_tool,
    fallback_answer,
    heuristic_route,
    rag_search,
    run_advisor,
    student_tool,
)


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "eval_questions_balanced_24.json"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[*_\x60]", "", text)
    text = " ".join(text.split())

    # Treat 50%, 50 %, LaTeX percentages and 50 percent alike.
    text = re.sub(
        r"(?<=\d)\s*(?:\\?%|percent\b)",
        "%",
        text,
    )

    return text


def contains_term(text: str, term: str) -> bool:
    pattern = (
        r"(?<!\w)"
        + re.escape(normalize(term))
        + r"(?!\w)"
    )
    return bool(re.search(pattern, normalize(text)))


def answer_checks(text: str, case: Dict) -> tuple:
    groups = [[term] for term in case.get("must_include", [])]
    groups += case.get("answer_term_groups", [])

    missing = [
        group
        for group in groups
        if not any(contains_term(text, term) for term in group)
    ]

    return (not missing if groups else None), missing


def evidence_labels(evidence: List[Dict]) -> List[str]:
    labels = []

    for item in evidence:
        if item.get("type") == "system":
            continue

        labels.append(str(item.get("source", "")))
        labels.append(str(item.get("citation", "")))

    return [label for label in labels if label]


def has_expected_sources(
    evidence: List[Dict],
    expected_sources: Iterable[str],
) -> bool:
    labels = " | ".join(evidence_labels(evidence)).lower()
    return all(source.lower() in labels for source in expected_sources)


def run_without_llm(question: str) -> Dict:
    state = {
        "query": question,
        "chat_history": [],
        "retrieved_docs": [],
        "course_data": None,
        "student_data": None,
        "evidence": [],
    }

    decision = heuristic_route(question)
    routes = []

    if decision.use_rag:
        routes.append("rag_search")
    if decision.use_course_tool:
        routes.append("course_tool")
    if decision.use_student_tool:
        routes.append("student_tool")

    state["routes"] = routes or ["rag_search"]

    if "rag_search" in state["routes"]:
        state = {**state, **rag_search(state)}
    if "course_tool" in state["routes"]:
        state = {**state, **course_tool(state)}
    if "student_tool" in state["routes"]:
        state = {**state, **student_tool(state)}

    state = {**state, **aggregate_context(state)}
    state["answer"] = fallback_answer(state)

    return state


def evaluate_case(case: Dict, use_llm: bool = False) -> Dict:
    started_at = time.perf_counter()

    result = (
        run_advisor(case["question"])
        if use_llm
        else run_without_llm(case["question"])
    )

    latency_ms = round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )

    expected_routes = set(case.get("expected_routes", []))
    actual_routes = set(result.get("routes", []))
    evidence = result.get("evidence", [])
    expected_sources = case.get("expected_sources", [])

    missing_sources = [
        source
        for source in expected_sources
        if not has_expected_sources(evidence, [source])
    ]

    answer = result.get("answer", "")
    term_pass, missing_groups = answer_checks(answer, case)

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_routes": sorted(expected_routes),
        "actual_routes": result.get("routes", []),
        "route_exact": actual_routes == expected_routes,
        "source_hit": not missing_sources if expected_sources else None,
        "expected_sources": expected_sources,
        "actual_source_labels": evidence_labels(evidence),
        "missing_sources": missing_sources,
        "answer_terms_pass": term_pass,
        "missing_answer_term_groups": missing_groups,
        "llm_error": result.get("llm_error"),
        "answer": answer,
        "evidence": [
            {
                key: str(item.get(key, ""))
                for key in ("type", "source", "citation", "content")
            }
            for item in evidence
        ],
        "source_references": case.get("source_references", []),
        "latency_ms": latency_ms,
        "answer_preview": answer.replace("\n", " ")[:240],
    }


def pct(numerator: int, denominator: int):
    if denominator == 0:
        return None

    return round((numerator / denominator) * 100, 2)


def summarize(results: List[Dict]) -> Dict:
    answer_checked = [
        row
        for row in results
        if row["answer_terms_pass"] is not None
    ]

    source_checked = [
        row
        for row in results
        if row["source_hit"] is not None
    ]

    matched_source_count = sum(
        len(row["expected_sources"]) - len(row["missing_sources"])
        for row in source_checked
    )

    expected_source_count = sum(
        len(row["expected_sources"])
        for row in source_checked
    )

    return {
        "total_cases": len(results),
        "route_exact_match": pct(
            sum(row["route_exact"] for row in results),
            len(results),
        ),
        "source_hit_rate": pct(
            sum(row["source_hit"] for row in source_checked),
            len(source_checked),
        ),
        "source_checked_cases": len(source_checked),
        "expected_source_coverage": pct(
            matched_source_count,
            expected_source_count,
        ),
        "llm_error_cases": sum(
            bool(row.get("llm_error"))
            for row in results
        ),
        "answer_term_pass_rate": pct(
            sum(row["answer_terms_pass"] for row in answer_checked),
            len(answer_checked),
        ),
        "answer_checked_cases": len(answer_checked),
        "avg_latency_ms": (
            round(mean(row["latency_ms"] for row in results), 2)
            if results
            else 0
        ),
    }


def print_report(summary: Dict, results: List[Dict]) -> None:
    print("\nAdvisor validation metrics")
    print("=" * 28)

    for key, value in summary.items():
        displayed = (
            value
            if value is not None
            else "N/A (no checked cases)"
        )
        print(f"{key}: {displayed}")

    failures = [
        row
        for row in results
        if not row["route_exact"]
        or row["source_hit"] is False
        or row.get("llm_error")
        or row["answer_terms_pass"] is False
    ]

    if not failures:
        print(
            "\nAll enabled automatic checks passed. "
            "Review answer facts and citations manually."
        )
        return

    print("\nFailures")
    print("=" * 8)

    for row in failures:
        print(f"- {row['id']}")
        print(f"  expected_routes: {row['expected_routes']}")
        print(f"  actual_routes:   {row['actual_routes']}")
        print(f"  source_hit:      {row['source_hit']}")
        print(f"  missing_sources: {row['missing_sources']}")
        print(f"  actual_sources:  {row['actual_source_labels']}")
        print(f"  missing_terms:   {row['missing_answer_term_groups']}")

        if row.get("llm_error"):
            print(f"  llm_error:       {row['llm_error']}")

        print(f"  answer_terms:    {row['answer_terms_pass']}")
        print(f"  answer_preview:  {row['answer_preview']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the academic advisor pipeline."
    )

    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file with full answers and evidence.",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help=(
            "Run full answer generation with the configured LLM. "
            "Default skips LLM calls."
        ),
    )

    args = parser.parse_args()

    cases = json.loads(
        args.eval_file.read_text(encoding="utf-8-sig")
    )

    if not args.use_llm:
        print(
            "LLM disabled: answer checks evaluate fallback text, "
            "not generated answers."
        )

    print(
        "Answer-term metrics measure lexical coverage, "
        "not semantic accuracy."
    )

    results = [
        evaluate_case(case, use_llm=args.use_llm)
        for case in cases
    ]

    summary = summarize(results)
    print_report(summary, results)

    if args.output:
        payload = {
            "summary": summary,
            "results": results,
        }

        args.output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()

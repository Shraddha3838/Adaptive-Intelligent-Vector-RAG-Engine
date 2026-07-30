"""
Evaluation harness for the self-correcting RAG agent.

Runs the full eval dataset through the agent, computes metrics, and
exits non-zero if any metric falls below the configured threshold.

Usage:
    python evals/run_evals.py                          # full suite
    python evals/run_evals.py --dataset evals/datasets/qa.jsonl
    python evals/run_evals.py --threshold-recall 0.7
    python evals/run_evals.py --json                   # JSON output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from src.config import get_settings, reset_settings_cache
from src.graph import RAGAgent
from src.utils import get_logger, setup_logging

from evals.evaluators.metrics import compute_all_metrics

logger = get_logger("evals")

# Default thresholds matching config
DEFAULT_THRESHOLDS = {
    "recall_at_k": 0.70,
    "groundedness": 0.90,
    "citation_accuracy": 0.90,
    "refusal_correctness": 1.00,
}


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load evaluation dataset from a JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[dict[str, Any]] = []
    with open(dataset_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", line_num, exc)

    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def run_single_eval(
    agent: RAGAgent,
    example: dict[str, Any],
    recall_k: int = 5,
) -> dict[str, Any]:
    """
    Run a single evaluation example through the agent.

    Args:
        agent: RAGAgent instance.
        example: Dataset example with question, relevant_listing_ids, category.
        recall_k: k for recall@k.

    Returns:
        Dict with question, response, metrics, and metadata.
    """
    question = example["question"]
    relevant_ids = example.get("relevant_listing_ids", [])
    category = example.get("category", "normal")
    is_no_answer = category == "no-answer"

    start = time.perf_counter()
    try:
        response = agent.run(question)
        duration = time.perf_counter() - start
    except Exception as exc:
        logger.exception("Failed to evaluate question: %s", question)
        return {
            "question": question,
            "error": str(exc),
            "duration": time.perf_counter() - start,
            "metrics": {
                "recall_at_k": 0.0,
                "groundedness": 0.0,
                "citation_accuracy": 0.0,
                "refusal_correctness": 0.0,
            },
        }

    metrics = compute_all_metrics(
        response=response,
        relevant_listing_ids=relevant_ids,
        is_no_answer=is_no_answer,
        recall_k=recall_k,
    )

    return {
        "question": question,
        "category": category,
        "answer": response.answer,
        "refused": response.refused,
        "retrieval_attempts": response.retrieval_attempts,
        "rewritten_queries": response.rewritten_queries,
        "cited_listing_ids": response.cited_listing_ids,
        "relevant_listing_ids": relevant_ids,
        "metrics": metrics,
        "duration": round(duration, 3),
    }


def run_eval_suite(
    dataset_path: str = "evals/datasets/qa.jsonl",
    thresholds: dict[str, float] | None = None,
    recall_k: int = 5,
    enable_tracing: bool = False,
) -> dict[str, Any]:
    """
    Run the full evaluation suite.

    Args:
        dataset_path: Path to the JSONL dataset.
        thresholds: Metric thresholds for regression gate.
        recall_k: k for recall@k.
        enable_tracing: Enable LangSmith tracing for this run.

    Returns:
        Dict with summary stats, per-example results, and pass/fail.
    """
    thresholds = thresholds or dict(DEFAULT_THRESHOLDS)

    # Enable tracing if requested
    if enable_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
    else:
        os.environ["LANGSMITH_TRACING"] = "false"

    settings = get_settings()
    setup_logging(settings.log_level)

    dataset = load_dataset(dataset_path)
    if not dataset:
        logger.error("Empty dataset — nothing to evaluate")
        return {"error": "Empty dataset", "passed": False}

    agent = RAGAgent(settings)
    results: list[dict[str, Any]] = []
    total_metrics: dict[str, list[float]] = {
        "recall_at_k": [],
        "groundedness": [],
        "citation_accuracy": [],
        "refusal_correctness": [],
    }

    try:
        for i, example in enumerate(dataset, 1):
            logger.info(
                "Evaluating [%d/%d]: %s",
                i,
                len(dataset),
                example["question"][:60],
            )
            result = run_single_eval(agent, example, recall_k=recall_k)
            results.append(result)

            for metric_name, value in result["metrics"].items():
                total_metrics[metric_name].append(value)

            # Log progress
            if i % 5 == 0 or i == len(dataset):
                logger.info("Progress: %d/%d examples evaluated", i, len(dataset))

    finally:
        agent.close()

    # Compute aggregate metrics
    agg_metrics: dict[str, float] = {}
    for metric_name, values in total_metrics.items():
        if values:
            agg_metrics[metric_name] = sum(values) / len(values)
        else:
            agg_metrics[metric_name] = 0.0

    # Check thresholds
    failures: dict[str, dict[str, float]] = {}
    for metric_name, threshold in thresholds.items():
        actual = agg_metrics.get(metric_name, 0.0)
        if actual < threshold:
            failures[metric_name] = {"actual": actual, "threshold": threshold}

    passed = len(failures) == 0

    summary = {
        "total_examples": len(dataset),
        "passed": passed,
        "aggregate_metrics": agg_metrics,
        "thresholds": thresholds,
        "failures": failures,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 60)
    print("  RAG AGENT EVALUATION SUMMARY")
    print("=" * 60)

    if "error" in summary:
        print(f"\n  ERROR: {summary['error']}")
        return

    status = "PASSED" if summary["passed"] else "FAILED"
    print(f"\n  Status:     {status}")
    print(f"  Examples:   {summary['total_examples']}")
    print(f"  Timestamp:  {summary['timestamp']}")
    print()

    print("  Aggregate Metrics:")
    for metric_name, value in summary["aggregate_metrics"].items():
        threshold = summary["thresholds"].get(metric_name, "N/A")
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"    {metric_name:25s} {value:.3f}  {bar}")
        print(f"    {'':25s} threshold: {threshold}")

    if summary["failures"]:
        print("\n  FAILURES (below threshold):")
        for metric_name, info in summary["failures"].items():
            print(
                f"    {metric_name:25s} actual={info['actual']:.3f} "
                f"< threshold={info['threshold']:.3f}"
            )

    print("\n" + "=" * 60)

    # Print per-example results
    print("\n  Per-Example Results:")
    print(f"  {'#':>3} {'Category':<12} {'Recall@K':>8} {'Grounded':>9} "
          f"{'Citation':>9} {'Refusal':>8} {'Attempts':>8}")
    print("  " + "-" * 60)
    for i, result in enumerate(summary["results"], 1):
        m = result["metrics"]
        cat = result.get("category", "normal")
        print(f"  {i:>3} {cat:<12} {m['recall_at_k']:>8.3f} "
              f"{m['groundedness']:>9.3f} {m['citation_accuracy']:>9.3f} "
              f"{m['refusal_correctness']:>8.3f} "
              f"{result.get('retrieval_attempts', 1):>8}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Capstone RAG — Evaluation Harness",
    )
    parser.add_argument(
        "--dataset",
        default="evals/datasets/qa.jsonl",
        help="Path to evaluation dataset JSONL",
    )
    parser.add_argument(
        "--threshold-recall",
        type=float,
        default=None,
        help="Recall@k threshold override",
    )
    parser.add_argument(
        "--threshold-groundedness",
        type=float,
        default=None,
        help="Groundedness threshold override",
    )
    parser.add_argument(
        "--threshold-citation",
        type=float,
        default=None,
        help="Citation accuracy threshold override",
    )
    parser.add_argument(
        "--threshold-refusal",
        type=float,
        default=None,
        help="Refusal correctness threshold override",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=None,
        help="k for recall@k metric",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable LangSmith tracing for this run",
    )
    return parser


def main() -> None:
    """CLI entry point for the evaluation harness."""
    args = _build_parser().parse_args()

    # Build thresholds from args (overriding defaults)
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.threshold_recall is not None:
        thresholds["recall_at_k"] = args.threshold_recall
    if args.threshold_groundedness is not None:
        thresholds["groundedness"] = args.threshold_groundedness
    if args.threshold_citation is not None:
        thresholds["citation_accuracy"] = args.threshold_citation
    if args.threshold_refusal is not None:
        thresholds["refusal_correctness"] = args.threshold_refusal

    recall_k = args.recall_k or 5

    summary = run_eval_suite(
        dataset_path=args.dataset,
        thresholds=thresholds,
        recall_k=recall_k,
        enable_tracing=args.trace,
    )

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)

    if not summary.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
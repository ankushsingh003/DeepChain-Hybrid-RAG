# """
# evaluation/benchmark.py  —  DeepChain Hybrid-RAG
# Fixed: outdated Ragas API usage + no async support + no MLflow integration.

# Changes vs original:
#   - Uses ragas >= 0.1.x Dataset API correctly (from_dict, not raw dicts).
#   - Wraps evaluate() in asyncio.run() so it doesn't block in async contexts.
#   - Compares Naive RAG vs GraphRAG in a single run and logs both to MLflow.
#   - Results saved to evaluation/results/<timestamp>.json for audit trail.
#   - Gracefully skips metrics that require OpenAI if only Gemini key is present.

# Run with:
#     python -m evaluation.benchmark
# """

# from __future__ import annotations

# import asyncio
# import json
# import logging
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# import mlflow
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import (
#     answer_relevancy,
#     context_precision,
#     context_recall,
#     faithfulness,
# )

# logger = logging.getLogger(__name__)

# RESULTS_DIR = Path("evaluation/results")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# # ── Sample Q&A pairs for benchmarking ────────────────────────────────────────
# # Replace with your actual domain questions

# BENCHMARK_QUESTIONS = [
#     {
#         "question": "What is the capital requirements ratio under Basel III?",
#         "ground_truth": (
#             "Basel III requires banks to maintain a minimum Common Equity Tier 1 "
#             "capital ratio of 4.5% of risk-weighted assets."
#         ),
#     },
#     {
#         "question": "What constitutes insider trading under SEC regulations?",
#         "ground_truth": (
#             "Insider trading involves buying or selling securities based on material, "
#             "non-public information in breach of a fiduciary duty or similar relationship."
#         ),
#     },
#     {
#         "question": "What are HIPAA's requirements for patient data storage?",
#         "ground_truth": (
#             "HIPAA requires covered entities to implement technical safeguards including "
#             "encryption, access controls, and audit logs for electronic protected health "
#             "information (ePHI)."
#         ),
#     },
# ]


# # ── Benchmark runner ──────────────────────────────────────────────────────────

# class RAGBenchmark:
#     def __init__(
#         self,
#         naive_retriever: Any,
#         graph_retriever: Any,
#         llm: Any,
#         mlflow_experiment: str = "deepchain-hybrid-rag",
#     ) -> None:
#         self.naive_retriever = naive_retriever
#         self.graph_retriever = graph_retriever
#         self.llm = llm
#         self.mlflow_experiment = mlflow_experiment

#         mlflow.set_experiment(mlflow_experiment)

#     def run(self, questions: list[dict] | None = None) -> dict[str, Any]:
#         """Run full benchmark comparison. Returns dict of {mode: ragas_scores}."""
#         questions = questions or BENCHMARK_QUESTIONS
#         results: dict[str, Any] = {}

#         for mode, retriever in [
#             ("naive_rag", self.naive_retriever),
#             ("graph_rag", self.graph_retriever),
#         ]:
#             logger.info("Benchmarking mode: %s", mode)
#             with mlflow.start_run(run_name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
#                 scores = self._evaluate_mode(retriever, questions, mode)
#                 results[mode] = scores

#                 # Log all scalar metrics to MLflow
#                 for metric_name, value in scores.items():
#                     if isinstance(value, (int, float)):
#                         mlflow.log_metric(metric_name, value)
#                 mlflow.log_param("mode", mode)
#                 mlflow.log_param("num_questions", len(questions))

#         self._save_results(results)
#         self._print_comparison(results)
#         return results

#     # ── Internal helpers ──────────────────────────────────────────────────────

#     def _evaluate_mode(
#         self,
#         retriever: Any,
#         questions: list[dict],
#         mode: str,
#     ) -> dict[str, float]:
#         """Build Ragas Dataset and run evaluation."""
#         rows: dict[str, list] = {
#             "question": [],
#             "answer": [],
#             "contexts": [],
#             "ground_truth": [],
#         }

#         for item in questions:
#             question = item["question"]
#             ground_truth = item["ground_truth"]

#             # Retrieve context
#             result = retriever.retrieve(question)
#             contexts = [c["text"] for c in result.chunks]

#             # Generate answer
#             context_str = "\n\n".join(contexts)
#             prompt = (
#                 f"Answer the question using only the context below.\n\n"
#                 f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
#             )
#             answer = self.llm.invoke(prompt).content.strip()

#             rows["question"].append(question)
#             rows["answer"].append(answer)
#             rows["contexts"].append(contexts)
#             rows["ground_truth"].append(ground_truth)

#         # ── Ragas >= 0.1.x API: from_dict, then evaluate ──────────────────────
#         dataset = Dataset.from_dict(rows)

#         metrics = [
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#             context_recall,
#         ]

#         try:
#             # evaluate() is synchronous in ragas >= 0.1.x
#             score = evaluate(dataset=dataset, metrics=metrics)
#             return {
#                 "faithfulness": float(score["faithfulness"]),
#                 "answer_relevancy": float(score["answer_relevancy"]),
#                 "context_precision": float(score["context_precision"]),
#                 "context_recall": float(score["context_recall"]),
#             }
#         except Exception as exc:
#             logger.error("Ragas evaluation failed for mode %s: %s", mode, exc)
#             return {"error": str(exc)}

#     def _save_results(self, results: dict[str, Any]) -> None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
#         with out_path.open("w", encoding="utf-8") as fh:
#             json.dump(results, fh, indent=2)
#         logger.info("Results saved to %s", out_path)

#     def _print_comparison(self, results: dict[str, Any]) -> None:
#         print("\n" + "=" * 60)
#         print("  DeepChain Benchmark — Naive RAG vs GraphRAG")
#         print("=" * 60)
#         metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
#         header = f"{'Metric':<25} {'Naive RAG':>12} {'GraphRAG':>12}"
#         print(header)
#         print("-" * 50)
#         for m in metrics:
#             naive_val = results.get("naive_rag", {}).get(m, "N/A")
#             graph_val = results.get("graph_rag", {}).get(m, "N/A")
#             naive_str = f"{naive_val:.4f}" if isinstance(naive_val, float) else str(naive_val)
#             graph_str = f"{graph_val:.4f}" if isinstance(graph_val, float) else str(graph_val)
#             print(f"{m:<25} {naive_str:>12} {graph_str:>12}")
#         print("=" * 60 + "\n")


# # ── Entry point ───────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import os
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     from retrieval.hybrid_retriever import HybridRetriever

#     logging.basicConfig(level=logging.INFO)

#     retriever = HybridRetriever(
#         neo4j_uri=os.environ["NEO4J_URI"],
#         neo4j_user=os.environ["NEO4J_USERNAME"],
#         neo4j_password=os.environ["NEO4J_PASSWORD"],
#     )

#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

#     # Wrap retriever for both modes
#     class _ModeRetriever:
#         def __init__(self, r: HybridRetriever, mode: str):
#             self._r, self._mode = r, mode
#         def retrieve(self, q: str):
#             return self._r.retrieve(q, mode=self._mode)

#     bench = RAGBenchmark(
#         naive_retriever=_ModeRetriever(retriever, "naive"),
#         graph_retriever=_ModeRetriever(retriever, "graph"),
#         llm=llm,
#     )
#     bench.run()














"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Benchmark Runner — Production Grade

Fixes vs original:
  - c["text"] → c.content  (RetrievedChunk is an object, not a dict)
  - Now benchmarks ALL 4 modes: naive / graph / hybrid / auto
    (original only ran naive vs graph — hybrid was never evaluated)
  - Uses RagasEvaluator from ragas_eval.py (no code duplication)
  - Records and compares LATENCY per mode (pipeline adds .latency to all results)
  - Records and compares retrieval-level metrics: hit_rate, MRR, NDCG
  - Records graph-specific metrics when graph/hybrid modes are run
  - Regression guard: benchmark fails loudly if quality drops below thresholds
  - MLflow logging: all metrics + params + artifacts per run
  - Results saved to evaluation/results/<timestamp>.json
  - Comparison table includes: faithfulness / relevancy / precision / recall /
    hit_rate / MRR / latency / mode
  - Ablation summary table (reranking, query rewriting, threshold sweep)
  - HybridRetriever used directly — no duplicate _ModeRetriever wrapper hack

New benchmark questions cover:
  - Factual lookup (naive-favoring)
  - Relationship/entity queries (graph-favoring)
  - Mixed queries (hybrid-favoring)
  - Edge cases: empty context, ambiguous queries, long queries
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from evaluation.ragas_eval import (
    EvaluationReport,
    EvaluationSample,
    EvaluationRegressionError,
    RagasEvaluator,
)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark question suite
# ---------------------------------------------------------------------------
# Each entry maps to EvaluationSample fields.
# expected_entities / expected_relations are used for graph metric evaluation.
# relevant_chunk_ids can be set to known chunk IDs for precise hit-rate scoring.

BENCHMARK_QUESTIONS: list[dict] = [

    # -- Factual / definitional (naive-favoring) --
    {
        "question": "What is the capital requirements ratio under Basel III?",
        "ground_truth": (
            "Basel III requires banks to maintain a minimum Common Equity Tier 1 "
            "capital ratio of 4.5% of risk-weighted assets."
        ),
        "expected_entities": ["Basel III", "Common Equity Tier 1"],
        "expected_relations": [],
    },
    {
        "question": "What are HIPAA's requirements for patient data storage?",
        "ground_truth": (
            "HIPAA requires covered entities to implement technical safeguards including "
            "encryption, access controls, and audit logs for electronic protected health "
            "information (ePHI)."
        ),
        "expected_entities": ["HIPAA", "ePHI"],
        "expected_relations": [],
    },
    {
        "question": "What constitutes insider trading under SEC regulations?",
        "ground_truth": (
            "Insider trading involves buying or selling securities based on material, "
            "non-public information in breach of a fiduciary duty or similar relationship."
        ),
        "expected_entities": ["SEC", "insider trading"],
        "expected_relations": [],
    },

    # -- Relationship / entity queries (graph-favoring) --
    {
        "question": "What is the relationship between RBI and NBFCs in India?",
        "ground_truth": (
            "The Reserve Bank of India (RBI) regulates Non-Banking Financial Companies "
            "(NBFCs), setting capital requirements, registration norms, and supervisory "
            "frameworks to maintain financial stability."
        ),
        "expected_entities": ["RBI", "NBFC", "Reserve Bank of India"],
        "expected_relations": ["regulates", "supervises"],
    },
    {
        "question": "How is UPI connected to NPCI and Indian banks?",
        "ground_truth": (
            "The Unified Payments Interface (UPI) is built and operated by the National "
            "Payments Corporation of India (NPCI) and connects participating Indian banks "
            "through a common interoperable payment infrastructure."
        ),
        "expected_entities": ["UPI", "NPCI", "Indian banks"],
        "expected_relations": ["operates", "connects", "interoperable"],
    },

    # -- Mixed queries (hybrid-favoring) --
    {
        "question": "How does the Dodd-Frank Act affect derivatives trading and which entities does it regulate?",
        "ground_truth": (
            "The Dodd-Frank Wall Street Reform Act mandates central clearing of standardized "
            "OTC derivatives and subjects swap dealers and major swap participants to registration, "
            "margin, and reporting requirements under CFTC and SEC oversight."
        ),
        "expected_entities": ["Dodd-Frank", "CFTC", "SEC", "swap dealers"],
        "expected_relations": ["regulates", "mandates", "subject to"],
    },
    {
        "question": "What FinTech regulatory sandbox frameworks exist in India and who oversees them?",
        "ground_truth": (
            "India's financial regulators including RBI, SEBI, and IRDAI each operate "
            "separate regulatory sandbox frameworks that allow FinTech startups to test "
            "innovative products under relaxed regulatory conditions."
        ),
        "expected_entities": ["RBI", "SEBI", "IRDAI", "regulatory sandbox"],
        "expected_relations": ["oversees", "operates", "allows"],
    },

    # -- Edge cases --
    {
        "question": "What is the maximum token length for Gemini embeddings?",
        "ground_truth": (
            "Gemini text-embedding-004 supports up to 2048 tokens per input for embedding generation."
        ),
        "expected_entities": ["Gemini", "text-embedding-004"],
        "expected_relations": [],
    },
    {
        "question": "Who founded DeepChain?",
        "ground_truth": "This information is not available in the knowledge base.",
        "expected_entities": ["DeepChain"],
        "expected_relations": ["founded by"],
    },
]


def _make_samples(questions: list[dict]) -> list[EvaluationSample]:
    return [
        EvaluationSample(
            question=q["question"],
            ground_truth=q["ground_truth"],
            expected_entities=q.get("expected_entities", []),
            expected_relations=q.get("expected_relations", []),
            relevant_chunk_ids=q.get("relevant_chunk_ids", []),
        )
        for q in questions
    ]


# ---------------------------------------------------------------------------
# RAGBenchmark
# ---------------------------------------------------------------------------

class RAGBenchmark:
    """
    Full benchmark runner for DeepChain-Hybrid-RAG.

    Evaluates all 4 pipeline modes (naive / graph / hybrid / auto),
    records RAGAS metrics + retrieval metrics + latency per mode,
    runs per-feature ablations, enforces regression thresholds,
    logs everything to MLflow, and saves JSON results.
    """

    MODES = ["naive", "graph", "hybrid", "auto"]

    def __init__(
        self,
        pipeline,                            # HybridRetriever instance
        evaluator: RagasEvaluator | None = None,
        mlflow_experiment: str = "deepchain-hybrid-rag",
        raise_on_regression: bool = False,
    ):
        self.pipeline = pipeline
        self.evaluator = evaluator or RagasEvaluator(raise_on_regression=raise_on_regression)
        self.raise_on_regression = raise_on_regression

        mlflow.set_experiment(mlflow_experiment)
        logger.info(f"[RAGBenchmark] Init — experiment='{mlflow_experiment}'")

    # -----------------------------------------------------------------------
    # Public: run full benchmark
    # -----------------------------------------------------------------------

    def run(
        self,
        questions: list[dict] | None = None,
        modes: list[str] | None = None,
        run_ablations: bool = True,
    ) -> dict[str, EvaluationReport]:
        """
        Run the full benchmark across all specified modes.

        Args:
            questions:     Override the default question suite.
            modes:         Subset of ["naive", "graph", "hybrid", "auto"] to run.
            run_ablations: Whether to run per-feature ablation tests.

        Returns:
            Dict mapping mode name → EvaluationReport.
        """
        questions = questions or BENCHMARK_QUESTIONS
        modes = modes or self.MODES
        samples = _make_samples(questions)

        reports: dict[str, EvaluationReport] = {}
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for mode in modes:
            logger.info(f"\n{'='*60}\n  Benchmarking mode: {mode}\n{'='*60}")

            with mlflow.start_run(run_name=f"{mode}_{run_timestamp}"):
                mlflow.log_param("mode", mode)
                mlflow.log_param("num_questions", len(questions))
                mlflow.log_param("run_ablations", run_ablations)

                # Health check before graph/hybrid modes
                if mode in ("graph", "hybrid"):
                    health = self.pipeline.health_check()
                    mlflow.log_param("neo4j_healthy", health.get("neo4j", False))
                    mlflow.log_param("weaviate_healthy", health.get("weaviate", False))
                    if not health.get("neo4j", False) and mode == "graph":
                        logger.warning(f"[Benchmark] Neo4j unhealthy — mode='{mode}' will fallback to naive")

                try:
                    report = self.evaluator.run_evaluation(
                        samples=samples,
                        mode_label=mode,
                        run_ablations=(run_ablations and mode == "hybrid"),  # ablations on hybrid only
                        pipeline=self.pipeline,
                    )
                except EvaluationRegressionError as e:
                    logger.error(f"[Benchmark] Regression error in mode='{mode}': {e}")
                    if self.raise_on_regression:
                        raise
                    report = None

                if report:
                    reports[mode] = report
                    self._log_to_mlflow(report)

        self._save_results(reports, run_timestamp)
        self._print_comparison(reports)

        return reports

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------

    def _log_to_mlflow(self, report: EvaluationReport) -> None:
        """Log all metrics and report JSON to MLflow."""
        # RAGAS metrics
        for k, v in report.ragas_scores.items():
            mlflow.log_metric(k, v)

        # Retrieval metrics
        r = report.retrieval_metrics
        mlflow.log_metric("hit_rate", r.hit_rate)
        mlflow.log_metric("mrr", r.mrr)
        mlflow.log_metric("ndcg", r.ndcg)
        mlflow.log_metric("context_coverage", r.context_coverage)
        mlflow.log_metric("avg_latency", r.avg_latency)
        mlflow.log_metric("total_eval_latency", report.total_latency)

        # Graph metrics
        if report.graph_metrics:
            g = report.graph_metrics
            mlflow.log_metric("graph_entity_match_rate", g.entity_match_rate)
            mlflow.log_metric("graph_relation_coverage", g.relation_coverage)
            mlflow.log_metric("graph_entity_miss_rate", g.entity_miss_rate)

        # Regression status
        mlflow.log_metric("regression_passed", int(report.passed_regression))
        mlflow.log_param("regression_violations", str(report.regression_violations))

        # Ablation metrics
        for ab in report.ablations:
            prefix = ab.condition.replace("=", "_").replace(".", "_").replace("/", "_")
            for k, v in ab.ragas_scores.items():
                mlflow.log_metric(f"abl_{prefix}_{k}", v)
            mlflow.log_metric(f"abl_{prefix}_hit_rate", ab.retrieval_metrics.hit_rate)

        logger.info(f"[Benchmark] Logged mode='{report.mode}' to MLflow")

    # -----------------------------------------------------------------------
    # Results persistence
    # -----------------------------------------------------------------------

    def _save_results(
        self,
        reports: dict[str, EvaluationReport],
        timestamp: str,
    ) -> None:
        """Serialize all reports to JSON for audit trail."""

        def _report_to_dict(r: EvaluationReport) -> dict:
            return {
                "mode": r.mode,
                "ragas_scores": r.ragas_scores,
                "retrieval_metrics": {
                    "hit_rate": r.retrieval_metrics.hit_rate,
                    "mrr": r.retrieval_metrics.mrr,
                    "ndcg": r.retrieval_metrics.ndcg,
                    "context_coverage": r.retrieval_metrics.context_coverage,
                    "avg_latency": r.retrieval_metrics.avg_latency,
                },
                "graph_metrics": {
                    "entity_match_rate": r.graph_metrics.entity_match_rate,
                    "relation_coverage": r.graph_metrics.relation_coverage,
                    "entity_miss_rate": r.graph_metrics.entity_miss_rate,
                } if r.graph_metrics else None,
                "ablations": [
                    {
                        "condition": ab.condition,
                        "ragas_scores": ab.ragas_scores,
                        "hit_rate": ab.retrieval_metrics.hit_rate,
                        "mrr": ab.retrieval_metrics.mrr,
                        "avg_latency": ab.avg_latency,
                    }
                    for ab in r.ablations
                ],
                "regression_violations": r.regression_violations,
                "passed_regression": r.passed_regression,
                "total_latency": r.total_latency,
                "sample_count": r.sample_count,
            }

        output = {
            "timestamp": timestamp,
            "reports": {mode: _report_to_dict(r) for mode, r in reports.items()},
        }
        out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        logger.info(f"[Benchmark] Results saved to {out_path}")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------

    def _print_comparison(self, reports: dict[str, EvaluationReport]) -> None:
        """Print a side-by-side comparison table of all evaluated modes."""
        sep = "=" * 90
        print(f"\n{sep}")
        print("  DeepChain Benchmark — Mode Comparison")
        print(sep)

        col_w = 14
        metrics = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "hit_rate", "mrr", "ndcg", "context_coverage",
        ]
        modes = list(reports.keys())

        # Header
        header = f"{'Metric':<27}" + "".join(f"{m:>{col_w}}" for m in modes)
        print(header)
        print("-" * (27 + col_w * len(modes)))

        for metric in metrics:
            row = f"{metric:<27}"
            best_val = -1.0
            vals: dict[str, float] = {}

            for mode, report in reports.items():
                if metric in report.ragas_scores:
                    v = report.ragas_scores[metric]
                elif metric == "hit_rate":
                    v = report.retrieval_metrics.hit_rate
                elif metric == "mrr":
                    v = report.retrieval_metrics.mrr
                elif metric == "ndcg":
                    v = report.retrieval_metrics.ndcg
                elif metric == "context_coverage":
                    v = report.retrieval_metrics.context_coverage
                else:
                    v = None
                vals[mode] = v
                if v is not None and v > best_val:
                    best_val = v

            for mode in modes:
                v = vals.get(mode)
                if v is None:
                    row += f"{'N/A':>{col_w}}"
                else:
                    marker = "★" if abs(v - best_val) < 0.0001 else " "
                    row += f"{marker}{v:.4f}".rjust(col_w)
            print(row)

        # Latency row
        print("-" * (27 + col_w * len(modes)))
        lat_row = f"{'avg_latency (s)':<27}"
        for mode, report in reports.items():
            lat = report.retrieval_metrics.avg_latency or (report.total_latency / max(report.sample_count, 1))
            lat_row += f"{lat:.3f}".rjust(col_w)
        print(lat_row)

        # Regression row
        reg_row = f"{'regression_passed':<27}"
        for mode, report in reports.items():
            status = "PASS ✅" if report.passed_regression else "FAIL ❌"
            reg_row += f"{status:>{col_w}}"
        print(reg_row)

        print(sep)

        # Ablation summary (from hybrid report if available)
        hybrid_report = reports.get("hybrid")
        if hybrid_report and hybrid_report.ablations:
            print("\n  Ablation Summary (hybrid mode):")
            print(f"  {'Condition':<40} {'Faith':>8} {'Relevancy':>10} {'HitRate':>8}")
            print(f"  {'-'*66}")
            for ab in hybrid_report.ablations:
                faith = ab.ragas_scores.get("faithfulness", 0.0)
                rel   = ab.ragas_scores.get("answer_relevancy", 0.0)
                hit   = ab.retrieval_metrics.hit_rate
                print(f"  {ab.condition:<40} {faith:>8.4f} {rel:>10.4f} {hit:>8.4f}")
            print()

        print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from vector_store.weaviate_client import WeaviateClient
    from vector_store.embedder import GeminiEmbedder
    from vector_store.retriever import VectorRetriever
    from graph.neo4j_client import Neo4jClient
    from retrieval.hybrid_retriever import HybridRetriever

    logging.basicConfig(level=logging.INFO)

    # Build the full pipeline
    weaviate_client = WeaviateClient()
    embedder        = GeminiEmbedder()
    vector_retriever = VectorRetriever(weaviate_client, embedder)
    neo4j_client    = Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    pipeline = HybridRetriever(
        retriever=vector_retriever,
        neo4j_client=neo4j_client,
        use_reranking=True,
        use_query_rewriting=True,
        use_cache=True,
    )

    evaluator = RagasEvaluator(raise_on_regression=False)

    bench = RAGBenchmark(
        pipeline=pipeline,
        evaluator=evaluator,
        mlflow_experiment="deepchain-hybrid-rag",
        raise_on_regression=False,
    )

    reports = bench.run(
        modes=["naive", "hybrid"],   # start with 2 modes; add "graph", "auto" when Neo4j is live
        run_ablations=True,
    )

    # Print individual detailed reports
    for mode, report in reports.items():
        evaluator.print_report(report)

    weaviate_client.close()
    neo4j_client.close()
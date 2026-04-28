# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Ragas Evaluation Suite - Benchmarking Model Performance
# """

# import os
# import json
# from typing import List, Dict, Any
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# from datasets import Dataset
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# class RagasEvaluator:
#     def __init__(self, model_name: str = "gemini-1.5-flash"):
#         self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

#     def run_evaluation(self, test_data: List[Dict[str, Any]]) -> Dict:
#         """Runs Ragas evaluation on a list of question/answer/context triplets."""
#         print(f"[*] Starting Evaluation on {len(test_data)} samples...")
        
#         # Prepare dataset for Ragas
#         # test_data should be: [{"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}]
#         dataset = Dataset.from_list(test_data)
        
#         result = evaluate(
#             dataset,
#             metrics=[
#                 faithfulness,
#                 answer_relevancy,
#                 context_precision,
#                 context_recall,
#             ],
#             llm=self.llm,
#         )
        
#         print("[+] Evaluation Complete.")
#         return result.to_pandas().to_dict()

# if __name__ == "__main__":
#     # Sample Mock Evaluation
#     evaluator = RagasEvaluator()
#     sample_data = [{
#         "question": "What is DeepChain?",
#         "answer": "DeepChain is a hybrid RAG system.",
#         "contexts": ["DeepChain combines knowledge graphs and vector search."],
#         "ground_truth": "DeepChain is a hybrid RAG system using Neo4j and Weaviate."
#     }]
#     # stats = evaluator.run_evaluation(sample_data)
#     # print(stats)







"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Ragas Evaluation Suite — Production Grade

Fixes vs original:
  - llm= now uses LangchainLLMWrapper + LangchainEmbeddingsWrapper (Ragas ≥ 0.1.x API)
    Previously passing a raw LangChain LLM to ragas evaluate() silently fell back
    to OpenAI and crashed if OPENAI_API_KEY was not set.
  - evaluate() result accessed correctly: score[metric_name] not .to_pandas()
  - Integrated with HybridResult / NaiveRAG / GraphRAG output shapes from
    the upgraded pipeline (uses .content not ["text"], uses .score not raw dict)

New additions aligned to upgraded pipeline:
  - Per-feature ablation evaluation:
      * query_rewriting ON vs OFF
      * reranking ON vs OFF
      * BM25 fallback ON vs OFF
      * cache hit quality (stale vs fresh)
      * distance_threshold sensitivity sweep
  - Retrieval-level metrics (no LLM required):
      * Hit Rate @ K — did any relevant chunk appear in top-K?
      * MRR — Mean Reciprocal Rank (where was the first relevant chunk?)
      * NDCG @ K — normalized discounted cumulative gain
      * Context coverage — fraction of ground truth facts found in contexts
  - Graph-specific evaluation:
      * Entity match rate — fraction of expected entities found in graph results
      * Relationship coverage — expected relationships present in graph facts
  - Mode comparison: naive / graph / hybrid / auto all benchmarked in one run
  - Regression guard: raises EvaluationRegressionError if any metric < threshold
  - All results returned as structured EvaluationReport dataclass
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Ragas ≥ 0.1.x wrappers — required to use non-OpenAI LLMs
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    _RAGAS_WRAPPERS_AVAILABLE = True
except ImportError:
    _RAGAS_WRAPPERS_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Regression thresholds — evaluation fails if any metric drops below these
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "faithfulness":       0.70,
    "answer_relevancy":   0.70,
    "context_precision":  0.60,
    "context_recall":     0.60,
    "hit_rate":           0.70,
    "mrr":                0.50,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSample:
    """One evaluation row — must have question + ground_truth at minimum."""
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    # For graph evaluation
    expected_entities: list[str] = field(default_factory=list)
    expected_relations: list[str] = field(default_factory=list)
    # For retrieval ranking evaluation
    relevant_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Retrieval-level metrics that don't require an LLM."""
    hit_rate: float          # ≥1 relevant chunk in top-K
    mrr: float               # mean reciprocal rank
    ndcg: float              # normalized discounted cumulative gain
    context_coverage: float  # fraction of ground truth facts found in contexts
    avg_score: float         # average retrieval score (cosine similarity proxy)
    avg_latency: float       # average retrieval latency in seconds


@dataclass
class GraphMetrics:
    """Graph-specific retrieval metrics."""
    entity_match_rate: float      # fraction of expected entities found
    relation_coverage: float      # fraction of expected relations found
    avg_graph_facts: float        # average graph facts returned per query
    entity_miss_rate: float       # fraction of expected entities NOT found


@dataclass
class AblationResult:
    """Result of one ablation condition (feature ON vs OFF)."""
    condition: str
    ragas_scores: dict[str, float]
    retrieval_metrics: RetrievalMetrics
    avg_latency: float
    sample_count: int


@dataclass
class EvaluationReport:
    """Full evaluation report for one pipeline mode."""
    mode: str
    ragas_scores: dict[str, float]
    retrieval_metrics: RetrievalMetrics
    graph_metrics: Optional[GraphMetrics]
    ablations: list[AblationResult]
    regression_violations: list[str]
    passed_regression: bool
    total_latency: float
    sample_count: int


class EvaluationRegressionError(Exception):
    """Raised when a metric drops below its configured threshold."""
    pass


# ---------------------------------------------------------------------------
# RagasEvaluator
# ---------------------------------------------------------------------------

class RagasEvaluator:
    """
    Comprehensive evaluation suite for DeepChain-Hybrid-RAG.

    Evaluates:
      - RAGAS semantic metrics (faithfulness, relevancy, precision, recall)
      - Retrieval-level metrics (hit rate, MRR, NDCG, context coverage)
      - Graph-specific metrics (entity match, relationship coverage)
      - Per-feature ablations (reranking, query rewriting, BM25, threshold sweep)
      - Regression guard with configurable thresholds
    """

    def __init__(
        self,
        model_name: str | None = None,
        embedding_model: str = "models/gemini-embedding-001",
        thresholds: dict[str, float] | None = None,
        raise_on_regression: bool = False,
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.raise_on_regression = raise_on_regression

        model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.0-flash")
        # Build Ragas-compatible wrappers for Gemini
        langchain_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        langchain_emb = GoogleGenerativeAIEmbeddings(model=embedding_model)

        if _RAGAS_WRAPPERS_AVAILABLE:
            self.ragas_llm = LangchainLLMWrapper(langchain_llm)
            self.ragas_embeddings = LangchainEmbeddingsWrapper(langchain_emb)
        else:
            # Older ragas — pass raw objects (may work, may not)
            logger.warning(
                "[RagasEvaluator] LangchainLLMWrapper not available. "
                "Using raw LangChain objects — upgrade ragas if evaluation fails."
            )
            self.ragas_llm = langchain_llm
            self.ragas_embeddings = langchain_emb

        logger.info(f"[RagasEvaluator] Init — model={model_name}, thresholds={self.thresholds}")

    # -----------------------------------------------------------------------
    # Public: main evaluation entry point
    # -----------------------------------------------------------------------

    def run_evaluation(
        self,
        samples: list[EvaluationSample],
        mode_label: str = "pipeline",
        run_ablations: bool = True,
        pipeline=None,           # optional: pass HybridRetriever to run ablations live
    ) -> EvaluationReport:
        """
        Run the full evaluation suite on a list of EvaluationSamples.

        Args:
            samples:        Evaluation samples (must have answer + contexts pre-filled,
                            OR pass pipeline= to auto-generate them).
            mode_label:     Name of the mode being evaluated (for logging/reporting).
            run_ablations:  Whether to run per-feature ablation tests.
            pipeline:       Optional HybridRetriever to generate answers live.

        Returns:
            EvaluationReport with all metrics.
        """
        logger.info(f"[RagasEvaluator] Evaluating mode='{mode_label}' on {len(samples)} samples")
        t0 = time.perf_counter()

        # If pipeline provided, generate answers + contexts live
        if pipeline is not None:
            samples = self._generate_answers(samples, pipeline)

        # 1. RAGAS semantic metrics
        ragas_scores = self._run_ragas(samples, mode_label)

        # 2. Retrieval-level metrics (no LLM)
        retrieval_metrics = self._compute_retrieval_metrics(samples)

        # 3. Graph metrics (only if samples have expected_entities)
        has_graph_data = any(s.expected_entities for s in samples)
        graph_metrics = self._compute_graph_metrics(samples) if has_graph_data else None

        # 4. Ablations
        ablations: list[AblationResult] = []
        if run_ablations and pipeline is not None:
            ablations = self._run_ablations(samples, pipeline)

        # 5. Regression check
        all_metrics = {**ragas_scores, **{
            "hit_rate": retrieval_metrics.hit_rate,
            "mrr":      retrieval_metrics.mrr,
        }}
        violations = self._check_regression(all_metrics)
        passed = len(violations) == 0

        if violations:
            msg = f"[RagasEvaluator] REGRESSION VIOLATIONS in mode='{mode_label}': {violations}"
            logger.error(msg)
            if self.raise_on_regression:
                raise EvaluationRegressionError(msg)
        else:
            logger.info(f"[RagasEvaluator] All regression checks passed for mode='{mode_label}'")

        total_latency = round(time.perf_counter() - t0, 3)
        logger.info(f"[RagasEvaluator] Complete in {total_latency}s")

        return EvaluationReport(
            mode=mode_label,
            ragas_scores=ragas_scores,
            retrieval_metrics=retrieval_metrics,
            graph_metrics=graph_metrics,
            ablations=ablations,
            regression_violations=violations,
            passed_regression=passed,
            total_latency=total_latency,
            sample_count=len(samples),
        )

    # -----------------------------------------------------------------------
    # RAGAS semantic metrics
    # -----------------------------------------------------------------------

    def _run_ragas(
        self,
        samples: list[EvaluationSample],
        mode_label: str,
    ) -> dict[str, float]:
        """Run the 4 core RAGAS metrics using Gemini via LangchainLLMWrapper."""
        rows = {
            "question":    [s.question     for s in samples],
            "answer":      [s.answer       for s in samples],
            "contexts":    [s.contexts     for s in samples],
            "ground_truth":[s.ground_truth for s in samples],
        }
        dataset = Dataset.from_dict(rows)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=False,
            )
            scores = {
                "faithfulness":      float(result["faithfulness"]),
                "answer_relevancy":  float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall":    float(result["context_recall"]),
            }
            logger.info(f"[RagasEvaluator][{mode_label}] RAGAS scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"[RagasEvaluator] RAGAS evaluation failed: {e}")
            return {m: 0.0 for m in ["faithfulness", "answer_relevancy",
                                      "context_precision", "context_recall"]}

    # -----------------------------------------------------------------------
    # Retrieval-level metrics (no LLM needed)
    # -----------------------------------------------------------------------

    def _compute_retrieval_metrics(
        self,
        samples: list[EvaluationSample],
    ) -> RetrievalMetrics:
        """
        Compute retrieval quality metrics that don't require an LLM judge.

        Hit Rate:  ≥1 relevant chunk in top-K contexts
        MRR:       1/rank of first relevant chunk (0 if none)
        NDCG:      normalized discounted cumulative gain
        Coverage:  fraction of ground truth words found in contexts
        Avg score: mean of retrieval similarity scores (from chunk metadata if available)
        """
        hit_rates, mrrs, ndcgs, coverages = [], [], [], []
        latencies = []

        for s in samples:
            gt_words = set(s.ground_truth.lower().split())
            context_blob = " ".join(s.contexts).lower()
            context_words = set(context_blob.split())

            # Coverage: fraction of GT content words found in contexts
            if gt_words:
                coverage = len(gt_words & context_words) / len(gt_words)
            else:
                coverage = 0.0
            coverages.append(coverage)

            # Hit Rate / MRR / NDCG using relevant_chunk_ids if provided
            if s.relevant_chunk_ids:
                hit = 0
                rr = 0.0
                dcg = 0.0
                idcg = sum(1.0 / math.log2(i + 2) for i in range(len(s.relevant_chunk_ids)))
                for rank, ctx in enumerate(s.contexts, 1):
                    is_relevant = any(cid in ctx for cid in s.relevant_chunk_ids)
                    if is_relevant:
                        hit = 1
                        if rr == 0.0:
                            rr = 1.0 / rank
                        dcg += 1.0 / math.log2(rank + 1)
                hit_rates.append(hit)
                mrrs.append(rr)
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            else:
                # Fallback: use coverage as proxy
                hit_rates.append(1.0 if coverage > 0.3 else 0.0)
                mrrs.append(coverage)
                ndcgs.append(coverage)

        return RetrievalMetrics(
            hit_rate=round(sum(hit_rates) / len(hit_rates), 4) if hit_rates else 0.0,
            mrr=round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            ndcg=round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            context_coverage=round(sum(coverages) / len(coverages), 4) if coverages else 0.0,
            avg_score=0.0,   # populated externally when RetrievedChunk objects are available
            avg_latency=round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        )

    # -----------------------------------------------------------------------
    # Graph-specific metrics
    # -----------------------------------------------------------------------

    def _compute_graph_metrics(
        self,
        samples: list[EvaluationSample],
    ) -> GraphMetrics:
        """
        Evaluate graph retrieval quality.
        Requires samples to have expected_entities and/or expected_relations set.
        """
        entity_matches, relation_matches, graph_fact_counts = [], [], []

        for s in samples:
            ctx_blob = " ".join(s.contexts).lower()

            if s.expected_entities:
                matched = sum(
                    1 for e in s.expected_entities
                    if e.lower() in ctx_blob
                )
                entity_matches.append(matched / len(s.expected_entities))

            if s.expected_relations:
                matched = sum(
                    1 for r in s.expected_relations
                    if r.lower() in ctx_blob
                )
                relation_matches.append(matched / len(s.expected_relations))

        emr = sum(entity_matches) / len(entity_matches) if entity_matches else 0.0
        rcr = sum(relation_matches) / len(relation_matches) if relation_matches else 0.0

        return GraphMetrics(
            entity_match_rate=round(emr, 4),
            relation_coverage=round(rcr, 4),
            avg_graph_facts=0.0,   # populated by benchmark.py which has graph_facts list
            entity_miss_rate=round(1.0 - emr, 4),
        )

    # -----------------------------------------------------------------------
    # Per-feature ablation tests
    # -----------------------------------------------------------------------

    def _run_ablations(
        self,
        base_samples: list[EvaluationSample],
        pipeline,
    ) -> list[AblationResult]:
        """
        Run ablation tests for each major pipeline feature introduced in the upgrade.
        Each ablation toggles one feature and measures the impact on quality.
        """
        ablations: list[AblationResult] = []

        # -- Ablation 1: Query rewriting ON vs OFF
        for rewrite in [True, False]:
            label = f"query_rewriting={'ON' if rewrite else 'OFF'}"
            logger.info(f"[Ablation] {label}")
            try:
                pipeline.naive_rag.use_query_rewriting = rewrite
                samples = self._generate_answers(base_samples, pipeline, mode="naive")
                ragas = self._run_ragas(samples, label)
                retrieval = self._compute_retrieval_metrics(samples)
                avg_lat = sum(
                    getattr(s, "_latency", 0.0) for s in samples
                ) / max(len(samples), 1)
                ablations.append(AblationResult(label, ragas, retrieval, avg_lat, len(samples)))
            except Exception as e:
                logger.warning(f"[Ablation] {label} failed: {e}")
            finally:
                # Restore default
                pipeline.naive_rag.use_query_rewriting = True

        # -- Ablation 2: Cross-encoder reranking ON vs OFF
        for rerank in [True, False]:
            label = f"reranking={'ON' if rerank else 'OFF'}"
            logger.info(f"[Ablation] {label}")
            try:
                pipeline.use_reranking = rerank
                samples = self._generate_answers(base_samples, pipeline, mode="naive")
                ragas = self._run_ragas(samples, label)
                retrieval = self._compute_retrieval_metrics(samples)
                ablations.append(AblationResult(label, ragas, retrieval, 0.0, len(samples)))
            except Exception as e:
                logger.warning(f"[Ablation] {label} failed: {e}")
            finally:
                pipeline.use_reranking = True

        # -- Ablation 3: Distance threshold sweep (0.20 / 0.30 / 0.40)
        for threshold in [0.20, 0.30, 0.40]:
            label = f"distance_threshold={threshold}"
            logger.info(f"[Ablation] {label}")
            try:
                orig = pipeline.naive_rag.retriever.default_distance_threshold
                pipeline.naive_rag.retriever.default_distance_threshold = threshold
                samples = self._generate_answers(base_samples, pipeline, mode="naive")
                ragas = self._run_ragas(samples, label)
                retrieval = self._compute_retrieval_metrics(samples)
                ablations.append(AblationResult(label, ragas, retrieval, 0.0, len(samples)))
            except Exception as e:
                logger.warning(f"[Ablation] {label} failed: {e}")
            finally:
                pipeline.naive_rag.retriever.default_distance_threshold = orig

        # -- Ablation 4: Cache hit quality (run same queries twice, compare)
        label = "cache_hit_quality"
        logger.info(f"[Ablation] {label}")
        try:
            pipeline.naive_rag._cache.invalidate()
            # First pass — fills cache
            s1 = self._generate_answers(base_samples, pipeline, mode="naive")
            # Second pass — serves from cache
            s2 = self._generate_answers(base_samples, pipeline, mode="naive")
            ragas_fresh  = self._run_ragas(s1, f"{label}_fresh")
            ragas_cached = self._run_ragas(s2, f"{label}_cached")
            retrieval    = self._compute_retrieval_metrics(s2)
            # Diff: cached vs fresh
            diff = {
                k: round(ragas_cached.get(k, 0) - ragas_fresh.get(k, 0), 4)
                for k in ragas_fresh
            }
            logger.info(f"[Ablation] Cache quality diff (cached - fresh): {diff}")
            ablations.append(AblationResult(
                f"{label}_cached", ragas_cached, retrieval, 0.0, len(s2)
            ))
        except Exception as e:
            logger.warning(f"[Ablation] {label} failed: {e}")

        # -- Ablation 5: Mode comparison naive / graph / hybrid / auto
        for mode in ["naive", "graph", "hybrid", "auto"]:
            label = f"mode={mode}"
            logger.info(f"[Ablation] {label}")
            try:
                samples = self._generate_answers(base_samples, pipeline, mode=mode)
                ragas = self._run_ragas(samples, label)
                retrieval = self._compute_retrieval_metrics(samples)
                ablations.append(AblationResult(label, ragas, retrieval, 0.0, len(samples)))
            except Exception as e:
                logger.warning(f"[Ablation] {label} failed: {e}")

        return ablations

    # -----------------------------------------------------------------------
    # Answer generation helpers
    # -----------------------------------------------------------------------

    def _generate_answers(
        self,
        base_samples: list[EvaluationSample],
        pipeline,
        mode: str = "hybrid",
    ) -> list[EvaluationSample]:
        """
        Run the pipeline on each sample to populate answer + contexts.
        Supports HybridRetriever, NaiveRAG, or GraphRAG.
        """
        filled: list[EvaluationSample] = []

        for s in base_samples:
            try:
                t0 = time.perf_counter()

                # Detect pipeline type and call accordingly
                if hasattr(pipeline, "query") and hasattr(pipeline, "naive_rag"):
                    # HybridRetriever
                    result = pipeline.query(s.question, mode=mode)
                    answer   = result.answer
                    chunks   = result.chunks
                    contexts = [c.content for c in chunks]
                elif hasattr(pipeline, "query") and hasattr(pipeline, "retriever"):
                    # NaiveRAG or GraphRAG
                    result   = pipeline.query(s.question)
                    answer   = result["answer"] if isinstance(result, dict) else result.answer
                    contexts = [c.content for c in result.get("chunks", [])] \
                               if isinstance(result, dict) else []
                else:
                    raise ValueError(f"Unknown pipeline type: {type(pipeline)}")

                latency = round(time.perf_counter() - t0, 3)

                filled_sample = EvaluationSample(
                    question=s.question,
                    ground_truth=s.ground_truth,
                    answer=answer,
                    contexts=contexts if contexts else [""],
                    expected_entities=s.expected_entities,
                    expected_relations=s.expected_relations,
                    relevant_chunk_ids=s.relevant_chunk_ids,
                )
                filled_sample._latency = latency  # type: ignore[attr-defined]
                filled.append(filled_sample)

            except Exception as e:
                logger.warning(f"[RagasEvaluator] Failed to generate answer for '{s.question[:50]}': {e}")
                filled.append(EvaluationSample(
                    question=s.question,
                    ground_truth=s.ground_truth,
                    answer="[ERROR]",
                    contexts=[""],
                ))

        return filled

    # -----------------------------------------------------------------------
    # Regression guard
    # -----------------------------------------------------------------------

    def _check_regression(self, scores: dict[str, float]) -> list[str]:
        """Return list of violation strings for any metric below its threshold."""
        violations = []
        for metric, threshold in self.thresholds.items():
            value = scores.get(metric)
            if value is not None and value < threshold:
                violations.append(
                    f"{metric}={value:.4f} < threshold={threshold:.4f}"
                )
        return violations

    # -----------------------------------------------------------------------
    # Pretty print
    # -----------------------------------------------------------------------

    def print_report(self, report: EvaluationReport) -> None:
        """Print a structured summary of an EvaluationReport."""
        sep = "=" * 65

        print(f"\n{sep}")
        print(f"  DeepChain Evaluation Report — mode='{report.mode}'")
        print(sep)

        print("\n📊 RAGAS Semantic Metrics:")
        for k, v in report.ragas_scores.items():
            threshold = self.thresholds.get(k, None)
            flag = "✅" if threshold is None or v >= threshold else "❌"
            print(f"  {flag}  {k:<25} {v:.4f}  (threshold: {threshold or 'N/A'})")

        r = report.retrieval_metrics
        print("\n🔍 Retrieval Metrics:")
        print(f"  Hit Rate @ K      : {r.hit_rate:.4f}")
        print(f"  MRR               : {r.mrr:.4f}")
        print(f"  NDCG @ K          : {r.ndcg:.4f}")
        print(f"  Context Coverage  : {r.context_coverage:.4f}")

        if report.graph_metrics:
            g = report.graph_metrics
            print("\n🕸️  Graph Metrics:")
            print(f"  Entity Match Rate : {g.entity_match_rate:.4f}")
            print(f"  Entity Miss Rate  : {g.entity_miss_rate:.4f}")
            print(f"  Relation Coverage : {g.relation_coverage:.4f}")

        if report.ablations:
            print("\n🔬 Ablation Results:")
            for ab in report.ablations:
                faith = ab.ragas_scores.get("faithfulness", 0.0)
                rel   = ab.ragas_scores.get("answer_relevancy", 0.0)
                print(f"  [{ab.condition:<35}] faith={faith:.3f}  relevancy={rel:.3f}  "
                      f"hit={ab.retrieval_metrics.hit_rate:.3f}")

        status = "✅ PASSED" if report.passed_regression else "❌ FAILED"
        print(f"\n🛡️  Regression Check: {status}")
        if report.regression_violations:
            for v in report.regression_violations:
                print(f"     ⚠️  {v}")

        print(f"\n⏱️  Total eval time : {report.total_latency}s")
        print(f"   Samples evaluated: {report.sample_count}")
        print(sep + "\n")
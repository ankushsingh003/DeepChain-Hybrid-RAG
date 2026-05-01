"""
finance/strategies/advisor.py — DeepChain
Module: Strategy Advisor

IMPROVEMENTS:
  - Model updated to gemini-2.0-flash with fallback to gemini-1.5-flash
  - Strategy report now returns structured JSON sections (not just raw text)
    so the frontend can render each section independently.
  - Added 'confidence_score', 'recommended_instruments', and 'warnings' fields.
  - Added _safe_generate() wrapper with retry + model fallback on 404.
  - RAG context is now trimmed to MAX_CONTEXT_CHARS to avoid token overflows.
"""

import os
import time
import logging
from typing import Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
PRIMARY_MODEL   = os.getenv("LLM_MODEL", "gemini-2.0-flash")
FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
MAX_CONTEXT_CHARS = 4000   # trim RAG context to avoid token limit


def _is_model_not_found(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "404" in msg or "NOT_FOUND" in msg or "NOT FOUND" in msg


class StrategyAdvisor:
    """
    The Strategy Advisor that provides full trading strategy approaches.
    Uses RAG to retrieve established strategies and generates custom analysis.
    """

    def __init__(self, model_name: str | None = None):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")

        genai.configure(api_key=self.api_key)
        self._model_name = model_name or PRIMARY_MODEL
        self._build_model(self._model_name)
        self.retriever = None

    def _build_model(self, model_name: str) -> None:
        self.model = genai.GenerativeModel(model_name)
        self._model_name = model_name
        logger.info("[StrategyAdvisor] Using model: %s", model_name)

    def _switch_fallback(self) -> bool:
        candidates = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))
        try:
            idx = candidates.index(self._model_name)
        except ValueError:
            idx = -1
        for model in candidates[idx + 1:]:
            if model != self._model_name:
                logger.warning("[StrategyAdvisor] Switching to fallback model: %s", model)
                self._build_model(model)
                return True
        logger.error("[StrategyAdvisor] All fallback models exhausted.")
        return False

    def _safe_generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content with retry and model fallback support."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as exc:
                if _is_model_not_found(exc):
                    if self._switch_fallback():
                        continue
                    return f"Error: No available model could process this request."
                wait = 2 ** attempt
                logger.warning("[StrategyAdvisor] Attempt %d/%d failed: %s — retry in %ds",
                               attempt, max_retries, exc, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                else:
                    return f"Error generating strategy after {max_retries} attempts: {str(exc)}"
        return "Error: strategy generation failed."

    def _get_retriever(self):
        if self.retriever is None:
            try:
                from retrieval.hybrid_retriever import HybridRetriever
                from vector_store.weaviate_client import WeaviateClient
                from vector_store.embedder import GeminiEmbedder
                from vector_store.retriever import VectorRetriever
                from graph.neo4j_client import Neo4jClient

                w_client   = WeaviateClient()
                emb        = GeminiEmbedder()
                v_retriever = VectorRetriever(w_client, emb)
                n_client   = Neo4jClient()

                self.retriever = HybridRetriever(
                    retriever=v_retriever,
                    neo4j_client=n_client,
                    model_name=self._model_name,
                )
            except Exception as e:
                logger.error("[StrategyAdvisor] Could not initialize retriever: %s", e)
                return None
        return self.retriever

    async def get_strategy_approach(self, user_intent: str) -> Dict[str, Any]:
        """
        Retrieves relevant strategy context and generates a full strategy report.
        Returns structured dict with separate sections for easier frontend rendering.
        """
        start_time = time.time()

        # 1. Retrieve Knowledge Base context
        retriever = self._get_retriever()
        context   = ""
        if retriever:
            try:
                rag_result = await retriever.query(
                    question=f"Which trading strategies match this intent: {user_intent}? Provide details.",
                    mode="hybrid",
                )
                context = rag_result.answer[:MAX_CONTEXT_CHARS]
            except Exception as e:
                logger.warning("[StrategyAdvisor] RAG retrieval failed: %s", e)

        # 2. Generate Structured Strategy Report
        prompt = f"""
System: You are an expert Quant Trading Strategist for DeepChain.
Context from Knowledge Base:
{context}

User Request: {user_intent}

Task: Provide a comprehensive 'Full Strategy and Trading Code Approach' report.
Return your response as a valid JSON object with these exact keys:
{{
  "strategy_name": "...",
  "overview": "...",
  "mathematical_formulation": "...",
  "implementation_code": "...(complete Python def strategy_function(df): ...)",
  "entry_exit_logic": "...",
  "backtesting_approach": "...",
  "risk_management": "...",
  "recommended_instruments": ["...", "..."],
  "confidence_score": 0.0-1.0,
  "warnings": ["..."]
}}

IMPORTANT: Return ONLY the JSON object. No markdown fences, no extra text.
The implementation_code MUST use signature `def strategy_function(df):` and return df with a 'signals' column (1=buy, -1=sell, 0=hold).
"""

        raw = self._safe_generate(prompt)

        # Parse JSON or fall back to raw text
        structured = None
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            structured = __import__("json").loads(cleaned.strip())
        except Exception:
            structured = {"overview": raw, "implementation_code": "", "warnings": ["Raw text returned — JSON parsing failed"]}

        return {
            "approach_report": raw,
            "structured":      structured,
            "latency":         round(time.time() - start_time, 3),
            "retrieved_context": context,
            "model_used":      self._model_name,
        }


if __name__ == "__main__":
    import asyncio
    async def test():
        advisor = StrategyAdvisor()
        result  = await advisor.get_strategy_approach("I want a strategy based on volatility and breakouts.")
        print(result["approach_report"])
    asyncio.run(test())

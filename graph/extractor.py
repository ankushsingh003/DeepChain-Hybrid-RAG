"""
graph/extractor.py  —  DeepChain Hybrid-RAG
Module: Triplet Extractor (Core LLM Logic)

FIXES:
  - Replaced deprecated 'gemini-1.5-flash-latest' with 'gemini-2.0-flash'
  - Added FALLBACK_MODELS list: on 404/NOT_FOUND, automatically retries with
    the next model in the list (sticky — once one works, all chunks use it).
  - Improved TRIPLET_PROMPT: richer predicates, entity-type tags, exhaustive.
  - _is_model_not_found() helper distinguishes 404 from transient errors.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
PRIMARY_MODEL   = os.getenv("LLM_MODEL", "gemini-2.0-flash")
FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES   = 3
BACKOFF_BASE  = 2
SKIP_LOG_PATH = Path("extraction_errors.jsonl")

# ── Improved Prompt ───────────────────────────────────────────────────────────
TRIPLET_PROMPT = """\
You are a knowledge-graph extraction engine.
Extract ALL factual relationships from the text below.

Rules:
1. Return ONLY a JSON array — no prose, no markdown fences, no explanation.
2. Each element must have exactly these keys:
   "subject"   — source entity (string, title-case preferred)
   "predicate" — relationship label (UPPER_SNAKE_CASE, e.g. TREATS, CAUSES, LOCATED_IN)
   "object"    — target entity or value (string)
   "subj_type" — entity type of subject  (Drug, Disease, Person, Org, Location, Concept, etc.)
   "obj_type"  — entity type of object   (same vocabulary)
3. Be exhaustive: extract every factual pair you can find.
4. Keep entity names consistent across triplets.
5. If no relationships exist, return an empty array: []

Text:
{text}
"""


def _is_model_not_found(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "404" in msg or "NOT_FOUND" in msg or "NOT FOUND" in msg


class TripletExtractor:
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI | None = None,
        model_name: str | None = None,
    ) -> None:
        self._model_name = model_name or PRIMARY_MODEL
        self._build_llm(self._model_name, llm)

    # ── LLM management ────────────────────────────────────────────────────────

    def _build_llm(self, model_name: str, llm=None) -> None:
        self.llm = llm or ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self._model_name = model_name
        logger.info("[TripletExtractor] Using model: %s", model_name)

    def _switch_to_fallback(self, failed_model: str) -> bool:
        candidates = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))
        try:
            idx = candidates.index(failed_model)
        except ValueError:
            idx = -1
        for model in candidates[idx + 1:]:
            if model != failed_model:
                logger.warning("[TripletExtractor] '%s' not found — switching to '%s'", failed_model, model)
                self._build_llm(model)
                return True
        logger.error("[TripletExtractor] All fallback models exhausted.")
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        all_triplets: list[dict[str, str]] = []
        failed = 0
        for chunk in chunks:
            triplets = self._extract_with_retry(chunk)
            if triplets is None:
                failed += 1
                self._log_skip(chunk, reason="max_retries_exceeded")
            else:
                for t in triplets:
                    t["source_chunk_id"] = chunk.get("chunk_id", "unknown")
                all_triplets.extend(triplets)
        logger.info(
            "Triplet extraction — chunks=%d ok=%d failed=%d triplets=%d",
            len(chunks), len(chunks) - failed, failed, len(all_triplets),
        )
        return all_triplets

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_with_retry(self, chunk: dict[str, Any]) -> list[dict[str, str]] | None:
        text = chunk.get("text", "").strip()
        if not text:
            return []

        prompt = TRIPLET_PROMPT.format(text=text)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                return self._parse_response(content, chunk)

            except json.JSONDecodeError as exc:
                logger.warning("JSON parse error chunk %s attempt %d: %s", chunk.get("chunk_id"), attempt, exc)
                self._log_skip(chunk, reason=f"json_parse_error: {exc}")
                return []

            except Exception as exc:  # noqa: BLE001
                if _is_model_not_found(exc):
                    switched = self._switch_to_fallback(self._model_name)
                    if not switched:
                        return None
                    logger.info("[TripletExtractor] Retrying chunk %s with %s", chunk.get("chunk_id"), self._model_name)
                    continue  # retry same attempt with new model

                wait = BACKOFF_BASE ** attempt
                logger.warning("API error chunk %s attempt %d/%d: %s — retry in %ds",
                               chunk.get("chunk_id"), attempt, MAX_RETRIES, exc, wait)
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                else:
                    logger.error("Giving up on chunk %s after %d attempts", chunk.get("chunk_id"), MAX_RETRIES)
                    return None
        return None

    def _parse_response(self, raw: str, chunk: dict[str, Any]) -> list[dict[str, str]]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            if "[" in cleaned and "]" in cleaned:
                parsed = json.loads(cleaned[cleaned.find("["):cleaned.rfind("]") + 1])
            else:
                raise

        if not isinstance(parsed, list):
            return []

        valid = []
        for item in parsed:
            if not all(k in item for k in ("subject", "predicate", "object")):
                logger.debug("Dropping malformed triplet: %s", item)
                continue
            valid.append({
                "subject":   str(item["subject"]).strip(),
                "predicate": str(item["predicate"]).strip().upper().replace(" ", "_"),
                "object":    str(item["object"]).strip(),
                "subj_type": str(item.get("subj_type", "Entity")).strip(),
                "obj_type":  str(item.get("obj_type",  "Entity")).strip(),
            })
        return valid

    def _log_skip(self, chunk: dict[str, Any], reason: str) -> None:
        record = {
            "chunk_id":  chunk.get("chunk_id", "unknown"),
            "source":    chunk.get("source",   "unknown"),
            "reason":    reason,
            "model":     self._model_name,
            "timestamp": time.time(),
        }
        with SKIP_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

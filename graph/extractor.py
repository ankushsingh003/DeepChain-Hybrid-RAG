"""
graph/extractor.py  —  DeepChain Hybrid-RAG
Module: Triplet Extractor (Core LLM Logic)

FIXES v3:
  - 429 RESOURCE_EXHAUSTED: parse retryDelay from error message and sleep
    for exactly as long as Google tells us to (+ small jitter).
  - On sustained 429 (quota exhausted for the day), auto-switch to next
    fallback model instead of hammering the same model.
  - FALLBACK_MODELS ordered by free-tier generosity:
      gemini-2.0-flash → gemini-1.5-flash → gemini-1.5-pro → gemini-pro
  - 404 NOT_FOUND: immediate model switch, no sleep.
  - Transient errors (5xx, network): exponential backoff as before.
  - Improved prompt with entity-type tags and UPPER_SNAKE_CASE predicates.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# ── Retry / rate-limit config ─────────────────────────────────────────────────
MAX_RETRIES        = 4
BACKOFF_BASE       = 2       # seconds for transient errors
DEFAULT_429_WAIT   = 30      # fallback wait if retryDelay not in error message
MAX_429_WAIT       = 120     # cap on any single sleep
# How many 429s on one model before we give up and switch fallback
MAX_QUOTA_HITS     = 2
SKIP_LOG_PATH      = Path("extraction_errors.jsonl")

# ── Prompt ────────────────────────────────────────────────────────────────────
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


# ── Error classification helpers ───────────────────────────────────────────────

def _is_model_not_found(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "404" in msg or "NOT_FOUND" in msg or "NOT FOUND" in msg


def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg


def _parse_retry_delay(exc: Exception) -> float:
    """
    Extract the retryDelay seconds Google embeds in 429 error messages.
    Falls back to DEFAULT_429_WAIT if not found.
    Example: "Please retry in 19.126905886s."
    """
    text = str(exc)
    # Try JSON field first: 'retryDelay': '19s'
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s?['\"]", text)
    if m:
        return min(float(m.group(1)) + 2, MAX_429_WAIT)   # +2s jitter
    # Try prose form: "retry in 19.1s" or "retry in 19s"
    m = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 2, MAX_429_WAIT)
    return DEFAULT_429_WAIT


class TripletExtractor:
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI | None = None,
        model_name: str | None = None,
    ) -> None:
        self._model_name  = model_name or PRIMARY_MODEL
        self._quota_hits  = 0    # consecutive 429s on current model
        self._build_llm(self._model_name, llm)

    # ── LLM management ────────────────────────────────────────────────────────

    def _build_llm(self, model_name: str, llm=None) -> None:
        self.llm         = llm or ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self._model_name = model_name
        self._quota_hits = 0
        logger.info("[TripletExtractor] Active model: %s", model_name)

    def _switch_to_fallback(self, failed_model: str, reason: str) -> bool:
        candidates = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))
        try:
            idx = candidates.index(failed_model)
        except ValueError:
            idx = -1
        for model in candidates[idx + 1:]:
            if model != failed_model:
                logger.warning("[TripletExtractor] %s — switching '%s' → '%s'",
                               reason, failed_model, model)
                self._build_llm(model)
                return True
        logger.error("[TripletExtractor] All fallback models exhausted (%s).", reason)
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
                content  = response.content
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                self._quota_hits = 0   # reset on success
                return self._parse_response(content, chunk)

            except json.JSONDecodeError as exc:
                logger.warning("JSON parse error chunk %s attempt %d: %s",
                               chunk.get("chunk_id"), attempt, exc)
                self._log_skip(chunk, reason=f"json_parse_error: {exc}")
                return []

            except Exception as exc:  # noqa: BLE001

                # ── 404: model not found → switch immediately ─────────────
                if _is_model_not_found(exc):
                    switched = self._switch_to_fallback(self._model_name, "404 NOT_FOUND")
                    if not switched:
                        return None
                    continue   # retry same attempt counter with new model

                # ── 429: quota exhausted → wait, then maybe switch ────────
                if _is_quota_exhausted(exc):
                    self._quota_hits += 1
                    wait = _parse_retry_delay(exc)
                    logger.warning(
                        "[TripletExtractor] 429 quota hit #%d on '%s'. "
                        "Sleeping %.0fs (Google retryDelay)...",
                        self._quota_hits, self._model_name, wait,
                    )
                    time.sleep(wait)

                    if self._quota_hits >= MAX_QUOTA_HITS:
                        switched = self._switch_to_fallback(
                            self._model_name,
                            f"quota exhausted after {self._quota_hits} hits"
                        )
                        if not switched:
                            return None
                    continue   # retry (possibly with new model)

                # ── Transient error → exponential backoff ─────────────────
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "API error chunk %s attempt %d/%d: %s — retry in %ds",
                    chunk.get("chunk_id"), attempt, MAX_RETRIES, exc, wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                else:
                    logger.error("Giving up on chunk %s after %d attempts",
                                 chunk.get("chunk_id"), MAX_RETRIES)
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

"""
graph/extractor.py  —  DeepChain Hybrid-RAG
Module: Triplet Extractor (Core LLM Logic)

Extracts {subject, predicate, object} triplets from text chunks.
Includes exponential backoff retry logic and JSON sanitization.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES = 3
BACKOFF_BASE = 2          # seconds; doubles each attempt: 2, 4, 8
SKIP_LOG_PATH = Path("extraction_errors.jsonl")

# ── Prompt ────────────────────────────────────────────────────────────────────
TRIPLET_PROMPT = """Extract all factual relationships from the text below.
Return ONLY a JSON array of objects, no prose, no markdown fences.
Each object must have exactly three keys: "subject", "predicate", "object".
If no relationships exist return an empty array [].

Text:
{text}
"""


class TripletExtractor:
    def __init__(self, llm: ChatGoogleGenerativeAI | None = None, model_name: str = "gemini-flash-latest") -> None:
        self.llm = llm or ChatGoogleGenerativeAI(model=model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Extract triplets from a list of chunk dicts.
        Each chunk dict must have at least: {"text": str, "chunk_id": str}

        Returns flat list of {subject, predicate, object, source_chunk_id}.
        Failed chunks are skipped and logged — never raise mid-pipeline.
        """
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
            "Triplet extraction complete — chunks=%d ok=%d failed=%d triplets=%d",
            len(chunks),
            len(chunks) - failed,
            failed,
            len(all_triplets),
        )
        return all_triplets

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_with_retry(
        self, chunk: dict[str, Any]
    ) -> list[dict[str, str]] | None:
        """
        Call Gemini with exponential backoff.
        Returns parsed triplet list, or None if all retries exhausted.
        """
        text = chunk.get("text", "").strip()
        if not text:
            logger.debug("Skipping empty chunk %s", chunk.get("chunk_id"))
            return []

        prompt = TRIPLET_PROMPT.format(text=text)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return self._parse_response(response.content, chunk)

            except json.JSONDecodeError as exc:
                # LLM returned non-JSON — log and skip (no retry; same prompt will fail again)
                logger.warning(
                    "JSON parse error on chunk %s (attempt %d): %s",
                    chunk.get("chunk_id"),
                    attempt,
                    exc,
                )
                self._log_skip(chunk, reason=f"json_parse_error: {exc}")
                return []  # treat as zero triplets, not a fatal error

            except Exception as exc:  # noqa: BLE001  (broad — covers API errors)
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "API error on chunk %s (attempt %d/%d): %s — retrying in %ds",
                    chunk.get("chunk_id"),
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                else:
                    logger.error(
                        "Giving up on chunk %s after %d attempts",
                        chunk.get("chunk_id"),
                        MAX_RETRIES,
                    )
                    return None  # caller will log skip

        return None  # unreachable but satisfies type checker

    def _parse_response(
        self, raw: str, chunk: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Strip markdown fences if present, then parse JSON."""
        cleaned = raw.strip()
        # Strip ```json ... ``` wrappers that Gemini sometimes adds
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
            # Fallback: if there's text after the array, try to find the array boundaries
            if "[" in cleaned and "]" in cleaned:
                start = cleaned.find("[")
                end = cleaned.rfind("]") + 1
                try:
                    parsed = json.loads(cleaned[start:end])
                except:
                    raise
            else:
                raise

        if not isinstance(parsed, list):
            logger.warning(
                "Unexpected LLM output type %s on chunk %s — expected list",
                type(parsed).__name__,
                chunk.get("chunk_id"),
            )
            return []

        # Validate each triplet has the required keys
        valid = []
        for item in parsed:
            if all(k in item for k in ("subject", "predicate", "object")):
                valid.append(
                    {
                        "subject": str(item["subject"]).strip(),
                        "predicate": str(item["predicate"]).strip(),
                        "object": str(item["object"]).strip(),
                    }
                )
            else:
                logger.debug("Dropping malformed triplet: %s", item)

        return valid

    def _log_skip(self, chunk: dict[str, Any], reason: str) -> None:
        """Append failed chunk metadata to JSONL skip-log for later re-processing."""
        record = {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "source": chunk.get("source", "unknown"),
            "reason": reason,
            "timestamp": time.time(),
        }
        with SKIP_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Logged skip: %s", record)
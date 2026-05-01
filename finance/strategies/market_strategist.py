"""
finance/strategies/market_strategist.py — DeepChain
Module: Market Strategy Advisor

IMPROVEMENTS:
  - Inherits _safe_generate() and fallback logic from updated StrategyAdvisor.
  - analyze_and_build() returns structured JSON (not just raw text).
  - Added 'signal_summary', 'strategy_type', and 'risk_level' to output.
  - Market summary now includes RSI proxy and momentum indicators.
  - Handles empty/missing yfinance data gracefully with a clear error message.
"""

import os
import logging
from typing import Dict, Any

import yfinance as yf
import pandas as pd
import numpy as np

from .advisor import StrategyAdvisor

logger = logging.getLogger(__name__)


class MarketStrategyAdvisor(StrategyAdvisor):
    """
    Analyzes live market data and selects/combines strategies from the knowledge base.
    """

    def get_live_data(self, symbol: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
        logger.info("[MarketStrategyAdvisor] Fetching live data for %s ...", symbol)
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data found for symbol '{symbol}'")
            return df
        except Exception as e:
            logger.error("[MarketStrategyAdvisor] Data fetch error: %s", e)
            raise

    def _build_market_summary(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute enriched market indicators from OHLCV data."""
        close  = df["Close"]
        volume = df["Volume"]

        # Simple RSI proxy (14-period)
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, float("nan"))
        rsi    = float((100 - 100 / (1 + rs)).iloc[-1]) if not rs.iloc[-1] != rs.iloc[-1] else 50.0

        # Moving averages
        sma20  = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean())
        sma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())

        return {
            "symbol":         symbol,
            "current_price":  round(float(close.iloc[-1]), 4),
            "30d_high":       round(float(df["High"].iloc[-30:].max()), 4),
            "30d_low":        round(float(df["Low"].iloc[-30:].min()),  4),
            "avg_volume_30d": int(volume.iloc[-30:].mean()),
            "recent_trend":   "Bullish" if float(close.iloc[-1]) > float(close.iloc[-30]) else "Bearish",
            "volatility_30d": round(float(close.iloc[-30:].std()), 4),
            "rsi_14":         round(rsi, 2),
            "sma_20":         round(sma20, 4),
            "sma_50":         round(sma50, 4),
            "above_sma20":    float(close.iloc[-1]) > sma20,
            "above_sma50":    float(close.iloc[-1]) > sma50,
            "momentum_signal": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
        }

    async def analyze_and_build(self, symbol: str) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Fetch live data  →  2. Compute indicators  →  3. RAG retrieval
        4. Generate structured strategy  →  5. Return plan
        """
        # 1. Fetch & summarize
        try:
            df = self.get_live_data(symbol)
        except Exception as e:
            return {"symbol": symbol, "error": str(e), "market_summary": {}, "dynamic_report": ""}

        market_summary = self._build_market_summary(symbol, df)

        # 2. RAG context
        retriever = self._get_retriever()
        context   = ""
        if retriever:
            try:
                rag_result = await retriever.query(
                    question=(
                        f"Which of the 10 trading strategies is best for a "
                        f"{market_summary['recent_trend']} market with "
                        f"RSI={market_summary['rsi_14']} and "
                        f"volatility={market_summary['volatility_30d']}?"
                    ),
                    mode="hybrid",
                )
                context = rag_result.answer[:4000]
            except Exception as e:
                logger.warning("[MarketStrategyAdvisor] RAG failed: %s", e)

        # 3. Generate strategy
        prompt = f"""
System: You are a Lead Quant Analyst for DeepChain.

Live Market Data for {symbol}:
{market_summary}

Knowledge Base Context:
{context}

Task:
1. Based on the live indicators, select one of the 10 known strategies OR propose a hybrid combination.
2. Explain WHY this strategy fits the current market conditions for {symbol}.
3. Provide the full Python implementation code.

Return ONLY a valid JSON object with these keys:
{{
  "strategy_name": "...",
  "strategy_type": "trend_following|mean_reversion|breakout|hybrid|...",
  "risk_level": "low|medium|high",
  "signal_summary": "brief buy/sell/hold recommendation for right now",
  "rationale": "...",
  "implementation_code": "...(def strategy_function(df): ... returns df with 'signals' col)",
  "entry_exit_logic": "...",
  "risk_management": "...",
  "warnings": ["..."]
}}

IMPORTANT: Return ONLY the JSON. No markdown fences. No extra text.
The implementation_code MUST use signature `def strategy_function(df):` and return the dataframe with a 'signals' column (1=buy, -1=sell, 0=hold).
"""

        raw = self._safe_generate(prompt)

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
            structured = {"rationale": raw, "implementation_code": "", "warnings": ["JSON parsing failed"]}

        return {
            "symbol":           symbol,
            "market_summary":   market_summary,
            "dynamic_report":   raw,
            "structured":       structured,
            "retrieved_context": context,
            "model_used":       self._model_name,
        }


if __name__ == "__main__":
    import asyncio
    async def test():
        advisor = MarketStrategyAdvisor()
        result  = await advisor.analyze_and_build("TSLA")
        print(f"--- {result['symbol']} ---")
        print(result["dynamic_report"])
    asyncio.run(test())

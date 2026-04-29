import logging
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from .backtester import StrategyBacktester, sma_crossover_strategy, rsi_strategy
from .ml_evaluator import StrategyMLEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeTestingPipeline:
    """
    Orchestrates the Trade Testing process: Data Fetching -> Backtesting -> ML Evaluation.
    """
    def __init__(self):
        self.backtester = StrategyBacktester()
        self.evaluator = StrategyMLEvaluator()
        self.strategies = {
            "SMA_Crossover": sma_crossover_strategy,
            "RSI_Standard": rsi_strategy
        }

    def run_test(self, symbol: str, strategy_name: str, period: str = "1y") -> Dict[str, Any]:
        """
        Runs a full test for a given symbol and strategy.
        """
        logger.info(f"--- Starting Trade Test: {symbol} | {strategy_name} ---")
        
        # 1. Fetch Historical Data
        logger.info(f"Step 1: Fetching {period} historical data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return {"success": False, "error": f"No data found for {symbol}"}

        # 2. Select Strategy
        strategy_func = self.strategies.get(strategy_name)
        if not strategy_func:
            return {"success": False, "error": f"Strategy {strategy_name} not found"}

        # 3. Backtest
        logger.info("Step 2: Running backtest...")
        backtest_results = self.backtester.backtest(df, strategy_func)

        # 4. ML Evaluation
        logger.info("Step 3: Running ML evaluation...")
        ml_results = self.evaluator.evaluate(backtest_results)

        logger.info("--- Trade Test Complete ---")

        return {
            "success": True,
            "symbol": symbol,
            "strategy": strategy_name,
            "period": period,
            "backtest": backtest_results,
            "ml_evaluation": ml_results
        }

if __name__ == "__main__":
    # Integration Test
    pipeline = TradeTestingPipeline()
    
    # Test Reliance on NSE
    result = pipeline.run_test("RELIANCE.NS", "SMA_Crossover")
    print("\n--- TEST RESULT ---")
    print(json.dumps(result, indent=4))

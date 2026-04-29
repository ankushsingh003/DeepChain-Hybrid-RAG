import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyMLEvaluator:
    """
    Evaluates the viability of a trading strategy using a pre-trained ML model.
    If no model is found, it uses a heuristic scoring system based on backtest metrics.
    """
    def __init__(self, model_path: str = "models/strategy_scorer.joblib"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning(f"No model found at {self.model_path}. Using heuristic scoring.")
        return None

    def evaluate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scores a strategy based on backtest performance.
        """
        if self.model:
            # Prepare features for the model
            features = np.array([[
                backtest_results['total_return_pct'],
                backtest_results['sharpe_ratio'],
                backtest_results['max_drawdown_pct'],
                backtest_results['win_rate_pct']
            ]])
            # Mock prediction (replace with actual model logic)
            confidence = self.model.predict_proba(features)[0][1] 
            score = self.model.predict(features)[0]
        else:
            # Heuristic Scoring (Logic mimicry)
            # A good strategy has:
            # 1. Positive Sharpe Ratio (> 1.0 is good)
            # 2. Win Rate > 50%
            # 3. Max Drawdown < 20%
            
            sharpe = backtest_results['sharpe_ratio']
            win_rate = backtest_results['win_rate_pct']
            drawdown = abs(backtest_results['max_drawdown_pct'])
            
            score_val = 0
            if sharpe > 1.5: score_val += 40
            elif sharpe > 0.8: score_val += 20
            
            if win_rate > 55: score_val += 30
            elif win_rate > 45: score_val += 15
            
            if drawdown < 10: score_val += 30
            elif drawdown < 20: score_val += 15
            
            confidence = min(1.0, score_val / 100)
            
            if confidence > 0.7: recommendation = "Highly Recommended"
            elif confidence > 0.4: recommendation = "Neutral / Moderate"
            else: recommendation = "Not Recommended (High Risk)"

        return {
            "ml_score": round(confidence * 10, 2), # Scale of 1-10
            "confidence_pct": round(confidence * 100, 2),
            "recommendation": recommendation,
            "using_ml_model": self.model is not None
        }

if __name__ == "__main__":
    # Test Evaluator
    evaluator = StrategyMLEvaluator()
    mock_results = {
        "total_return_pct": 25.5,
        "sharpe_ratio": 1.2,
        "max_drawdown_pct": -12.0,
        "win_rate_pct": 58.0
    }
    print(evaluator.evaluate(mock_results))

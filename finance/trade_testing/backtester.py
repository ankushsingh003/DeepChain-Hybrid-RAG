import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyBacktester:
    """
    Core engine for backtesting trading strategies against historical data.
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def backtest(self, df: pd.DataFrame, strategy_func: Callable) -> Dict[str, Any]:
        """
        Runs a strategy function on historical data and calculates performance.
        strategy_func should take a DataFrame and return a 'signals' column (1 for Buy, -1 for Sell, 0 for Hold).
        """
        logger.info(f"Starting backtest with initial capital: {self.initial_capital}")
        
        # Ensure we have a copy to avoid side effects
        data = df.copy()
        
        # 1. Generate Signals
        data = strategy_func(data)
        
        if 'signals' not in data.columns:
            raise ValueError("Strategy function must return a DataFrame with a 'signals' column.")

        # 2. Calculate Returns
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['strategy_returns'] = data['signals'].shift(1) * data['log_returns']
        
        # 3. Performance Metrics
        cumulative_returns = np.exp(data['strategy_returns'].cumsum())
        total_return = cumulative_returns.iloc[-1] - 1 if not cumulative_returns.empty else 0
        
        # Sharpe Ratio (annualized)
        std_dev = data['strategy_returns'].std()
        sharpe_ratio = (data['strategy_returns'].mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
        
        # Max Drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win Rate
        trades = data[data['signals'] != 0]
        if len(trades) > 1:
            trade_returns = data['strategy_returns'][data['signals'].shift(1) != 0]
            win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) if len(trade_returns) > 0 else 0
        else:
            win_rate = 0

        return {
            "initial_capital": self.initial_capital,
            "final_value": self.initial_capital * (1 + total_return),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "trades_count": int(abs(data['signals'].diff()).sum() / 2) # Rough estimate of full trades
        }

# --- Predefined Strategy Functions ---

def sma_crossover_strategy(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Buy when short SMA crosses above long SMA, Sell when it crosses below.
    """
    df['sma_short'] = df['Close'].rolling(window=short_window).mean()
    df['sma_long'] = df['Close'].rolling(window=long_window).mean()
    
    df['signals'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signals'] = 1
    df.loc[df['sma_short'] < df['sma_long'], 'signals'] = -1
    return df

def rsi_strategy(df: pd.DataFrame, period: int = 14, overbought: int = 70, oversold: int = 30) -> pd.DataFrame:
    """
    Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought).
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['signals'] = 0
    df.loc[df['rsi'] < oversold, 'signals'] = 1
    df.loc[df['rsi'] > overbought, 'signals'] = -1
    return df

if __name__ == "__main__":
    # Test with mock data
    dates = pd.date_range(start="2023-01-01", periods=200)
    prices = 100 + np.cumsum(np.random.randn(200))
    mock_df = pd.DataFrame({'Close': prices}, index=dates)
    
    tester = StrategyBacktester()
    
    print("\n--- Testing SMA Crossover ---")
    res_sma = tester.backtest(mock_df, sma_crossover_strategy)
    print(res_sma)
    
    print("\n--- Testing RSI Strategy ---")
    res_rsi = tester.backtest(mock_df, rsi_strategy)
    print(res_rsi)

# Trading Strategy Knowledge Base

This document contains 10 core trading strategies used by the DeepChain Strategy Advisor.

## 1. Moving Average Crossover (SMA/EMA)
- **Type**: Trend Following
- **Logic**: Uses two moving averages (Short and Long). Buy when the Short MA crosses above the Long MA. Sell when it crosses below.
- **Formula**: `SMA = (Sum of Close Prices over N periods) / N`
- **Code Approach**:
```python
def sma_crossover(df, short=20, long=50):
    df['sma_s'] = df['Close'].rolling(window=short).mean()
    df['sma_l'] = df['Close'].rolling(window=long).mean()
    df['signals'] = 0
    df.loc[df['sma_s'] > df['sma_l'], 'signals'] = 1
    df.loc[df['sma_s'] < df['sma_l'], 'signals'] = -1
    return df
```

## 2. RSI Mean Reversion
- **Type**: Mean Reversion
- **Logic**: Uses the Relative Strength Index (RSI). Buy when RSI is below 30 (oversold) and sell when above 70 (overbought).
- **Formula**: `RSI = 100 - [100 / (1 + (Avg Gain / Avg Loss))]`
- **Code Approach**:
```python
def rsi_strategy(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['signals'] = 0
    df.loc[df['rsi'] < 30, 'signals'] = 1
    df.loc[df['rsi'] > 70, 'signals'] = -1
    return df
```

## 3. Bollinger Bands Breakout
- **Type**: Volatility Breakout
- **Logic**: Uses a central SMA and two standard deviation bands. Buy when price touches the lower band and starts moving up. Sell when it touches the upper band.
- **Formula**: `Upper = SMA + (2 * StdDev)`, `Lower = SMA - (2 * StdDev)`
- **Code Approach**:
```python
def bollinger_bands(df, n=20, k=2):
    df['sma'] = df['Close'].rolling(n).mean()
    df['std'] = df['Close'].rolling(n).std()
    df['upper'] = df['sma'] + (k * df['std'])
    df['lower'] = df['sma'] - (k * df['std'])
    df['signals'] = 0
    df.loc[df['Close'] < df['lower'], 'signals'] = 1
    df.loc[df['Close'] > df['upper'], 'signals'] = -1
    return df
```

## 4. MACD Divergence
- **Type**: Momentum / Trend Reversal
- **Logic**: Uses the difference between two EMAs. Buy when MACD line crosses above the Signal line.
- **Formula**: `MACD = EMA(12) - EMA(26)`, `Signal = EMA(MACD, 9)`
- **Code Approach**:
```python
def macd_strategy(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['signals'] = 0
    df.loc[df['macd'] > df['signal'], 'signals'] = 1
    df.loc[df['macd'] < df['signal'], 'signals'] = -1
    return df
```

## 5. Mean Reversion (Z-Score)
- **Type**: Statistical Arbitrage
- **Logic**: Measures how many standard deviations the price is from the mean. Buy if Z-Score < -2, Sell if Z-Score > 2.
- **Formula**: `Z = (Price - Mean) / StdDev`
- **Code Approach**:
```python
def z_score_strategy(df, window=20):
    df['mean'] = df['Close'].rolling(window).mean()
    df['std'] = df['Close'].rolling(window).std()
    df['z_score'] = (df['Close'] - df['mean']) / df['std']
    df['signals'] = 0
    df.loc[df['z_score'] < -2, 'signals'] = 1
    df.loc[df['z_score'] > 2, 'signals'] = -1
    return df
```

## 6. Momentum (Rate of Change)
- **Type**: Momentum
- **Logic**: Measures the percentage change in price over a period. Buy if ROC is positive and increasing.
- **Formula**: `ROC = [(Close - Close_n) / Close_n] * 100`
- **Code Approach**:
```python
def roc_strategy(df, n=12):
    df['roc'] = ((df['Close'] - df['Close'].shift(n)) / df['Close'].shift(n)) * 100
    df['signals'] = 0
    df.loc[df['roc'] > 0, 'signals'] = 1
    df.loc[df['roc'] < 0, 'signals'] = -1
    return df
```

## 7. Parabolic SAR
- **Type**: Trend Reversal / Stop-Loss
- **Logic**: A trailing stop-loss that flips based on price action. Buy when dots move below the price.
- **Formula**: `SAR_{n+1} = SAR_n + AF * (EP - SAR_n)`
- **Code Approach**: Typically implemented iteratively. High-level: Signals flip when price crosses the SAR dot.

## 8. Stochastic Oscillator
- **Type**: Precision Range
- **Logic**: Compares a closing price to its price range over a period.
- **Formula**: `%K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100`
- **Code Approach**:
```python
def stochastic_strategy(df, k=14, d=3):
    df['low_min'] = df['Low'].rolling(window=k).min()
    df['high_max'] = df['High'].rolling(window=k).max()
    df['%K'] = (df['Close'] - df['low_min']) / (df['high_max'] - df['low_min']) * 100
    df['%D'] = df['%K'].rolling(window=d).mean()
    df['signals'] = 0
    df.loc[df['%K'] < 20, 'signals'] = 1
    df.loc[df['%K'] > 80, 'signals'] = -1
    return df
```

## 9. VWAP (Volume Weighted Average Price)
- **Type**: Institutional / Intraday
- **Logic**: The average price weighted by volume. Buy when price is below VWAP (undervalued), Sell when above.
- **Formula**: `VWAP = Sum(Price * Volume) / Sum(Volume)`
- **Code Approach**:
```python
def vwap_strategy(df):
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['signals'] = 0
    df.loc[df['Close'] < df['vwap'], 'signals'] = 1
    df.loc[df['Close'] > df['vwap'], 'signals'] = -1
    return df
```

## 10. Dual Thrust
- **Type**: Opening Range Breakout
- **Logic**: Uses a range based on previous days' highs/lows.
- **Formula**: `Range = Max(HH-LC, HC-LL)`, `BuyLine = Open + K1*Range`, `SellLine = Open - K2*Range`
- **Code Approach**: Calculates a fixed range at the start of the period and sets entry triggers.

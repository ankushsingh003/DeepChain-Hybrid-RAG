import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECTOR_MAP = {
    "Nifty IT": {"symbol": "^CNXIT", "top_stocks": ["TCS.NS", "INFY.NS", "WIPRO.NS"]},
    "Nifty Bank": {"symbol": "^NSEBANK", "top_stocks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]},
    "Nifty Auto": {"symbol": "^CNXAUTO", "top_stocks": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"]},
    "Nifty Pharma": {"symbol": "^CNXPHARMA", "top_stocks": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"]},
    "Nifty FMCG": {"symbol": "^CNXFMCG", "top_stocks": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]},
    "Nifty Energy": {"symbol": "^CNXENERGY", "top_stocks": ["RELIANCE.NS", "NTPC.NS", "ONGC.NS"]},
    "Nifty Infra": {"symbol": "^CNXINFRA", "top_stocks": ["LT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS"]},
    "Nifty Metal": {"symbol": "^CNXMETAL", "top_stocks": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"]},
    "Nifty Realty": {"symbol": "^CNXREALTY", "top_stocks": ["DLF.NS", "GODREJPROP.NS", "OBEROREALTY.NS"]}
}

def get_proxy_pe_div(stocks):
    """
    Calculates average PE and Dividend Yield from a list of top stocks as a proxy for the sector.
    """
    pes = []
    divs = []
    for s in stocks:
        try:
            t = yf.Ticker(s)
            info = t.info
            pe = info.get('trailingPE')
            dy = info.get('dividendYield')
            if pe: pes.append(pe)
            if dy: divs.append(dy * 100)
        except:
            continue
    
    avg_pe = sum(pes) / len(pes) if pes else 0.0
    avg_div = sum(divs) / len(divs) if divs else 0.0
    return round(avg_pe, 2), round(avg_div, 2)

def fetch_sector_data():
    """
    Fetches real-time and historical data for Indian market sectors.
    """
    results = {}
    
    for sector_name, config in SECTOR_MAP.items():
        symbol = config["symbol"]
        stocks = config["top_stocks"]
        try:
            logger.info(f"Fetching data for {sector_name} ({symbol})...")
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist = ticker.history(period="1y")
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                continue
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            pct_change = ((current_price - prev_close) / prev_close) * 100
            
            # 52-week range
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            position_52w = (current_price - low_52w) / (high_52w - low_52w) if (high_52w - low_52w) != 0 else 0
            
            # Momentum calculations
            def calculate_momentum(days):
                if len(hist) < days: return 0.0
                past_price = hist['Close'].iloc[-days]
                return ((current_price - past_price) / past_price) * 100

            m1 = calculate_momentum(21)
            m3 = calculate_momentum(63)
            m6 = calculate_momentum(126)
            
            # Get Proxy PE and Dividend Yield from top stocks
            pe_ratio, div_yield = get_proxy_pe_div(stocks)
            
            # Placeholder for FII/DII flows (requires specialized API or scraping)
            fii_flow_1m = 0.0 
            
            results[sector_name] = {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "pct_change": round(pct_change, 2),
                "pe_ratio": pe_ratio if pe_ratio > 0 else "N/A",
                "momentum_1m": round(m1, 2),
                "momentum_3m": round(m3, 2),
                "momentum_6m": round(m6, 2),
                "52w_position": round(position_52w, 2),
                "fii_flow_1m": fii_flow_1m,
                "dividend_yield": div_yield
            }
            logger.info(f"Completed {sector_name}")
            
        except Exception as e:
            logger.error(f"Error fetching {sector_name}: {e}")
            
    return results

if __name__ == "__main__":
    data = fetch_sector_data()
    print(json.dumps(data, indent=4))

from nsepy import get_index_pe_pb_div
from datetime import date, timedelta
import pandas as pd

try:
    # Try to get PE for Nifty 50 as a test
    end_date = date.today()
    start_date = end_date - timedelta(days=5)
    df = get_index_pe_pb_div(symbol="NIFTY 50", start=start_date, end=end_date)
    print("NIFTY 50 PE Data:")
    print(df)
except Exception as e:
    print(f"Error fetching from nsepy: {e}")

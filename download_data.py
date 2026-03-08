import yfinance as yf
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
TICKER_SYMBOL = "^BSESN"  # BSE SENSEX Symbol
START_DATE = "2010-01-01"
END_DATE = "2026-02-7"   
OUTPUT_FILE = "sensex_data_clean.csv"

def download_data(ticker, start, end):
    """
    Fetches historical data from Yahoo Finance and fixes MultiIndex issues.
    """
    print(f"[INFO] Downloading data for {ticker}...")
    try:
        # auto_adjust=True fixes the Close vs Adj Close confusion
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, multi_level_index=False)
        
        # Check if data is empty
        if df.empty:
            print("[ERROR] No data found. Check internet or ticker.")
            return None
            
        # Reset index so 'Date' becomes a proper column, not an index
        df = df.reset_index()
        
        print(f"[SUCCESS] Download complete! Retrieved {len(df)} days of data.")
        return df
    
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        return None

def clean_data(df):
    """
    Removes null values.
    """
    print("[INFO] Cleaning data...")
    
    # 1. Drop rows where any column is missing
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    
    print(f"   - Removed {initial_count - final_count} empty rows.")
    return df

def add_technical_indicators(df):
    """
    Engineering Features: SMA and RSI.
    """
    print("[INFO] Engineering features (RSI, SMA)...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate indicators
    # 1. SMA (Simple Moving Average) - 50 Days
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 2. SMA (Simple Moving Average) - 200 Days
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 3. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop NaN values created by the calculations
    df = df.dropna()
    
    print(f"[SUCCESS] Features added. Final dataset size: {df.shape}")
    return df

if __name__ == "__main__":
    # 1. Download
    raw_data = download_data(TICKER_SYMBOL, START_DATE, END_DATE)
    
    if raw_data is not None:
        # 2. Clean
        clean_data_df = clean_data(raw_data)
        
        # 3. Add Features
        final_data = add_technical_indicators(clean_data_df)
        
        # 4. Save to CSV
        final_data.to_csv(OUTPUT_FILE)
        print(f"[DONE] Clean data saved to: {OUTPUT_FILE}")
        print("       Open this file in Excel to inspect.")
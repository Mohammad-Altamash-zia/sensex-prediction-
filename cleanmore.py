import pandas as pd

# 1. Load the messy file
print("[INFO] Loading messy data...")
# We read it as a normal CSV first
df = pd.read_csv("sensex_data_clean.csv")

# 2. Check if the 'Price' column is actually the Date (The common bug)
if 'Price' in df.columns and df.iloc[0]['Price'] == 'Ticker':
    print("[INFO] Detected Multi-Level Header Bug. Fixing...")
    
    # Remove the first two rows (which contain 'Ticker' and empty 'Date' info)
    df_clean = df.iloc[2:].reset_index(drop=True)
    
    # Rename the first column from 'Price' to 'Date'
    df_clean.rename(columns={'Price': 'Date'}, inplace=True)
    
    # Convert all numeric columns to numbers (floats)
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_50', 'SMA_200', 'RSI']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col])
        
    # Round to 2 decimal places for a professional look
    df_clean = df_clean.round(2)
    
    # Save the Fixed File
    df_clean.to_csv("sensex_data_final.csv", index=False)
    print("[SUCCESS] Data cleaned! Saved as 'sensex_data_final.csv'")
    print(df_clean.head())

else:
    print("[INFO] Data looks fine. No fix needed.")
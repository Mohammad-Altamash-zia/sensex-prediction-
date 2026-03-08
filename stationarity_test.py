import pandas as pd
from statsmodels.tsa.stattools import adfuller

# --- CONFIGURATION ---
INPUT_FILE = "sensex_data_perfect.csv"

def run_adf_test(filepath):
    print("[INFO] Loading clean dataset for Stationarity Test...")
    df = pd.read_csv(filepath)
    
    # We test the 'Close' price
    close_prices = df['Close'].values
    
    print("[INFO] Running Augmented Dickey-Fuller (ADF) Test...")
    # This is the complex math function that calculates stationarity
    result = adfuller(close_prices, autolag='AIC')
    
    # Extracting the results
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Printing the results like a professional statistical report
    print("-" * 40)
    print(" ADF Test Results for Sensex")
    print("-" * 40)
    print(f"ADF Statistic:   {adf_statistic:.4f}")
    print(f"p-value:         {p_value:.4f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value:.4f}")
    print("-" * 40)
    
    # The AI Engineer's Interpretation Logic
    if p_value < 0.05:
        print("[CONCLUSION] The data is STATIONARY (p < 0.05).")
        print("Meaning: The market has no long-term trend. (Very rare for stocks!).")
    else:
        print("[CONCLUSION] The data is NON-STATIONARY (p >= 0.05).")
        print("Meaning: The market has a clear trend over time.")
        print("Next Step: This proves we need a powerful model like LSTM to handle the trend!")

if __name__ == "__main__":
    try:
        run_adf_test(INPUT_FILE)
    except Exception as e:
        print(f"[ERROR] Did you forget to install statsmodels? {e}")
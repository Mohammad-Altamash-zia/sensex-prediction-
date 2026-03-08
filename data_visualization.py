import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = "sensex_data_clean.csv" 
OUTPUT_FOLDER = "project_graphs"

def clean_and_engineer_data(filepath):
    """
    Step 1: Data Engineering
    Loads the messy CSV, fixes the yfinance header bug, 
    and ensures the data types are correct for the AI model.
    """
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Auto-Fix for the multi-header bug
    if 'Price' in df.columns and str(df.iloc[0]['Price']) == 'Ticker':
        print("[INFO] Cleaning messy headers...")
        # Drop the first 2 garbage rows
        df = df.iloc[2:].reset_index(drop=True)
        # Rename the 'Price' column to 'Date'
        df.rename(columns={'Price': 'Date'}, inplace=True)
        
        # Convert columns from text to numbers
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_50', 'SMA_200', 'RSI']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
            
    # Convert 'Date' column to actual Python Date objects
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Save a perfect version for the AI team to use later
    df.to_csv("sensex_data_perfect.csv", index=False)
    print("[SUCCESS] Data engineered and saved as 'sensex_data_perfect.csv'.")
    
    return df

def plot_moving_averages(df):
    """
    Step 2: Visualization 1
    Creates the 'Trend' Chart showing Close Price vs Moving Averages.
    """
    print("[INFO] Generating Moving Average Trend Chart...")
    plt.figure(figsize=(14, 7)) # Wide, professional aspect ratio
    
    # Plotting the lines
    plt.plot(df['Date'], df['Close'], label='Actual Close Price', color='black', alpha=0.6, linewidth=1.5)
    plt.plot(df['Date'], df['SMA_50'], label='50-Day Moving Average', color='blue', linewidth=2)
    plt.plot(df['Date'], df['SMA_200'], label='200-Day Moving Average', color='red', linewidth=2)
    
    # Formatting
    plt.title('BSE Sensex: Price History & Moving Averages', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price (INR)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the image
    plt.savefig(f"{OUTPUT_FOLDER}/sensex_trend_chart.png", bbox_inches='tight')
    plt.close()

def plot_rsi(df):
    """
    Step 3: Visualization 2
    Creates the 'Momentum' Chart showing the RSI indicator.
    """
    print("[INFO] Generating RSI Chart...")
    # Filter for the last few years so the chart isn't too squished
    recent_df = df[df['Date'] > '2023-01-01'] 
    
    plt.figure(figsize=(14, 4))
    plt.plot(recent_df['Date'], recent_df['RSI'], label='RSI (14-Day)', color='purple', linewidth=1.5)
    
    # Overbought and Oversold threshold lines
    plt.axhline(70, color='red', linestyle='--', label='Overbought (>70: Sell Alert)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (<30: Buy Alert)')
    
    plt.title('Relative Strength Index (RSI) - Since 2023', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"{OUTPUT_FOLDER}/sensex_rsi_chart.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create the folder for the graphs
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    try:
        # Run the professional pipeline
        clean_df = clean_and_engineer_data(INPUT_FILE)
        plot_moving_averages(clean_df)
        plot_rsi(clean_df)
        print(f"[SUCCESS] All tasks completed! Check the '{OUTPUT_FOLDER}' folder for your charts.")
    except Exception as e:
        print(f"[ERROR] {e}")
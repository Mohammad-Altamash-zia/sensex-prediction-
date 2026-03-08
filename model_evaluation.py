import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- CONFIGURATION ---
INPUT_FILE = "sensex_data_perfect.csv"
MODEL_FOLDER = "model_data"
# Change to .h5 if your team didn't use the .keras format!
MODEL_FILE = "sensex_model.h5" 
SCALER_FILE = "price_scaler.pkl"
OUTPUT_FOLDER = "project_graphs"
WINDOW_SIZE = 60
TEST_DAYS = 250 # We will test the AI on the last ~1 year of trading days

def evaluate_ai():
    print("[INFO] Loading the AI Brain and Scaler...")
    try:
        # Load the saved model and scaler
        model = load_model(f"{MODEL_FOLDER}/{MODEL_FILE}")
        scaler = joblib.load(f"{MODEL_FOLDER}/{SCALER_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not load model or scaler. Error: {e}")
        return

    print("[INFO] Loading and preparing Test Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # We take the tail end of the dataset for testing
    test_data_raw = df.tail(TEST_DAYS + WINDOW_SIZE)
    actual_dates = test_data_raw['Date'].tail(TEST_DAYS).values
    
    # Isolate the 'Close' prices and scale them
    close_prices = test_data_raw.filter(['Close']).values
    scaled_data = scaler.transform(close_prices)

    # Create the 60-day testing bundles
    X_test = []
    y_test_actual_scaled = []
    
    for i in range(WINDOW_SIZE, len(scaled_data)):
        X_test.append(scaled_data[i-WINDOW_SIZE:i, 0])
        y_test_actual_scaled.append(scaled_data[i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("[INFO] Asking AI to predict the test set...")
    predicted_scaled = model.predict(X_test, verbose=0)
    
    # Reverse the decimals back into real Sensex Prices
    predictions_real = scaler.inverse_transform(predicted_scaled)
    y_test_actual_real = df['Close'].tail(TEST_DAYS).values

    # ---------------------------------------------------------
    # CALCULATE STANDARD ERROR METRICS
    # ---------------------------------------------------------
    mae = mean_absolute_error(y_test_actual_real, predictions_real)
    rmse = np.sqrt(mean_squared_error(y_test_actual_real, predictions_real))
    
    # ---------------------------------------------------------
    # CALCULATE ADVANCED ACCURACY METRICS (The Fix is Here!)
    # ---------------------------------------------------------
    r2 = r2_score(y_test_actual_real, predictions_real)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_actual_real - predictions_real.flatten()) / y_test_actual_real)) * 100
    
    # Directional Accuracy
    correct_trends = 0
    total_trends = len(y_test_actual_real) - 1

    for i in range(1, len(y_test_actual_real)):
        actual_trend = y_test_actual_real[i] - y_test_actual_real[i-1]
        predicted_trend = predictions_real[i][0] - y_test_actual_real[i-1]
        
        if (actual_trend > 0 and predicted_trend > 0) or (actual_trend < 0 and predicted_trend < 0):
            correct_trends += 1

    directional_accuracy = (correct_trends / total_trends) * 100

    # --- PRINTING THE PROFESSIONAL REPORT ---
    print("-" * 50)
    print(" FINAL EXAM RESULTS (ACCURACY & ERROR)")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE)  : Rs. {mae:.2f}")
    print(f"Root Mean Sq. Error (RMSE) : Rs. {rmse:.2f}")
    print(f"R-Squared Score            : {r2:.4f} (Closer to 1.0 is better)")
    print(f"Model Error (MAPE)         : {mape:.2f}%")
    print(f"Functional Accuracy        : {100 - mape:.2f}%")
    print(f"Directional Accuracy       : {directional_accuracy:.2f}% (Guessed UP/DOWN correctly)")
    print("-" * 50)
    print("-> Tell Team Data to copy these numbers into the final report!")

    # ---------------------------------------------------------
    # VISUALIZATION: Actual vs Predicted Chart
    # ---------------------------------------------------------
    print("[INFO] Generating Comparison Chart...")
    plt.figure(figsize=(14, 6))
    plt.plot(actual_dates, y_test_actual_real, color='black', label='Actual Sensex Price', linewidth=2)
    plt.plot(actual_dates, predictions_real, color='blue', label='AI Predicted Price', linestyle='dashed', linewidth=2)
    
    plt.title('AI Performance: Actual vs. Predicted Sensex (Last 250 Days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sensex Price (INR)', fontsize=12)
    
    # Show fewer dates on the X-axis so it isn't messy
    plt.xticks(actual_dates[::30], rotation=45) 
    
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    chart_path = f"{OUTPUT_FOLDER}/actual_vs_predicted.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Chart saved as '{chart_path}'. Put this in your Presentation!")

if __name__ == "__main__":
    evaluate_ai()
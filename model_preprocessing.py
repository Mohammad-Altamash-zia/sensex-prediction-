import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- CONFIGURATION ---
INPUT_FILE = "sensex_data_perfect.csv"  # The clean file from Team Data
WINDOW_SIZE = 60                        # How many days the LSTM looks back
OUTPUT_FOLDER = "model_data"            # Where we save the AI-ready data

def prepare_lstm_data():
    print("[INFO] Loading clean dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # We only want to predict the 'Close' price, so we isolate it
    # We convert it to a NumPy array (which is what TensorFlow/Keras needs)
    close_data = df.filter(['Close']).values
    
    print("[INFO] Scaling data between 0 and 1...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    
    # SAVE THE SCALER: This is a Pro-Tip! 
    # We must save this scaler file so the Web App (Streamlit) can use it later
    # to convert the AI's 0-1 decimal prediction back into a real Rs/Sensex number.
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    joblib.dump(scaler, f"{OUTPUT_FOLDER}/price_scaler.pkl")
    print("[SUCCESS] Scaler saved as 'price_scaler.pkl'")

    print(f"[INFO] Building the {WINDOW_SIZE}-Day Window Sequences...")
    X_train = []
    y_train = []
    
    # Loop through the data to create the bundles
    for i in range(WINDOW_SIZE, len(scaled_data)):
        # X gets the previous 60 days
        X_train.append(scaled_data[i-WINDOW_SIZE:i, 0])
        # Y gets the 61st day (the target)
        y_train.append(scaled_data[i, 0])
        
    # Convert lists to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # CRITICAL STEP: LSTMs expect data in a 3D shape [Samples, Time Steps, Features]
    # Right now it's 2D. We reshape it to 3D.
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    print(f"[SUCCESS] Data shaped perfectly for LSTM!")
    print(f"   -> X shape: {X_train.shape} (Samples, Days, Features)")
    print(f"   -> Y shape: {y_train.shape} (Samples)")
    
    # Save the arrays so we can easily load them in the next script (Training)
    np.save(f"{OUTPUT_FOLDER}/X_train.npy", X_train)
    np.save(f"{OUTPUT_FOLDER}/y_train.npy", y_train)
    print(f"[SUCCESS] AI-ready data saved in the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    try:
        prepare_lstm_data()
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")
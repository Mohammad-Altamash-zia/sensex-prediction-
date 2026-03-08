from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# --- BULLETPROOF PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data is in the same folder as app.py
DATA_FILE = os.path.join(BASE_DIR, "sensex_data_perfect.csv")

# Model and Scaler are inside the 'model_data' folder
MODEL_FILE = os.path.join(BASE_DIR, "model_data", "sensex_model.keras") 
SCALER_FILE = os.path.join(BASE_DIR, "model_data", "price_scaler.pkl")
WINDOW_SIZE = 60

# Load AI safely
try:
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("[INFO] AI Brain Loaded Successfully.")
except Exception as e:
    print(f"[WARNING] AI Model not loaded. Error: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    """Fetches historical data based on the slider value."""
    days = int(request.args.get('days', 100))
    try:
        df = pd.read_csv(DATA_FILE)
        chart_data = df.tail(days)
        # Replace NaNs with None so JSON doesn't crash
        chart_data = chart_data.replace({np.nan: None})
        return jsonify(chart_data.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['GET'])
def predict():
    """Runs the AI model for tomorrow's prediction."""
    if model is None or scaler is None:
        return jsonify({"error": "AI Model is down. Check backend terminal logs."}), 500

    try:
        df = pd.read_csv(DATA_FILE)
        last_60_days = df['Close'].tail(WINDOW_SIZE).values
        last_price = float(last_60_days[-1])
        last_date = str(df['Date'].iloc[-1])
        
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_input = np.array([last_60_days_scaled])
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        
        predicted_decimal = model.predict(X_input, verbose=0)
        predicted_price = float(scaler.inverse_transform(predicted_decimal)[0][0])
        price_diff = predicted_price - last_price
        
        return jsonify({
            "last_date": last_date,
            "last_price": round(last_price, 2),
            "predicted_price": round(predicted_price, 2),
            "difference": round(price_diff, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download')
def download_csv():
    """Allows the user to download the dataset."""
    return send_file(DATA_FILE, as_attachment=True, download_name="sensex_data_perfect.csv")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
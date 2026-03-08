import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
DATA_FOLDER = "model_data"
MODEL_SAVE_PATH = "sensex_model.h5"

# AI Hyperparameters (The "Tuning" Settings)
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_lstm_model():
    print("[INFO] Loading preprocessed data from Week 2...")
    try:
        X_train = np.load(f"{DATA_FOLDER}/X_train.npy")
        y_train = np.load(f"{DATA_FOLDER}/y_train.npy")
    except FileNotFoundError:
        print("[ERROR] Could not find X_train.npy or y_train.npy.")
        print("        Did Team Model run 'model_preprocessing.py' first?")
        return

    print(f"[INFO] Data loaded successfully. X_train shape: {X_train.shape}")
    
    # ---------------------------------------------------------
    # TASK 1: THE ARCHITECTURE
    # ---------------------------------------------------------
    print("[INFO] Building the LSTM Neural Network Architecture...")
    model = Sequential()

    # Layer 1: The First LSTM Brain (Looks at the 60-day window)
    # return_sequences=True means it passes its thoughts to the next LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Prevents memorizing (Overfitting)

    # Layer 2: The Second LSTM Brain
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Layer 3: The Dense Layers (Condensing the thoughts into one number)
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # The final predicted decimal price

    # ---------------------------------------------------------
    # TASK 3: TUNING (Learning Rate & Compiler)
    # ---------------------------------------------------------
    print(f"[INFO] Compiling Model with Learning Rate: {LEARNING_RATE}")
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Professional Feature: Early Stopping
    # If the model stops getting smarter after 5 epochs, stop early to save time!
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # ---------------------------------------------------------
    # TASK 2: TRAINING
    # ---------------------------------------------------------
    print(f"[INFO] Starting Training for {EPOCHS} Epochs...")
    print("       (This might take a few minutes. Grab a coffee )")
    
    history = model.fit(
        X_train, y_train, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        callbacks=[early_stop],
        verbose=1 # Shows the progress bar in the terminal
    )

    # ---------------------------------------------------------
    # FINALIZATION: Save and Plot
    # ---------------------------------------------------------
    print("\n[SUCCESS] Training Complete!")
    
    # Save the model
    model.save(f"{DATA_FOLDER}/{MODEL_SAVE_PATH}")
    print(f"[INFO] AI Brain saved as '{MODEL_SAVE_PATH}' in '{DATA_FOLDER}' folder.")

    # Plot the "Mistake Score" (Loss) to prove it learned
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='blue')
    plt.title('AI Learning Progress (Loss Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch (Number of times it read the data)')
    plt.ylabel('Mistake Score (Lower is better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = f"{DATA_FOLDER}/training_loss_chart.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Learning curve chart saved as '{plot_path}'")

if __name__ == "__main__":
    train_lstm_model()
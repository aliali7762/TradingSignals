import os
import subprocess
import time
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from joblib import dump, load
import ta

# Automatically install required libraries
required_libraries = ["ccxt", "pandas", "numpy", "flask", "lightgbm", "scikit-learn", "joblib", "ta"]
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call(["pip", "install", lib])

# Initialize Flask
app = Flask(__name__)

# Analysis settings
ASSETS = ["EUR/USD", "GBP/USD", "USD/JPY"]  # Monitored assets
TIMEFRAME = "1m"  # Timeframe

# Generate sample data (replace with real data fetching)
def generate_sample_data():
    np.random.seed(42)
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=500, freq="T"),
        "open": np.random.rand(500),
        "high": np.random.rand(500),
        "low": np.random.rand(500),
        "close": np.random.rand(500),
        "volume": np.random.randint(100, 1000, size=500),
    }
    return pd.DataFrame(data)

# Calculate technical indicators
def calculate_indicators(df):
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["close"])
    df["stochastic"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["bollinger_high"] = ta.volatility.bollinger_hband(df["close"])
    df["bollinger_low"] = ta.volatility.bollinger_lband(df["close"])
    return df.dropna()

# Train the model
def train_model():
    data = generate_sample_data()
    data = calculate_indicators(data)
    features = data.drop(columns=["timestamp", "close"])
    target = (data["close"].shift(-1) > data["close"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    calibrated_model = CalibratedClassifierCV(model, method="isotonic")
    calibrated_model.fit(X_train, y_train)

    dump(calibrated_model, "model.joblib")
    return calibrated_model, X_test, y_test

# Serve live recommendations
@app.route("/signals", methods=["GET"])
def get_signals():
    if not os.path.exists("model.joblib"):
        train_model()

    model = load("model.joblib")
    data = generate_sample_data()
    data = calculate_indicators(data)
    features = data.drop(columns=["timestamp", "close"])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    signals = [
        {"asset": asset, "confidence": float(prob)}
        for asset, prob in zip(ASSETS, probabilities[-len(ASSETS):])
    ]
    return jsonify({"signals": signals})

# Run Flask
if __name__ == "__main__":
    print("Training model if needed...")
    if not os.path.exists("model.joblib"):
        model, X_test, y_test = train_model()
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Model trained with accuracy: {accuracy:.2f}")
    else:
        print("Model already exists. Skipping training.")

    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)

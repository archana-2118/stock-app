import os
import math
import warnings
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (fixes "Failed to fetch" from browser)


# ---------------------------------------------------------------------------
# Helper: fetch stock data
# ---------------------------------------------------------------------------
def fetch_stock_data(ticker: str, period: str = "2y"):
    """Download historical OHLCV data via yfinance.

    Returns (DataFrame, None) on success or (None, error_str) on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            return None, f"No price data found for ticker '{ticker}'. Check the symbol and try again."
        # Make the index timezone-naive for downstream use
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df, None
    except Exception as exc:
        return None, f"yfinance error: {exc}"


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------
def run_lstm(df: pd.DataFrame, days: int = 7):
    """Train a small LSTM and forecast `days` trading days ahead.

    Returns (result_dict, None) or (None, error_str).
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        close = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close)

        SEQ_LEN = 60
        train_end = int(len(scaled) * 0.80)

        if train_end < SEQ_LEN + 10:
            return None, "Not enough historical data to train LSTM (need at least ~80 trading days)."

        train = scaled[:train_end]
        test_raw = scaled[train_end - SEQ_LEN:]  # overlap so first test seq is valid

        # Build supervised sequences for training
        X_tr, y_tr = [], []
        for i in range(SEQ_LEN, len(train)):
            X_tr.append(train[i - SEQ_LEN:i, 0])
            y_tr.append(train[i, 0])
        X_tr = np.array(X_tr).reshape(-1, SEQ_LEN, 1)
        y_tr = np.array(y_tr)

        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_tr, y_tr, batch_size=32, epochs=10, verbose=0)

        # Evaluate on test set
        X_te, y_te = [], []
        for i in range(SEQ_LEN, len(test_raw)):
            X_te.append(test_raw[i - SEQ_LEN:i, 0])
            y_te.append(test_raw[i, 0])
        X_te = np.array(X_te).reshape(-1, SEQ_LEN, 1)
        y_te = np.array(y_te)

        pred_scaled = model.predict(X_te, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled).flatten()
        true_actual = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        rmse = round(math.sqrt(mean_squared_error(true_actual, pred_actual)), 4)

        # Forecast future `days`
        last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        future_preds = []
        cur = last_seq.copy()
        for _ in range(days):
            nxt = model.predict(cur, verbose=0)
            future_preds.append(float(nxt[0, 0]))
            cur = np.append(cur[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)

        future_actual = scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten().tolist()

        return {
            "predictions": future_actual,
            "rmse": rmse,
            "test_actual": true_actual[-30:].tolist(),
            "test_predicted": pred_actual[-30:].tolist(),
        }, None

    except Exception as exc:
        traceback.print_exc()
        return None, f"LSTM error: {exc}"


# ---------------------------------------------------------------------------
# Prophet model
# ---------------------------------------------------------------------------
def run_prophet(df: pd.DataFrame, days: int = 7):
    """Fit a Prophet model and forecast `days` trading days ahead.

    Returns (result_dict, None) or (None, error_str).
    """
    try:
        from prophet import Prophet

        prop_df = df[["Close"]].reset_index()
        prop_df.columns = ["ds", "y"]
        prop_df["ds"] = pd.to_datetime(prop_df["ds"])
        # Ensure tz-naive
        if prop_df["ds"].dt.tz is not None:
            prop_df["ds"] = prop_df["ds"].dt.tz_localize(None)

        train_end = int(len(prop_df) * 0.80)
        if train_end < 20:
            return None, "Not enough historical data for Prophet."

        train_df = prop_df.iloc[:train_end]
        test_df = prop_df.iloc[train_end:]

        # Fit on training split → RMSE
        m_eval = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
        )
        m_eval.fit(train_df, iter=300)
        test_pred = m_eval.predict(test_df[["ds"]])
        rmse = round(
            math.sqrt(mean_squared_error(test_df["y"].values, test_pred["yhat"].values)),
            4,
        )

        # Fit on full data → future forecast
        m_full = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
        )
        m_full.fit(prop_df, iter=300)
        future = m_full.make_future_dataframe(periods=days)
        forecast = m_full.predict(future)
        future_only = forecast.tail(days)

        return {
            "predictions": future_only["yhat"].tolist(),
            "rmse": rmse,
            "test_actual": test_df["y"].tolist()[-30:],
            "test_predicted": test_pred["yhat"].tolist()[-30:],
        }, None

    except Exception as exc:
        traceback.print_exc()
        return None, f"Prophet error: {exc}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict with a single model (LSTM or Prophet).

    Request JSON: { "ticker": "AAPL", "model": "lstm"|"prophet", "days": 7|30 }
    """
    try:
        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"error": "Request body must be JSON."}), 400

        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        ticker = str(body.get("ticker", "")).upper().strip()
        model_type = str(body.get("model", "lstm")).lower()
        try:
            days = int(body.get("days", 7))
        except (TypeError, ValueError):
            return jsonify({"error": "'days' must be an integer (7 or 30)."}), 400

        if not ticker:
            return jsonify({"error": "ticker is required."}), 400
        if model_type not in ("lstm", "prophet"):
            return jsonify({"error": "model must be 'lstm' or 'prophet'."}), 400
        if days not in (7, 30):
            days = 7

        print(f"[INFO] /predict  ticker={ticker}  model={model_type}  days={days}")

        df, err = fetch_stock_data(ticker)
        if err:
            return jsonify({"error": err}), 400

        # Historical window for display
        hist_prices = df["Close"].tail(90).round(4).tolist()
        hist_dates = [str(d.date()) for d in df.index[-90:]]

        if model_type == "lstm":
            result, err = run_lstm(df, days)
            if err:
                return jsonify({"error": err}), 500
            return jsonify({
                "lstm_predictions": result["predictions"],
                "lstm_rmse": result["rmse"],
                "historical_prices": hist_prices,
                "historical_dates": hist_dates,
                "test_actual": result["test_actual"],
                "test_predicted": result["test_predicted"],
            })

        else:  # prophet
            result, err = run_prophet(df, days)
            if err:
                return jsonify({"error": err}), 500
            return jsonify({
                "prophet_predictions": result["predictions"],
                "prophet_rmse": result["rmse"],
                "historical_prices": hist_prices,
                "historical_dates": hist_dates,
                "test_actual": result["test_actual"],
                "test_predicted": result["test_predicted"],
            })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Run both LSTM and Prophet and return side-by-side results.

    Request JSON: { "ticker": "AAPL", "days": 7|30 }
    """
    try:
        body = request.get_json(force=True, silent=True)
        if not body or not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        ticker = str(body.get("ticker", "")).upper().strip()
        try:
            days = int(body.get("days", 7))
        except (TypeError, ValueError):
            return jsonify({"error": "'days' must be an integer (7 or 30)."}), 400

        if not ticker:
            return jsonify({"error": "ticker is required."}), 400
        if days not in (7, 30):
            days = 7

        print(f"[INFO] /compare  ticker={ticker}  days={days}")

        df, err = fetch_stock_data(ticker)
        if err:
            return jsonify({"error": err}), 400

        hist_prices = df["Close"].tail(90).round(4).tolist()
        hist_dates = [str(d.date()) for d in df.index[-90:]]

        lstm_res, lstm_err = run_lstm(df, days)
        prophet_res, prophet_err = run_prophet(df, days)

        resp = {
            "historical_prices": hist_prices,
            "historical_dates": hist_dates,
        }

        if lstm_err:
            print(f"[WARN] LSTM failed: {lstm_err}")
            resp["lstm_error"] = lstm_err
        else:
            resp["lstm_predictions"] = lstm_res["predictions"]
            resp["lstm_rmse"] = lstm_res["rmse"]

        if prophet_err:
            print(f"[WARN] Prophet failed: {prophet_err}")
            resp["prophet_error"] = prophet_err
        else:
            resp["prophet_predictions"] = prophet_res["predictions"]
            resp["prophet_rmse"] = prophet_res["rmse"]

        return jsonify(resp)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[INFO] Starting Flask on 0.0.0.0:{port}")
    # debug=False in production; set to True locally for auto-reload
    app.run(host="0.0.0.0", port=port, debug=False)

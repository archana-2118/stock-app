# StockSense Python (Flask)

StockSense Python is a small web app that forecasts stock **closing prices** using two models:
- **LSTM** (TensorFlow/Keras)
- **Prophet** (Facebook Prophet)

It fetches historical price data from **yfinance**, trains the selected model, and returns future predictions (plus a basic test-set RMSE and charts).

> Educational use only — forecasts are not financial advice.

---

## Features

- Single-model prediction endpoint:
  - **`lstm`** or **`prophet`**
- Side-by-side comparison of both models
- Frontend charts (Chart.js):
  - Historical close prices (last 90 trading days)
  - Forecast for the next 7 or 30 “days”
  - Validation chart (test actual vs predicted)
- Validation metrics:
  - **RMSE** on a held-out test split

---

## Tech Stack

- **Backend:** Python + **Flask**
- **Frontend:** HTML templates + vanilla JS + **Chart.js**
- **Data source:** **yfinance**
- **ML models:**
  - **TensorFlow (tensorflow-cpu)** for LSTM
  - **prophet** for Prophet forecasting
- **Utilities:** pandas, numpy, scikit-learn (MinMaxScaler, RMSE)

---

## Project Structure

- `app.py` — Flask server + model logic + API routes
- `templates/index.html` — single-page UI
- `static/style.css` — styling
- `static/script.js` — calls API endpoints and renders charts
- `requirements.txt` — Python dependencies

---

## How It Works (High Level)

### Data fetching
`app.py` downloads historical OHLCV data via:
- `yf.Ticker(ticker).history(period="2y")`

It uses the **`Close`** column as the target.

### LSTM
- Scales `Close` to `[0, 1]` using `MinMaxScaler`
- Uses a rolling window sequence length (**SEQ_LEN = 60**)
- Trains an LSTM network and forecasts the next `days` points iteratively
- Computes RMSE on the internal test split

### Prophet
- Converts data to Prophet format: `ds` (date) and `y` (close)
- Fits on the training split to compute RMSE
- Fits on full data and forecasts the next `days` points

---

## API

All endpoints expect/return **JSON**.

### `GET /health`
Health check.

Response:
```json
{ "status": "ok" }
```

### `POST /predict`
Run **one** model.

Request JSON:
```json
{ "ticker": "AAPL", "model": "lstm", "days": 7 }
```
- `ticker`: required, stock symbol (e.g., `AAPL`)
- `model`: `"lstm"` or `"prophet"` (default: `lstm`)
- `days`: `7` or `30` (default: `7`)

Success response (key names depend on model):
- For LSTM:
  - `lstm_predictions`
  - `lstm_rmse`
- For Prophet:
  - `prophet_predictions`
  - `prophet_rmse`

Common fields:
- `historical_prices`
- `historical_dates`
- `test_actual`
- `test_predicted`

### `POST /compare`
Run **both** models.

Request JSON:
```json
{ "ticker": "AAPL", "days": 7 }
```
Response includes:
- `lstm_predictions`, `lstm_rmse`
- `prophet_predictions`, `prophet_rmse`
- `historical_prices`, `historical_dates`
- plus `lstm_error` / `prophet_error` if either model fails

---

## Run Locally

### 1) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> Notes:
> - `tensorflow-cpu` and `prophet` can be heavy; install may take time.

### 3) Start the Flask server
```bash
python app.py
```

The server listens on `0.0.0.0:<PORT>` (default `PORT=8000`).

Open:
- `http://localhost:8000/`

---

## Example Requests (cURL)

### Predict with LSTM
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"ticker\":\"AAPL\",\"model\":\"lstm\",\"days\":7}"
```

### Predict with Prophet
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"ticker\":\"AAPL\",\"model\":\"prophet\",\"days\":30}"
```

### Compare both models
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d "{\"ticker\":\"AAPL\",\"days\":7}"
```

---

## Troubleshooting

### “Failed to fetch” in the browser
Common causes:
- Flask server not running (verify by opening `/health`)
- Wrong endpoint URL
- Network/proxy blocking requests

### Model training errors / insufficient data
- If a ticker returns very little history, the LSTM/Prophet functions may return errors.
- LSTM requires enough rows to build sequences (internal guard in `run_lstm`).

### Performance / long runtime
- LSTM training uses ~10 epochs on CPU and can take noticeable time depending on your machine.
- Prophet may take a while on first run depending on installed dependencies.

---

## License

No license specified in this repository.


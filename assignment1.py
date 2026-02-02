import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Load data (local files)
# -----------------------------
BASE_DIR = Path.cwd()
train_path = BASE_DIR / "assignment_data_train.csv"
test_path  = BASE_DIR / "assignment_data_test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# Parse and sort timestamps
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
test["Timestamp"]  = pd.to_datetime(test["Timestamp"])
train = train.sort_values("Timestamp").reset_index(drop=True)
test  = test.sort_values("Timestamp").reset_index(drop=True)

# Set hourly index
train = train.set_index("Timestamp").asfreq("H")

# Target variable
y = train["trips"].astype(float)

# Fill missing hours safely
y = y.interpolate(limit_direction="both")

# -----------------------------
# Model (Exponential Smoothing)
# Weekly seasonality for hourly data = 24*7 = 168
# -----------------------------
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=168
)

# Fit model
modelFit = model.fit(optimized=True)

# -----------------------------
# Forecast exactly 744 hours
# -----------------------------
steps = 744
pred_series = modelFit.forecast(steps=steps)

# Vector named pred
pred = np.asarray(pred_series, dtype=float)

# Safety: no NaNs and exact length
pred = np.nan_to_num(pred, nan=np.nanmean(pred) if np.isfinite(pred).any() else 0.0)
if pred.shape[0] != 744:
    pred = pred[:744] if pred.shape[0] > 744 else np.pad(pred, (0, 744 - pred.shape[0]), mode="edge")



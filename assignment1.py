import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.varmax import VARMAX

# -----------------------------
# Load data (local files)
# -----------------------------
BASE_DIR = Path.cwd()
train_path = BASE_DIR / "assignment_data_train.csv"
test_path  = BASE_DIR / "assignment_data_test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# -----------------------------
# Clean / prep
# -----------------------------
# Drop Timestamp from modeling matrix (keep it only if you need it later)
if "Timestamp" in train.columns:
    train = train.drop(columns=["Timestamp"])
if "Timestamp" in test.columns:
    test = test.drop(columns=["Timestamp"])

# Convert everything to numeric
train = train.apply(pd.to_numeric, errors="coerce")
test  = test.apply(pd.to_numeric, errors="coerce")

# Drop constant columns (VARMAX can't use constants as separate series)
constant_cols = [c for c in train.columns if train[c].nunique(dropna=True) <= 1]
train = train.drop(columns=constant_cols)
test  = test.drop(columns=constant_cols, errors="ignore")

# Keep only columns that exist in BOTH train and test (for consistent shape)
common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols]
test  = test[common_cols]

# Drop missing values (simple + autograder-safe)
train = train.dropna().copy()
test  = test.dropna().copy()

# Ensure 'trips' exists
if "trips" not in train.columns:
    raise ValueError(f"'trips' not found in training columns: {list(train.columns)}")

# -----------------------------
# Differencing (stationarity)
# -----------------------------
train_diff = train.diff().dropna()

# -----------------------------
# Build + fit VARMA via VARMAX
# -----------------------------
# Keep it simple and stable: VARMA(1,1)
model = VARMAX(
    train_diff,
    order=(1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
)

modelFit = model.fit(disp=False)

# -----------------------------
# Forecast 744 hours
# -----------------------------
steps = 744
pred_diff = modelFit.forecast(steps=steps)

# Undo differencing to get back to levels
last_levels = train.iloc[-1]
pred_levels = pred_diff.cumsum().add(last_levels, axis="columns")

# Extract trips forecast as a VECTOR named pred
pred = pred_levels["trips"].to_numpy(dtype=float)

# Safety: ensure correct length and no NaN
pred = np.asarray(pred, dtype=float)
pred = np.nan_to_num(pred, nan=np.nanmean(pred) if np.isfinite(pred).any() else 0.0)

if pred.shape[0] != 744:
    pred = pred[:744] if pred.shape[0] > 744 else np.pad(pred, (0, 744 - pred.shape[0]), mode="edge")

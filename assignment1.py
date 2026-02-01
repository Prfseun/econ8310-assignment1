import pandas as pd
import numpy as np
from pathlib import Path
from pygam import LinearGAM, s

# -----------------------------
# Paths (your files are in the project root)
# -----------------------------
BASE_DIR = Path.cwd()
train_path = BASE_DIR / "assignment_data_train.csv"
test_path  = BASE_DIR / "assignment_data_test.csv"
out_path   = BASE_DIR / "predictions_pygam.csv"   # output file

# -----------------------------
# Load data from local files (NOT GitHub)
# -----------------------------
train_data = pd.read_csv(train_path)
test_data  = pd.read_csv(test_path)

# -----------------------------
# Date handling
# -----------------------------
train_data["Timestamp"] = pd.to_datetime(train_data["Timestamp"])
test_data["Timestamp"]  = pd.to_datetime(test_data["Timestamp"])

# Feature engineering
for df in (train_data, test_data):
    df["day_of_week"] = df["Timestamp"].dt.weekday + 1  # Mon=1 ... Sun=7
    df["hour"] = df["Timestamp"].dt.hour
    df["month"] = df["Timestamp"].dt.month

# -----------------------------
# Train model
# -----------------------------
X_train = train_data[["month", "day_of_week", "hour"]].values
y_train = train_data["trips"].values

# GAM with smooth terms for each feature
model = LinearGAM(s(0) + s(1) + s(2)).gridsearch(X_train, y_train)

# -----------------------------
# Predict on test set
# -----------------------------
X_test = test_data[["month", "day_of_week", "hour"]].values
test_data["trips"] = model.predict(X_test)

# Predictions array (like your friend had)
pred = test_data["trips"].values


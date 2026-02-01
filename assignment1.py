import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------
# 1) Load training + test data
# ----------------------------
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)

# ----------------------------
# 2) Clean / set time index
# ----------------------------
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
train = train.sort_values("Timestamp").set_index("Timestamp")

# enforce hourly frequency
train = train.asfreq("h")

# ensure trips is numeric
train["trips"] = pd.to_numeric(train["trips"], errors="coerce")
train = train.dropna(subset=["trips"])

y = train["trips"]

# ----------------------------
# 3) Model: Exponential Smoothing
# ----------------------------
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=168,
    initialization_method="estimated",
)

modelFit = model.fit(optimized=True)

# ----------------------------
# 4) Forecast: 744 hours
# ----------------------------
pred = modelFit.forecast(744)

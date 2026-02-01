import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------
# 1) Load data
# ----------------------------
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)

# ----------------------------
# 2) Prepare time series
# ----------------------------
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
train = train.sort_values("Timestamp").set_index("Timestamp")

# enforce hourly frequency
train = train.asfreq("h")

train["trips"] = pd.to_numeric(train["trips"], errors="coerce")
train = train.dropna(subset=["trips"])

y = train["trips"]

# ----------------------------
# 3) Holt-Winters Model
# ----------------------------
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="mul",          # 🔥 THIS IS THE KEY CHANGE
    seasonal_periods=168,    # weekly seasonality
    initialization_method="estimated"
)

modelFit = model.fit(optimized=True)

# ----------------------------
# 4) Forecast January (744 hours)
# ----------------------------
pred = modelFit.forecast(744)

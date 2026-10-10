import io
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

KAGGLE_DATASET = "heeraldedhia/groceries-dataset"

def load_kaggle_groceries():
    """Download a public Kaggle dataset through KaggleHub. Network access is required."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is missing. Install requirements.txt and retry.") from exc
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not candidates:
        raise RuntimeError("Kaggle download completed but no CSV was found.")
    # Prefer the common Groceries_dataset.csv file.
    preferred = [p for p in candidates if "groceries" in os.path.basename(p).lower()]
    file_path = preferred[0] if preferred else candidates[0]
    return pd.read_csv(file_path), f"Kaggle: {KAGGLE_DATASET} ({os.path.basename(file_path)})"

def load_uploaded_csv(uploaded_file):
    # A practical cap protects free-tier hosting from multi-GB Kaggle competition files.
    # For the biggest competitions, aggregate in Kaggle/locally first and upload the reduced CSV.
    df = pd.read_csv(uploaded_file, nrows=1_000_000, low_memory=False)
    return df, f"Uploaded Kaggle CSV: {uploaded_file.name} (up to 1,000,000 rows)"

def _find_col(df, options):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for option in options:
        if option in lower:
            return lower[option]
    for c in df.columns:
        if any(option in str(c).strip().lower() for option in options):
            return c
    return None

def build_daily_demand(raw):
    """Convert transaction rows into product-day transaction counts.
    For basket-level data, each row is treated as one product occurrence, not a verified unit quantity.
    """
    if raw is None or raw.empty:
        raise ValueError("The dataset is empty.")
    date_col = _find_col(raw, ["date", "order date", "transaction date", "invoice date", "datetime"])
    product_col = _find_col(raw, ["itemdescription", "item description", "product", "product name", "item", "description", "stockcode", "family", "item_nbr", "item number"])
    qty_col = _find_col(raw, ["quantity", "qty", "units sold", "sales quantity", "sales", "unit_sales", "units"])
    if date_col is None or product_col is None:
        raise ValueError(f"Could not identify date and product columns. Found: {list(raw.columns)[:30]}")
    df = raw[[date_col, product_col] + ([qty_col] if qty_col else [])].copy()
    df.columns = ["date", "product"] + (["quantity"] if qty_col else [])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["product"] = df["product"].astype(str).str.strip()
    df = df.dropna(subset=["date"])
    df = df[df["product"].ne("") & df["product"].ne("nan")]
    if df.empty:
        raise ValueError("No usable date/product rows remained after cleaning.")
    if qty_col:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
        # Sales data sometimes includes returns/negative quantities; remove nonpositive values for demand forecasting.
        df = df[df["quantity"] > 0]
        grouped = df.groupby([df["date"].dt.floor("D"), "product"], as_index=False)["quantity"].sum()
        grouped.columns = ["date", "product", "demand"]
    else:
        grouped = df.groupby([df["date"].dt.floor("D"), "product"], as_index=False).size()
        grouped.columns = ["date", "product", "demand"]
    # Limit extreme accidental data explosions while preserving reasonably large public datasets.
    grouped = grouped.sort_values("date")
    mapping = pd.DataFrame({"product": sorted(grouped["product"].unique())})
    return grouped, mapping

def _feature_frame(daily):
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product", "date"])
    # Reindex each product to a continuous daily calendar so zero-sales days are represented.
    frames = []
    for product, group in df.groupby("product", sort=False):
        group = group.set_index("date")[["demand"]].sort_index()
        idx = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(idx, fill_value=0).rename_axis("date").reset_index()
        group["product"] = product
        frames.append(group)
    full = pd.concat(frames, ignore_index=True)
    full["dow"] = full["date"].dt.dayofweek
    full["month"] = full["date"].dt.month
    full["day_of_month"] = full["date"].dt.day
    full["product_code"], _ = pd.factorize(full["product"], sort=True)
    full["lag_1"] = full.groupby("product")["demand"].shift(1)
    full["lag_7"] = full.groupby("product")["demand"].shift(7)
    full["roll7"] = full.groupby("product")["demand"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    full["target"] = full.groupby("product")["demand"].shift(-1)
    features = ["product_code", "dow", "month", "day_of_month", "lag_1", "lag_7", "roll7"]
    train = full.dropna(subset=["lag_1", "target"]).copy()
    train[features] = train[features].fillna(0)
    return full, train, features

def train_forecaster(daily):
    full, train, features = _feature_frame(daily)
    if len(train) < 20:
        raise ValueError("Not enough product-day history to train a useful demand model (need at least 20 rows).")
    # Keep resource use practical for free hosting and large CSVs.
    if len(train) > 250_000:
        train = train.sample(250_000, random_state=42)
    X = train[features]
    y = train["target"].clip(lower=0)
    if len(train) >= 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=20, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    mae = round(float(mean_absolute_error(y_test, pred)), 3) if len(y_test) else "n/a"
    model_info = {"model":"HistGradientBoostingRegressor (scikit-learn)", "mae":mae, "training_rows":int(len(X_train))}
    return {"model":model, "features":features, "full":full}, model_info

def make_forecast(model_bundle, daily, inventory):
    model = model_bundle["model"]
    full = model_bundle["full"]
    features = model_bundle["features"]
    last_rows = full.sort_values("date").groupby("product").tail(1).copy()
    # Need each product's latest day, not global final day.
    rows = []
    for product, group in full.groupby("product", sort=False):
        g = group.sort_values("date")
        last = g.iloc[-1].copy()
        hist = g["demand"].tolist()
        rows.append({
            "product": product,
            "product_code": int(last["product_code"]),
            "dow": int((pd.Timestamp(last["date"]) + pd.Timedelta(days=1)).dayofweek),
            "month": int((pd.Timestamp(last["date"]) + pd.Timedelta(days=1)).month),
            "day_of_month": int((pd.Timestamp(last["date"]) + pd.Timedelta(days=1)).day),
            "lag_1": float(hist[-1]) if hist else 0,
            "lag_7": float(hist[-7]) if len(hist) >= 7 else float(np.mean(hist) if hist else 0),
            "roll7": float(np.mean(hist[-7:])) if hist else 0,
        })
    features_df = pd.DataFrame(rows)
    forecast = np.clip(model.predict(features_df[features]), 0, None)
    return pd.DataFrame({"product":features_df["product"], "forecast_daily_units":np.round(forecast, 2)})

def build_demo_batches(products, forecast_df=None):
    """Reserved utility for future integration with real inventory and batch records."""
    return products

import pandas as pd
from sklearn.ensemble import IsolationForest
import config


def extract_tabular_features(df):
    """Extract temporal, aggregation, and unsupervised anomaly features."""
    print("[INFO] Engineering temporal and aggregational transaction features...")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    if "payment_format" in df.columns:
        df = pd.get_dummies(df, columns=["payment_format"], drop_first=True)

    agg_stats = (
        df.groupby("from_account")["amount"]
        .agg(["mean", "std", "max", "count"])
        .reset_index()
    )
    agg_stats.columns = [
        "from_account",
        "acc_avg_amount",
        "acc_std_amount",
        "acc_max_amount",
        "acc_tx_count",
    ]
    df = df.merge(agg_stats, on="from_account", how="left")

    df["amount_to_avg_ratio"] = df["amount"] / (df["acc_avg_amount"] + 1e-5)

    print("[INFO] Fitting Isolation Forest for unsupervised anomaly score...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=config.CONTAMINATION_RATE,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    numerical_cols = ["amount", "from_in_degree", "to_in_degree", "amount_to_avg_ratio"]
    df["isolation_forest_score"] = -iso.fit_predict(df[numerical_cols].fillna(0))

    return df

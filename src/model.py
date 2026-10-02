import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
import config


def train_aml_model(df):
    """Train supervised LightGBM model with scale_pos_weight for imbalance handling."""
    drop_cols = ["timestamp", "from_account", "to_account", "is_laundering"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()


    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category")


    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(0)

    y = df["is_laundering"]

    split_idx = int(len(df) * (1 - config.TEST_SPLIT_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test_time = df.iloc[split_idx:]["timestamp"]

    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)

    print(
        f"[INFO] Training LightGBM (Features: {len(feature_cols)}, Scale Pos Weight: {pos_weight:.1f})..."
    )
    model = lgb.LGBMClassifier(
        n_estimators=config.N_ESTIMATORS,
        learning_rate=config.LEARNING_RATE,
        max_depth=config.MAX_DEPTH,
        scale_pos_weight=pos_weight,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_pred_probs)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    print(f"[RESULT] Test ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return model, X_test, y_test, y_pred_probs, df_test_time
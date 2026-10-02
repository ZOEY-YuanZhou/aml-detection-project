import config
from src import (
    load_ibm_aml_dataset,
    extract_graph_features,
    extract_tabular_features,
    train_aml_model,
    optimize_threshold_for_capacity
)


def main():
    print("=======================================================")
    print(" STARTING END-TO-END AML DETECTION PIPELINE           ")
    print("=======================================================")

    # 1. Load Data
    raw_df = load_ibm_aml_dataset()

    # 2. Extract Graph Features
    graph_df = extract_graph_features(raw_df)

    # 3. Extract Tabular & Anomaly Features
    full_df = extract_tabular_features(graph_df)

    # 4. Train LightGBM Model
    model, X_test, y_test, y_probs, test_time = train_aml_model(full_df)

    # 5. Run Operational Threshold Tuning Engine
    optimal_thresh = optimize_threshold_for_capacity(
        y_true=y_test.values,
        y_probs=y_probs,
        df_time=test_time,
        max_daily_alerts=config.MAX_DAILY_ALERTS,
    )

    print("\n=======================================================")
    print(f" PIPELINE FINISHED SUCCESSFULLY. OPTIMAL THRESHOLD: {optimal_thresh:.4f}")
    print("=======================================================")


if __name__ == "__main__":
    main()

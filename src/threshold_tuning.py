import numpy as np


def optimize_threshold_for_capacity(y_true, y_probs, df_time, max_daily_alerts):
    """Optimize decision threshold constrained by human investigation capacity."""
    print("\n=======================================================")
    print("      COMPLIANCE THRESHOLD TUNING ENGINE               ")
    print("=======================================================")

    num_days = max((df_time.max() - df_time.min()).days, 1)
    max_total_alerts = max_daily_alerts * num_days

    print(f"[CONSTRAINT] Test Period: {num_days} Days | Max Daily Capacity: {max_daily_alerts} Alerts/Day")
    print(f"[CONSTRAINT] Maximum Allowable Test Alerts: {max_total_alerts}")

    best_thresh = 0.5
    best_recall = 0.0
    best_precision = 0.0
    best_alert_count = 0

    thresholds = np.linspace(0.01, 0.99, 200)

    for t in thresholds:
        alerts = (y_probs >= t).sum()
        if alerts <= max_total_alerts:
            tp = ((y_probs >= t) & (y_true == 1)).sum()
            fp = ((y_probs >= t) & (y_true == 0)).sum()
            fn = ((y_probs < t) & (y_true == 1)).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            if recall >= best_recall:
                best_recall = recall
                best_precision = precision
                best_thresh = t
                best_alert_count = alerts

    daily_alert_rate = best_alert_count / num_days
    print(f"\n[OPTIMIZATION RESULT]")
    print(f" -> Optimal Threshold: {best_thresh:.4f}")
    print(f" -> Daily Generated Alerts: {daily_alert_rate:.1f} / day")
    print(f" -> Achieved Precision: {best_precision:.2%}")
    print(f" -> Achieved Laundering Recall: {best_recall:.2%}")

    default_alerts = (y_probs >= 0.5).sum()
    print(f"\n[BASELINE COMPARISON (Default Threshold = 0.50)]")
    print(f" -> Default Daily Alerts: {default_alerts / num_days:.1f} / day")

    return best_thresh

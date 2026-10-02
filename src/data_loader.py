import os
import kagglehub
import pandas as pd
import config


def fetch_from_kaggle(
    dataset_handle=config.KAGGLE_DATASET_HANDLE,
    target_file=config.TARGET_FILE_NAME,
    sample_size=config.SAMPLE_SIZE,
):
    """Fetch target file (HI-Small_Trans.csv) directly using Kagglehub API."""
    print(f"[INFO] Downloading dataset via Kagglehub API ({dataset_handle})...")

  
    dataset_dir = kagglehub.dataset_download(dataset_handle)

 
    target_path = None
    for root, _, files in os.walk(dataset_dir):
        if target_file in files:
            target_path = os.path.join(root, target_file)
            break

    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(
            f"[ERROR] Could not locate '{target_file}' inside Kaggle downloaded directory: {dataset_dir}"
        )

    print(f"[SUCCESS] Located dataset file: {target_path}")
    print(f"[INFO] Reading top {sample_size} records...")

    df_full = pd.read_csv(target_path)
    df = df_full.sample(n=sample_size, random_state=42).reset_index(drop=True)
    # df = pd.read_csv(target_path, nrows=sample_size)
    return df


def load_ibm_aml_dataset():
    """Main data loader entry point fetching directly from Kaggle API."""
    df = fetch_from_kaggle()

    
    rename_dict = {
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",
        "To Bank": "to_bank",
        "Account.1": "to_account",
        "Amount Received": "amount",
        "Is Laundering": "is_laundering",
        "Payment Format": "payment_format",
    }
    df = df.rename(columns=rename_dict)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(
        f"[INFO] IBM AML Dataset Loaded Successfully! Total Rows: {len(df)}, Laundering Ratio: {df['is_laundering'].mean():.4%}"
    )
    return df

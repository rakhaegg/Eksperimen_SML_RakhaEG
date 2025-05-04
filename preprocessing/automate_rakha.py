"""
automate_rakha.py
================
$ python automate_rakha.py \
    --raw namadataset_raw/demand_history.csv \
    --outdir preprocessing/namadataset_preprocessing
"""
import argparse, sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ╭─────────────────────────── 1. LOAD RAW ─────────────────────────────╮
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["Date"],
        infer_datetime_format=True,
        low_memory=False,
    )
    print(f"[load_raw] shape={df.shape}")
    return df


# ╭─────────────────────────── 2. CLEAN DATA ───────────────────────────╮
def _parse_order(x: str) -> int:
    x = str(x).strip()
    if x.startswith("(") and x.endswith(")"):
        x = "-" + x[1:-1]
    return int(x)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # a) Convert Order_Demand → int
    df["Order_Demand"] = df["Order_Demand"].apply(_parse_order)

    # b) Drop NA Date & non‑positive demand
    df = df.dropna(subset=["Date"])
    df = df[df["Order_Demand"] > 0]

    # c) Winsorize P99 & remove extreme sentinel
    q99 = df["Order_Demand"].quantile(0.99)
    df = df[df["Order_Demand"] < 1_000_000]
    df["Order_Demand"] = np.where(df["Order_Demand"] > q99, q99, df["Order_Demand"])

    # d) Log‑transform
    df["Log_Demand"] = np.log1p(df["Order_Demand"])

    # e) Basic calendar features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    print(f"[clean_data] shape={df.shape}")
    return df.reset_index(drop=True)


# ╭───────────────────────── 3. FEATURE ENGINEERING ────────────────────╮
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Product_Code", "Date"]).reset_index(drop=True)
    grp = df.groupby("Product_Code")

    # Lag & rolling
    for lag in [1, 7, 28]:
        df[f"lag_{lag}"] = grp["Log_Demand"].shift(lag)
    for win in [7, 28]:
        df[f"roll_mean_{win}"] = grp["Log_Demand"].shift(1).rolling(win).mean()

    df = df.dropna().reset_index(drop=True)
    print(f"[engineer_features] shape={df.shape}")
    return df


# ╭──────────────────────────── 4. SPLIT ───────────────────────────────╮
def time_split(df: pd.DataFrame, cutoff: str = "2016-01-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["Date"] < cutoff]
    test_df = df[df["Date"] >= cutoff]
    print(f"[time_split] train={train_df.shape} test={test_df.shape}")
    return train_df, test_df


# ╭─────────────────────────── 5. SAVE ARTEFAK ─────────────────────────╮
def save_artifacts(train_df: pd.DataFrame, test_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(outdir / "train.parquet", index=False)
    test_df.to_parquet(outdir / "test.parquet", index=False)
    # Simpan full feature store juga
    full_df = pd.concat([train_df, test_df])
    full_df.to_parquet(outdir / "feature_df.parquet", index=False)
    print(f"[save_artifacts] saved to {outdir}")

# ╭──────────────────────────── MAIN ───────────────────────────────────╮
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="Path CSV mentah")
    parser.add_argument("--outdir", type=Path, default=Path("preprocessing/namadataset_preprocessing"))
    parser.add_argument("--cutoff", type=str, default="2016-01-01")
    args = parser.parse_args()

    df_raw = load_raw(args.raw)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    train_df, test_df = time_split(df_feat, args.cutoff)
    save_artifacts(train_df, test_df, args.outdir)


if __name__ == "__main__":
    sys.exit(main())

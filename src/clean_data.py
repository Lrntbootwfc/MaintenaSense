"""
Load raw CSVs from data/raw/, perform cleaning, feature engineering,
aggregation to hourly/daily features, and save processed csv to data/processed/
"""

import os
import pandas as pd
import numpy as np

def load_raw(raw_dir="data/raw"):
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
    df_list = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df_list.append(df)
    if not df_list:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")
    df_all = pd.concat(df_list, ignore_index=True)
    
    # ✅ Ensure failure column exists even if missing in some files
    if 'failure' not in df_all.columns: 
        df_all['failure'] = 0          
    
    return df_all

def basic_cleaning(df):
    # Ensure timestamp, sort
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(['machine_id', 'timestamp'], inplace=True)
    
    # ✅ Fill missing failure values with 0 to avoid NaNs later
    if df['failure'].isna().any():        
        df['failure'] = df['failure'].fillna(0).astype(int)  
    
    # Fill missing sensor values (forward then back)
    df[['temperature','vibration','pressure','humidity','runtime_hours']] = df[
        ['temperature','vibration','pressure','humidity','runtime_hours']].fillna(method='ffill').fillna(method='bfill')
    # Remove impossible values
    df = df[(df['vibration'] >= 0) & (df['temperature'] > -40) & (df['pressure'] > 0)]
    return df

def feature_engineering(df):
    df = df.copy()
    
    # ✅ Ensure no NaN in failure column before using it
    df['failure'] = df['failure'].fillna(0).astype(int) 
    
    # rolling features per machine: 3-sample rolling mean and std
    df['temp_roll_mean_3'] = df.groupby('machine_id')['temperature'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['vib_roll_mean_3'] = df.groupby('machine_id')['vibration'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['temp_roll_std_3'] = df.groupby('machine_id')['temperature'].transform(lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
    # time since last failure (simple)
    df['time_idx'] = (df['timestamp'] - df.groupby('machine_id')['timestamp'].transform('min')).dt.total_seconds()/3600.0
    # label aggregation: create a target that predicts failure in next 24 hours (1 day window)
    df = df.sort_values(['machine_id','timestamp'])
    df['failure_next_24h'] = df.groupby('machine_id')['failure'].transform(lambda x: x.rolling(window=24, min_periods=1).max().shift(-23).fillna(0).astype(int))
    return df

def downsample_to_hourly(df):
    # For scalability, aggregate to hourly per machine
    def agg_fn(sub):
        return sub.set_index('timestamp').resample('1h').agg({
            'temperature':'mean','vibration':'mean','pressure':'mean','humidity':'mean',
            'runtime_hours':'max','failure':'max','failure_next_24h':'max'
        }).reset_index()
    frames=[]
    for mid, g in df.groupby('machine_id'):
        frames.append(agg_fn(g))
    return pd.concat(frames, ignore_index=True)

def run_preprocessing(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    df = load_raw(raw_dir)
    df = basic_cleaning(df)
    df = feature_engineering(df)
    dfh = downsample_to_hourly(df)
    
    # ✅ Drop any remaining NaNs to avoid training errors
    dfh = dfh.dropna().reset_index(drop=True)
    
    processed_path = os.path.join(processed_dir, "machines_processed_hourly.csv")
    dfh.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

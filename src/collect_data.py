"""
Simulate machine sensor data and save CSVs in data/raw/.
Each machine has sensors: temperature, vibration, pressure, humidity, runtime_hours.
We also simulate binary 'failure' events based on thresholds and random noise.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def simulate_machine_series(machine_id, start_date, days=120, freq_minutes=60, seed=None):
    """
    Returns DataFrame with datetime index and sensor readings for one machine.
    """
    if seed is not None:
        np.random.seed(seed + machine_id)
    periods = int((24*60/days) * days * (60/freq_minutes)) if False else int((24*60//freq_minutes)*days)
    date_range = pd.date_range(start=start_date, periods=periods, freq=f"{freq_minutes}min")
    n = len(date_range)

    # baseline signals
    temp_base = np.random.uniform(60, 80)  # Fahrenheit or celcius depending on plant, keep generic
    vib_base = np.random.uniform(0.2, 0.8)
    pres_base = np.random.uniform(30, 80)
    hum_base = np.random.uniform(20, 60)
    runtime_cum = np.linspace(0, days*24, n)  # cumulative runtime hours

    # daily cycles + noise
    temp = temp_base + 5*np.sin(np.linspace(0, 3*np.pi, n)) + np.random.normal(0, 1.2, n)
    vib = vib_base + 0.5*np.sin(np.linspace(0, 6*np.pi, n)) + np.random.normal(0, 0.05, n)
    pres = pres_base + 2*np.cos(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 0.5, n)
    hum = hum_base + 3*np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 1.0, n)

    # simulate occasional spikes and gradual drift toward failure for some machines
    drift = np.linspace(0, np.random.uniform(0, 3), n)
    temp += drift
    vib += drift*0.05

    # Create failure label: if vibration > threshold and temp rising, mark imminent failure probabilistically
    failure_prob = np.clip((vib - 0.9) * 2 + (temp - (temp_base + 3)) * 0.05, 0, 1)
    failure_flag = (np.random.rand(n) < (failure_prob * 0.02)).astype(int)  # sparse failures

    df = pd.DataFrame({
        "timestamp": date_range,
        "machine_id": f"M_{machine_id:02d}",
        "temperature": np.round(temp, 3),
        "vibration": np.round(vib, 4),
        "pressure": np.round(pres, 3),
        "humidity": np.round(hum, 3),
        "runtime_hours": np.round(runtime_cum, 3),
        "failure": failure_flag
    })
    return df

def generate_simulated_data(output_dir="data/raw", n_machines=5, days=90, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    start = datetime.now() - timedelta(days=days)
    for m in range(1, n_machines+1):
        df = simulate_machine_series(machine_id=m, start_date=start, days=days, freq_minutes=60, seed=seed)
        fname = os.path.join(output_dir, f"machine_{m:02d}.csv")
        df.to_csv(fname, index=False)
    # also generate an aggregated CSV
    all_dfs = []
    for m in range(1, n_machines+1):
        all_dfs.append(pd.read_csv(os.path.join(output_dir, f"machine_{m:02d}.csv")))
    pd.concat(all_dfs, ignore_index=True).to_csv(os.path.join(output_dir, "machines_all.csv"), index=False)
    print(f"Simulated raw data saved to {output_dir} (machines: {n_machines}, days: {days})")

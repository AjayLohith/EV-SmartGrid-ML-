import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 5000

# Generate timestamps (hourly from 2026-01-01)
start = datetime(2026, 1, 1, 0, 0, 0)
timestamps = [start + timedelta(hours=i) for i in range(n_samples)]
hours = [t.hour for t in timestamps]

# Generate realistic values based on hour patterns
data = []
for i, (ts, hour) in enumerate(zip(timestamps, hours)):
    # Voltage varies by load (lower during peak hours)
    if 7 <= hour <= 21:
        voltage = np.random.uniform(0.87, 0.98)
    else:
        voltage = np.random.uniform(0.98, 1.03)
    
    # Current and power follow daily pattern
    base_load = 35 + 80 * np.sin(np.pi * (hour - 3) / 18) ** 2 if 3 <= hour <= 21 else 35
    current = base_load + np.random.uniform(-10, 15)
    active_power = current * 1.9 + np.random.uniform(-5, 10)
    reactive_power = active_power * np.random.uniform(0.35, 0.45)
    
    # Frequency (normally around 60 Hz)
    frequency = 60 + np.random.uniform(-0.25, 0.05) if voltage < 0.95 else 60 + np.random.uniform(-0.02, 0.02)
    
    # EV demand peaks morning and evening
    if 7 <= hour <= 9 or 17 <= hour <= 20:
        ev_demand = np.random.uniform(45, 85)
    elif 10 <= hour <= 16:
        ev_demand = np.random.uniform(25, 45)
    else:
        ev_demand = np.random.uniform(3, 20)
    
    total_load = active_power + ev_demand + np.random.uniform(50, 100)
    
    # Temperature varies by time of day
    temp_base = 15 + 8 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 15 - 2 * abs(hour - 12) / 12
    temperature = temp_base + np.random.uniform(-2, 2)
    
    # Stability depends on voltage and frequency
    if voltage < 0.92 or frequency < 59.85 or current > 100:
        stability = 'unstable'
    elif voltage < 0.95 and frequency < 59.92:
        stability = np.random.choice(['stable', 'unstable'], p=[0.3, 0.7])
    else:
        stability = np.random.choice(['stable', 'unstable'], p=[0.75, 0.25])
    
    data.append({
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'hour': hour,
        'voltage': round(voltage, 2),
        'current': round(current, 1),
        'active_power': round(active_power, 1),
        'reactive_power': round(reactive_power, 1),
        'frequency': round(frequency, 2),
        'ev_demand': round(ev_demand, 1),
        'total_load': round(total_load, 1),
        'temperature': round(temperature, 1),
        'grid_stability': stability
    })

df = pd.DataFrame(data)
df.to_csv('data/dataset.csv', index=False)
print(f'Created dataset with {len(df)} samples')
print(df.head(10))
print('...')
print(df.tail(5))
print(f"\nStability distribution: {df['grid_stability'].value_counts().to_dict()}")

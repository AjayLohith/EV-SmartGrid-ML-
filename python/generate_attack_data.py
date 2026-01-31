"""
Generate Attack/Malicious Data for Cybersecurity Demonstration
===============================================================
This script creates simulated cyber-attack data to compare with normal data.

Attack Types Simulated:
1. False Data Injection (FDI) - Manipulated sensor readings
2. Replay Attack - Repeated patterns indicating automated attacks
3. DoS Attack Patterns - Abnormal request frequencies
4. Unauthorized Access - Suspicious charging requests
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_attack_data(num_samples=100):
    """Generate dataset with cyber-attack patterns"""
    
    data = []
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    
    for i in range(num_samples):
        timestamp = start_time + timedelta(hours=i)
        hour = i % 24
        
        # Randomly assign attack type (40% attacks, 60% normal)
        is_attack = np.random.random() < 0.4
        
        if is_attack:
            attack_type = np.random.choice(['fdi', 'replay', 'dos', 'unauthorized'])
            
            if attack_type == 'fdi':
                # False Data Injection - Abnormally high/low values
                voltage = np.random.choice([0.7, 1.3]) + np.random.normal(0, 0.05)
                current = np.random.uniform(150, 250)  # Extremely high
                active_power = current * voltage * 2.5
                reactive_power = active_power * 0.6
                frequency = np.random.choice([58.5, 61.5])  # Out of normal range
                ev_demand = np.random.uniform(100, 200)  # Unrealistic demand
                
            elif attack_type == 'replay':
                # Replay Attack - Identical repeated values
                voltage = 0.95
                current = 85.5
                active_power = 162.5
                reactive_power = 68.2
                frequency = 59.85
                ev_demand = 45.0
                
            elif attack_type == 'dos':
                # DoS Attack - Rapid fluctuations
                voltage = np.random.uniform(0.8, 1.1)
                current = np.random.uniform(20, 180)
                active_power = np.random.uniform(50, 400)
                reactive_power = np.random.uniform(20, 150)
                frequency = np.random.uniform(59.5, 60.5)
                ev_demand = np.random.uniform(0, 150)
                
            else:  # unauthorized
                # Unauthorized Access - Suspicious patterns at odd hours
                voltage = 0.92
                current = 120 + np.random.normal(0, 5)
                active_power = 220 + np.random.normal(0, 10)
                reactive_power = 95 + np.random.normal(0, 5)
                frequency = 59.75
                ev_demand = 80 + np.random.normal(0, 3)
            
            total_load = active_power * 1.8
            temperature = np.random.uniform(10, 30)
            label = 'attack'
            
        else:
            # Normal behavior
            voltage = 1.0 + np.random.normal(0, 0.03)
            voltage = np.clip(voltage, 0.95, 1.05)
            
            # Normal daily pattern
            if 7 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
                current = 60 + np.random.normal(0, 10)
                ev_demand = 30 + np.random.normal(0, 8)
            elif 0 <= hour <= 5:  # Off-peak
                current = 35 + np.random.normal(0, 5)
                ev_demand = 5 + np.random.normal(0, 2)
            else:  # Regular hours
                current = 50 + np.random.normal(0, 8)
                ev_demand = 20 + np.random.normal(0, 5)
            
            active_power = voltage * current * 1.8
            reactive_power = active_power * 0.4
            frequency = 60.0 + np.random.normal(0, 0.02)
            total_load = active_power + ev_demand * 2
            temperature = 15 + np.random.normal(0, 3)
            label = 'normal'
        
        data.append({
            'timestamp': timestamp,
            'hour': hour,
            'voltage': round(voltage, 3),
            'current': round(current, 2),
            'active_power': round(active_power, 2),
            'reactive_power': round(reactive_power, 2),
            'frequency': round(frequency, 3),
            'ev_demand': round(ev_demand, 2),
            'total_load': round(total_load, 2),
            'temperature': round(temperature, 2),
            'request_label': label  # attack or normal
        })
    
    return pd.DataFrame(data)


def generate_pure_attack_data(num_samples=50):
    """Generate pure attack data for comparison"""
    
    data = []
    start_time = datetime(2026, 2, 1, 0, 0, 0)
    
    attack_types = ['fdi', 'replay', 'dos', 'unauthorized']
    
    for i in range(num_samples):
        timestamp = start_time + timedelta(hours=i)
        hour = i % 24
        attack_type = attack_types[i % 4]
        
        if attack_type == 'fdi':
            voltage = np.random.choice([0.65, 0.7, 1.25, 1.35])
            current = np.random.uniform(180, 300)
            frequency = np.random.choice([57.5, 58.0, 62.0, 62.5])
            ev_demand = np.random.uniform(150, 250)
            
        elif attack_type == 'replay':
            voltage = 0.95
            current = 85.5
            frequency = 59.85
            ev_demand = 45.0
            
        elif attack_type == 'dos':
            voltage = np.random.uniform(0.7, 1.2)
            current = np.random.uniform(10, 250)
            frequency = np.random.uniform(58.5, 61.5)
            ev_demand = np.random.uniform(0, 200)
            
        else:
            voltage = 0.88
            current = 140 + np.random.normal(0, 3)
            frequency = 59.6
            ev_demand = 95 + np.random.normal(0, 2)
        
        active_power = voltage * current * 2.2
        reactive_power = active_power * 0.55
        total_load = active_power * 2
        temperature = np.random.uniform(5, 35)
        
        data.append({
            'timestamp': timestamp,
            'hour': hour,
            'voltage': round(voltage, 3),
            'current': round(current, 2),
            'active_power': round(active_power, 2),
            'reactive_power': round(reactive_power, 2),
            'frequency': round(frequency, 3),
            'ev_demand': round(ev_demand, 2),
            'total_load': round(total_load, 2),
            'temperature': round(temperature, 2),
            'attack_type': attack_type
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Cyber-Attack Data for IDS Training")
    print("=" * 60)
    
    # Generate mixed dataset (normal + attack)
    print("\n[1] Generating mixed dataset (60% normal, 40% attack)...")
    mixed_df = generate_attack_data(100)
    mixed_df.to_csv('data_attack/attack_dataset.csv', index=False)
    print(f"    Saved: data_attack/attack_dataset.csv")
    print(f"    Total samples: {len(mixed_df)}")
    print(f"    Normal: {len(mixed_df[mixed_df['request_label'] == 'normal'])}")
    print(f"    Attack: {len(mixed_df[mixed_df['request_label'] == 'attack'])}")
    
    # Generate pure attack data
    print("\n[2] Generating pure attack data...")
    attack_df = generate_pure_attack_data(50)
    attack_df.to_csv('data_attack/pure_attack_data.csv', index=False)
    print(f"    Saved: data_attack/pure_attack_data.csv")
    print(f"    Attack types: FDI, Replay, DoS, Unauthorized")
    
    # Show sample attack patterns
    print("\n" + "=" * 60)
    print("Attack Pattern Examples:")
    print("=" * 60)
    
    print("\n[FDI Attack] - False Data Injection:")
    print("  - Voltage: 0.65-0.70 or 1.25-1.35 (normal: 0.95-1.05)")
    print("  - Current: 180-300A (normal: 35-80A)")
    print("  - Frequency: 57.5-58.0 or 62.0-62.5 Hz (normal: 59.98-60.02)")
    
    print("\n[Replay Attack] - Repeated identical values:")
    print("  - All values exactly same across multiple requests")
    print("  - Indicates automated/scripted attack")
    
    print("\n[DoS Attack] - Denial of Service:")
    print("  - Rapid random fluctuations in all parameters")
    print("  - Attempts to overwhelm the system")
    
    print("\n[Unauthorized Access] - Suspicious patterns:")
    print("  - High demand at unusual hours (3 AM)")
    print("  - Consistent abnormal patterns from same source")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)

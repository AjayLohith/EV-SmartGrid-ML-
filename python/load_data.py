"""
Load Data Module for EV SmartGrid ML Project
Handles loading and preprocessing of power grid data from Simulink exports
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import os


def load_mat_file(mat_path):
    """
    Load power signals from MATLAB .mat file exported from Simulink
    
    Parameters:
    -----------
    mat_path : str
        Path to the .mat file containing power signals
        
    Returns:
    --------
    dict : Dictionary containing signal data
    """
    try:
        mat_data = sio.loadmat(mat_path)
        print(f"Successfully loaded: {mat_path}")
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Available keys: {keys}")
        for key in keys:
            if isinstance(mat_data[key], np.ndarray):
                print(f"  {key}: shape {mat_data[key].shape}")
        return mat_data
    except FileNotFoundError:
        print(f"Error: File not found at {mat_path}")
        return None
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None


def load_simulink_csv(csv_path):
    """
    Load Simulink exported data from CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the simulink_data.csv file
        
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded: {csv_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Time range: {df['time'].min():.4f} to {df['time'].max():.4f} seconds")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def load_csv_data(csv_path):
    """
    Load dataset from CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV dataset file
        
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded: {csv_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def calculate_rms(signal, window_size):
    """
    Calculate RMS (Root Mean Square) value over a sliding window
    
    Parameters:
    -----------
    signal : np.array
        Input signal (current or voltage waveform)
    window_size : int
        Number of samples per window (one cycle)
        
    Returns:
    --------
    np.array : RMS values
    """
    rms_values = np.zeros(len(signal))
    
    for i in range(len(signal)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(signal), i + window_size // 2)
        window = signal[start_idx:end_idx]
        rms_values[i] = np.sqrt(np.mean(window ** 2))
    
    return rms_values


def calculate_power_from_waveforms(df, voltage_nominal=4160, frequency=60):
    """
    Calculate Active Power (P) and Reactive Power (Q) from current waveforms
    Assumes nominal voltage for IEEE 13-bus system
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with current waveform columns
    voltage_nominal : float
        Nominal line voltage in Volts (default 4160V for IEEE 13-bus)
    frequency : float
        System frequency in Hz
        
    Returns:
    --------
    pd.DataFrame : DataFrame with calculated power values
    """
    df_power = df.copy()
    
    # Calculate time step and samples per cycle
    time_diff = df['time'].diff().dropna()
    dt = time_diff[time_diff > 0].median()  # Use median to avoid zero issues
    if dt <= 0 or np.isnan(dt):
        dt = (df['time'].max() - df['time'].min()) / len(df)
    
    samples_per_cycle = max(int(round(1 / (frequency * dt))), 10)
    print(f"Time step: {dt:.6f}s, Samples per cycle: {samples_per_cycle}")
    
    # Current columns
    current_cols = [col for col in df.columns if col.startswith('I_')]
    
    for col in current_cols:
        # Calculate RMS current
        I_rms = calculate_rms(df[col].values, samples_per_cycle)
        df_power[f'{col}_rms'] = I_rms
        
        # Calculate apparent power (S = V * I)
        V_phase = voltage_nominal / np.sqrt(3)
        S = V_phase * I_rms  # VA
        
        # Assume power factor of 0.85 lagging for typical loads
        pf = 0.85
        P = S * pf  # Active Power (W)
        Q = S * np.sqrt(1 - pf**2)  # Reactive Power (VAR)
        
        df_power[f'{col}_P'] = P / 1000  # Convert to kW
        df_power[f'{col}_Q'] = Q / 1000  # Convert to kVAR
    
    return df_power


def process_simulink_data(df, downsample_factor=100):
    """
    Process raw Simulink data into ML-ready features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw Simulink data with current waveforms
    downsample_factor : int
        Factor to reduce data size (1 in N samples)
        
    Returns:
    --------
    pd.DataFrame : Processed dataset ready for ML
    """
    print(f"\nProcessing {len(df)} samples...")
    
    # Calculate power from waveforms
    df_power = calculate_power_from_waveforms(df)
    
    # Downsample to reduce size
    df_sampled = df_power.iloc[::downsample_factor].reset_index(drop=True)
    print(f"Downsampled to {len(df_sampled)} samples")
    
    # Create features
    df_features = pd.DataFrame()
    df_features['time'] = df_sampled['time']
    
    # RMS currents
    rms_cols = [col for col in df_sampled.columns if '_rms' in col]
    for col in rms_cols:
        df_features[col] = df_sampled[col]
    
    # Power values
    p_cols = [col for col in df_sampled.columns if '_P' in col]
    q_cols = [col for col in df_sampled.columns if '_Q' in col]
    
    for col in p_cols + q_cols:
        df_features[col] = df_sampled[col]
    
    # Total power
    if p_cols:
        df_features['total_P'] = df_sampled[p_cols].sum(axis=1)
        df_features['total_Q'] = df_sampled[q_cols].sum(axis=1)
    
    # Create stability label based on current magnitude variations
    if 'total_P' in df_features.columns:
        # Use multiple criteria for stability
        power_mean = df_features['total_P'].mean()
        power_std = df_features['total_P'].std()
        
        # High deviation from mean = unstable
        df_features['grid_stability'] = np.where(
            np.abs(df_features['total_P'] - power_mean) > power_std,
            'unstable', 'stable'
        )
        
        # If all same, use percentile-based labeling
        if df_features['grid_stability'].nunique() == 1:
            p75 = df_features['total_P'].quantile(0.75)
            p25 = df_features['total_P'].quantile(0.25)
            df_features['grid_stability'] = np.where(
                (df_features['total_P'] > p75) | (df_features['total_P'] < p25),
                'unstable', 'stable'
            )
    
    print(f"Created {len(df_features.columns)} features")
    return df_features


def preprocess_data(df):
    """
    Preprocess the loaded dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame : Preprocessed dataset
    """
    if df is None:
        return None
    
    # Make a copy
    df_processed = df.copy()
    
    # Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    # Remove duplicates
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    removed_rows = initial_rows - len(df_processed)
    if removed_rows > 0:
        print(f"Removed {removed_rows} duplicate rows")
    
    print(f"Preprocessed dataset shape: {df_processed.shape}")
    return df_processed


def split_features_target(df, target_column='grid_stability'):
    """
    Split dataset into features and target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataset
    target_column : str
        Name of the target column
        
    Returns:
    --------
    tuple : (X, y) features and target arrays
    """
    if df is None:
        return None, None
    
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    
    # Drop non-feature columns
    drop_cols = [target_column]
    if 'time' in df.columns:
        drop_cols.append('time')
    if 'timestamp' in df.columns:
        drop_cols.append('timestamp')
    
    X = df.drop(columns=drop_cols)
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def main():
    """Main function to load and process Simulink data"""
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simulink_csv = os.path.join(base_dir, 'data', 'simulink_data.csv')
    mat_path = os.path.join(base_dir, 'simulink', 'power_signals.mat')
    output_csv = os.path.join(base_dir, 'data', 'processed_data.csv')
    
    print("=" * 60)
    print("EV SmartGrid ML - Data Loading Module")
    print("=" * 60)
    
    # Load MAT file
    print("\n--- Loading MATLAB Data ---")
    mat_data = load_mat_file(mat_path)
    
    # Load Simulink CSV data
    print("\n--- Loading Simulink CSV Data ---")
    df_simulink = load_simulink_csv(simulink_csv)
    
    if df_simulink is not None:
        # Process into ML features
        print("\n--- Processing Simulink Data ---")
        df_processed = process_simulink_data(df_simulink, downsample_factor=100)
        
        # Save processed data
        df_processed.to_csv(output_csv, index=False)
        print(f"\nProcessed data saved to: {output_csv}")
        
        # Preview
        print("\n--- Processed Data Preview ---")
        print(df_processed.head(10))
        print(f"\nColumns: {list(df_processed.columns)}")
        
        if 'grid_stability' in df_processed.columns:
            print(f"\nStability distribution:")
            print(df_processed['grid_stability'].value_counts())
        
        return df_processed
    
    return None


if __name__ == "__main__":
    main()

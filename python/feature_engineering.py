"""
Feature Engineering Module for EV SmartGrid ML Project
Handles feature extraction and transformation for power grid data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import os


class FeatureEngineer:
    """Feature engineering class for EV SmartGrid data"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.selected_features = None
        
    def create_power_features(self, df):
        """
        Create power-related features from raw measurements
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with power measurements
            
        Returns:
        --------
        pd.DataFrame : Dataset with engineered features
        """
        df_feat = df.copy()
        
        # Power factor features
        if 'active_power' in df.columns and 'reactive_power' in df.columns:
            df_feat['apparent_power'] = np.sqrt(
                df_feat['active_power']**2 + df_feat['reactive_power']**2
            )
            df_feat['power_factor'] = df_feat['active_power'] / (
                df_feat['apparent_power'] + 1e-8
            )
        
        # Voltage features
        if 'voltage' in df.columns:
            df_feat['voltage_deviation'] = np.abs(df_feat['voltage'] - 1.0)
            df_feat['voltage_squared'] = df_feat['voltage'] ** 2
        
        # Current features
        if 'current' in df.columns:
            df_feat['current_squared'] = df_feat['current'] ** 2
        
        # EV charging features
        if 'ev_demand' in df.columns and 'total_load' in df.columns:
            df_feat['ev_load_ratio'] = df_feat['ev_demand'] / (
                df_feat['total_load'] + 1e-8
            )
        
        # Time-based features
        if 'hour' in df.columns:
            df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
            df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
            df_feat['is_peak_hour'] = df_feat['hour'].apply(
                lambda x: 1 if 17 <= x <= 21 else 0
            )
        
        print(f"Created {len(df_feat.columns) - len(df.columns)} new features")
        return df_feat
    
    def create_statistical_features(self, df, window_cols=None):
        """
        Create statistical features using rolling windows
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with time series data
        window_cols : list
            Columns to apply rolling statistics
            
        Returns:
        --------
        pd.DataFrame : Dataset with statistical features
        """
        df_stat = df.copy()
        
        if window_cols is None:
            window_cols = df.select_dtypes(include=[np.number]).columns[:5]
        
        for col in window_cols:
            if col in df.columns:
                # Rolling mean and std
                df_stat[f'{col}_rolling_mean'] = df[col].rolling(
                    window=3, min_periods=1
                ).mean()
                df_stat[f'{col}_rolling_std'] = df[col].rolling(
                    window=3, min_periods=1
                ).std().fillna(0)
                
                # Rate of change
                df_stat[f'{col}_diff'] = df[col].diff().fillna(0)
        
        return df_stat
    
    def scale_features(self, X, method='standard', fit=True):
        """
        Scale numerical features
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features to scale
        method : str
            Scaling method ('standard' or 'minmax')
        fit : bool
            Whether to fit the scaler or use existing
            
        Returns:
        --------
        np.array : Scaled features
        """
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def encode_labels(self, y, fit=True):
        """
        Encode categorical labels
        
        Parameters:
        -----------
        y : array-like
            Labels to encode
        fit : bool
            Whether to fit the encoder
            
        Returns:
        --------
        np.array : Encoded labels
        """
        if fit:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            if self.label_encoder is None:
                raise ValueError("Label encoder not fitted.")
            y_encoded = self.label_encoder.transform(y)
        
        return y_encoded
    
    def select_features(self, X, y, k=10, method='f_classif'):
        """
        Select top k features using statistical tests
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : array-like
            Target labels
        k : int
            Number of features to select
        method : str
            Selection method ('f_classif' or 'mutual_info')
            
        Returns:
        --------
        tuple : (selected features, feature indices)
        """
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support(indices=True)
        
        print(f"Selected {X_selected.shape[1]} features")
        return X_selected, self.selected_features


def engineer_features(df, target_col='grid_stability'):
    """
    Main feature engineering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    target_col : str
        Target column name
        
    Returns:
    --------
    tuple : (X_scaled, y_encoded, feature_engineer)
    """
    fe = FeatureEngineer()
    
    # Create features
    df_featured = fe.create_power_features(df)
    
    # Separate features and target
    if target_col in df_featured.columns:
        X = df_featured.drop(columns=[target_col])
        y = df_featured[target_col]
    else:
        X = df_featured
        y = None
    
    # Scale features
    X_scaled = fe.scale_features(X)
    
    # Encode labels if target exists
    y_encoded = fe.encode_labels(y) if y is not None else None
    
    return X_scaled, y_encoded, fe


def main():
    """Main function to demonstrate feature engineering"""
    from load_data import load_csv_data, preprocess_data
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'dataset.csv')
    
    print("=" * 50)
    print("EV SmartGrid ML - Feature Engineering Module")
    print("=" * 50)
    
    df = load_csv_data(csv_path)
    if df is not None:
        df = preprocess_data(df)
        X_scaled, y_encoded, fe = engineer_features(df)
        print(f"\nFinal feature matrix shape: {X_scaled.shape}")
        if y_encoded is not None:
            print(f"Encoded labels shape: {y_encoded.shape}")


if __name__ == "__main__":
    main()

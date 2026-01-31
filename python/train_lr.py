"""
Logistic Regression Training Module for EV SmartGrid ML Project
Trains and evaluates a Logistic Regression model for grid stability prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_data import load_csv_data, preprocess_data, split_features_target
from feature_engineering import FeatureEngineer


def train_logistic_regression(X_train, y_train, optimize=False):
    """
    Train a Logistic Regression model
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training labels
    optimize : bool
        Whether to perform hyperparameter optimization
        
    Returns:
    --------
    LogisticRegression : Trained model
    """
    if optimize:
        print("Performing hyperparameter optimization...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 500, 1000]
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        lr = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        lr.fit(X_train, y_train)
        return lr


def evaluate_model(model, X_test, y_test, model_name="Logistic Regression"):
    """
    Evaluate the trained model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : np.array
        Test features
    y_test : np.array
        Test labels
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    dict : Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n{model_name} Evaluation Results:")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics


def save_results(metrics, results_dir, model_name="lr"):
    """
    Save evaluation results to file
    
    Parameters:
    -----------
    metrics : dict
        Evaluation metrics
    results_dir : str
        Directory to save results
    model_name : str
        Model identifier for filenames
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Save accuracy to text file
    accuracy_path = os.path.join(results_dir, f'{model_name}_accuracy.txt')
    with open(accuracy_path, 'w') as f:
        f.write(f"Logistic Regression Model Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    
    print(f"Results saved to: {accuracy_path}")


def main():
    """Main training pipeline for Logistic Regression"""
    print("=" * 60)
    print("EV SmartGrid ML - Logistic Regression Training")
    print("=" * 60)
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    results_dir = os.path.join(base_dir, 'results')
    model_dir = os.path.join(base_dir, 'models')
    
    # Load and preprocess data
    print("\n--- Loading Data ---")
    df = load_csv_data(data_path)
    
    if df is None:
        print("Error: Could not load data. Exiting.")
        return
    
    df = preprocess_data(df)
    X, y = split_features_target(df, target_column='grid_stability')
    
    if X is None or y is None:
        print("Error: Could not split features and target. Exiting.")
        return
    
    # Feature engineering
    print("\n--- Feature Engineering ---")
    fe = FeatureEngineer()
    X_scaled = fe.scale_features(X)
    y_encoded = fe.encode_labels(y)
    
    # Split data
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    print("\n--- Training Logistic Regression ---")
    model = train_logistic_regression(X_train, y_train, optimize=False)
    
    # Cross-validation
    print("\n--- Cross-Validation ---")
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Evaluate model
    print("\n--- Evaluation ---")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save results
    print("\n--- Saving Results ---")
    save_results(metrics, results_dir, model_name="lr")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'logistic_regression.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model, metrics


if __name__ == "__main__":
    main()

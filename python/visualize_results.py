"""
Visualization Module for EV SmartGrid ML Project
Generates plots and visualizations for model evaluation and data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import os
import sys
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", 
                          save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix heatmap
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list
        Class labels
    title : str
        Plot title
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, title="Feature Importance",
                           save_path=None, top_n=15, figsize=(10, 8)):
    """
    Plot feature importance for models that support it
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ or coef_ attribute
    feature_names : list
        Names of features
    title : str
        Plot title
    save_path : str
        Path to save the figure
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        print("Model does not have feature importance attributes")
        return
    
    # Create DataFrame for sorting
    feat_imp = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path=None, figsize=(8, 6)):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Predicted probabilities for positive class
    title : str
        Plot title
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_learning_curve(model, X, y, title="Learning Curve", save_path=None,
                        cv=5, figsize=(10, 6)):
    """
    Plot learning curve showing training and validation scores
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Labels
    title : str
        Plot title
    save_path : str
        Path to save the figure
    cv : int
        Cross-validation folds
    figsize : tuple
        Figure size
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation score', color='green', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color='green')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_data_distribution(df, target_col='grid_stability', save_path=None, figsize=(12, 10)):
    """
    Plot data distribution and feature histograms
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target column name
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:9]
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_correlation_matrix(df, save_path=None, figsize=(12, 10)):
    """
    Plot correlation matrix heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to: {save_path}")
    
    plt.show()
    plt.close()


def compare_models(results_dict, save_path=None, figsize=(10, 6)):
    """
    Compare multiple models' performance metrics
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metric dicts as values
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, model in enumerate(models):
        values = [results_dict[model].get(m, 0) for m in metrics]
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()
    plt.close()


def main():
    """Main visualization pipeline"""
    print("=" * 60)
    print("EV SmartGrid ML - Visualization Module")
    print("=" * 60)
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    results_dir = os.path.join(base_dir, 'results')
    model_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Import data loading functions
    from load_data import load_csv_data, preprocess_data, split_features_target
    from feature_engineering import FeatureEngineer
    
    # Load data
    print("\n--- Loading Data ---")
    df = load_csv_data(data_path)
    
    if df is None:
        print("Error: Could not load data. Creating sample visualizations...")
        return
    
    df = preprocess_data(df)
    
    # Plot data distribution
    print("\n--- Plotting Data Distribution ---")
    plot_data_distribution(df, save_path=os.path.join(results_dir, 'data_distribution.png'))
    
    # Plot correlation matrix
    print("\n--- Plotting Correlation Matrix ---")
    plot_correlation_matrix(df, save_path=os.path.join(results_dir, 'correlation_matrix.png'))
    
    # Load models and generate confusion matrices if available
    X, y = split_features_target(df, target_column='grid_stability')
    
    if X is not None and y is not None:
        from sklearn.model_selection import train_test_split
        
        fe = FeatureEngineer()
        X_scaled = fe.scale_features(X)
        y_encoded = fe.encode_labels(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        # Try loading trained models
        lr_path = os.path.join(model_dir, 'logistic_regression.pkl')
        svm_path = os.path.join(model_dir, 'svm_model.pkl')
        
        results = {}
        
        if os.path.exists(lr_path):
            print("\n--- Logistic Regression Visualizations ---")
            lr_model = joblib.load(lr_path)
            y_pred_lr = lr_model.predict(X_test)
            
            plot_confusion_matrix(
                y_test, y_pred_lr,
                labels=['Unstable', 'Stable'],
                title='Logistic Regression - Confusion Matrix',
                save_path=os.path.join(results_dir, 'confusion_lr.png')
            )
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            results['Logistic Regression'] = {
                'accuracy': accuracy_score(y_test, y_pred_lr),
                'precision': precision_score(y_test, y_pred_lr, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_lr, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
            }
        
        if os.path.exists(svm_path):
            print("\n--- SVM Visualizations ---")
            svm_model = joblib.load(svm_path)
            y_pred_svm = svm_model.predict(X_test)
            
            plot_confusion_matrix(
                y_test, y_pred_svm,
                labels=['Unstable', 'Stable'],
                title='SVM - Confusion Matrix',
                save_path=os.path.join(results_dir, 'confusion_svm.png')
            )
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            results['SVM'] = {
                'accuracy': accuracy_score(y_test, y_pred_svm),
                'precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
            }
        
        # Model comparison
        if results:
            print("\n--- Model Comparison ---")
            compare_models(results, save_path=os.path.join(results_dir, 'model_comparison.png'))
    
    print("\n--- Visualization Complete ---")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

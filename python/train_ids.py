"""
Train Intrusion Detection System (IDS) Models
==============================================
Compare ML model performance on:
1. Normal data (grid stability prediction)
2. Attack data (intrusion detection)

This demonstrates how LR/SVM detect cyber-attacks in EV charging requests.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_ids_models(data_path, output_dir, scenario_name):
    """Train LR and SVM models for intrusion detection"""
    
    print(f"\n{'='*60}")
    print(f"Training IDS Models - {scenario_name}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nDataset: {data_path}")
    print(f"Samples: {len(df)}")
    
    # Determine label column
    if 'request_label' in df.columns:
        label_col = 'request_label'
    elif 'grid_stability' in df.columns:
        label_col = 'grid_stability'
    else:
        raise ValueError("No label column found")
    
    print(f"Label column: {label_col}")
    print(f"Class distribution:\n{df[label_col].value_counts()}")
    
    # Select features
    feature_cols = ['voltage', 'current', 'active_power', 'reactive_power', 
                    'frequency', 'ev_demand', 'total_load']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df[label_col]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Train Logistic Regression
    print("\n[1] Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred, average='weighted')
    lr_rec = recall_score(y_test, lr_pred, average='weighted')
    lr_f1 = f1_score(y_test, lr_pred, average='weighted')
    
    results['LR'] = {
        'accuracy': lr_acc,
        'precision': lr_prec,
        'recall': lr_rec,
        'f1': lr_f1,
        'confusion': confusion_matrix(y_test, lr_pred)
    }
    
    print(f"    Accuracy:  {lr_acc:.4f}")
    print(f"    Precision: {lr_prec:.4f}")
    print(f"    Recall:    {lr_rec:.4f}")
    print(f"    F1 Score:  {lr_f1:.4f}")
    
    # Train SVM
    print("\n[2] Training SVM (RBF Kernel)...")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred, average='weighted')
    svm_rec = recall_score(y_test, svm_pred, average='weighted')
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')
    
    results['SVM'] = {
        'accuracy': svm_acc,
        'precision': svm_prec,
        'recall': svm_rec,
        'f1': svm_f1,
        'confusion': confusion_matrix(y_test, svm_pred)
    }
    
    print(f"    Accuracy:  {svm_acc:.4f}")
    print(f"    Precision: {svm_prec:.4f}")
    print(f"    Recall:    {svm_rec:.4f}")
    print(f"    F1 Score:  {svm_f1:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save accuracy report
    with open(f"{output_dir}/ids_results.txt", 'w') as f:
        f.write(f"Intrusion Detection System Results - {scenario_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Features: {feature_cols}\n\n")
        f.write("Logistic Regression:\n")
        f.write(f"  Accuracy:  {lr_acc:.4f}\n")
        f.write(f"  Precision: {lr_prec:.4f}\n")
        f.write(f"  Recall:    {lr_rec:.4f}\n")
        f.write(f"  F1 Score:  {lr_f1:.4f}\n\n")
        f.write("Support Vector Machine:\n")
        f.write(f"  Accuracy:  {svm_acc:.4f}\n")
        f.write(f"  Precision: {svm_prec:.4f}\n")
        f.write(f"  Recall:    {svm_rec:.4f}\n")
        f.write(f"  F1 Score:  {svm_f1:.4f}\n")
    
    return results, le.classes_


def plot_comparison(normal_results, attack_results, output_dir):
    """Plot comparison between normal and attack detection"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    models = ['Logistic Regression', 'SVM']
    normal_acc = [normal_results['LR']['accuracy'], normal_results['SVM']['accuracy']]
    attack_acc = [attack_results['LR']['accuracy'], attack_results['SVM']['accuracy']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, normal_acc, width, label='Normal Data (Stability)', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, attack_acc, width, label='Attack Data (IDS)', color='#e74c3c')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy: Normal vs Attack Detection')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 2: LR Confusion Matrix (Attack Data)
    ax2 = axes[0, 1]
    cm = attack_results['LR']['confusion']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax2,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    ax2.set_title('LR Confusion Matrix (Attack Detection)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Plot 3: SVM Confusion Matrix (Attack Data)
    ax3 = axes[1, 0]
    cm = attack_results['SVM']['confusion']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax3,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    ax3.set_title('SVM Confusion Matrix (Attack Detection)')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Plot 4: All Metrics Comparison
    ax4 = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    lr_normal = [normal_results['LR']['accuracy'], normal_results['LR']['precision'],
                 normal_results['LR']['recall'], normal_results['LR']['f1']]
    lr_attack = [attack_results['LR']['accuracy'], attack_results['LR']['precision'],
                 attack_results['LR']['recall'], attack_results['LR']['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, lr_normal, width, label='Normal (Stability)', color='#3498db')
    ax4.bar(x + width/2, lr_attack, width, label='Attack (IDS)', color='#9b59b6')
    
    ax4.set_ylabel('Score')
    ax4.set_title('LR Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normal_vs_attack_comparison.png', dpi=150)
    plt.close()
    
    print(f"\nSaved: {output_dir}/normal_vs_attack_comparison.png")


def plot_attack_patterns(attack_data_path, output_dir):
    """Visualize attack patterns in the data"""
    
    df = pd.read_csv(attack_data_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Voltage Distribution
    ax1 = axes[0, 0]
    normal = df[df['request_label'] == 'normal']['voltage']
    attack = df[df['request_label'] == 'attack']['voltage']
    ax1.hist(normal, bins=15, alpha=0.7, label='Normal', color='#2ecc71')
    ax1.hist(attack, bins=15, alpha=0.7, label='Attack', color='#e74c3c')
    ax1.axvline(x=0.95, color='green', linestyle='--', label='Normal Range')
    ax1.axvline(x=1.05, color='green', linestyle='--')
    ax1.set_xlabel('Voltage (p.u.)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Voltage Distribution: Normal vs Attack')
    ax1.legend()
    
    # Plot 2: Current Distribution
    ax2 = axes[0, 1]
    normal = df[df['request_label'] == 'normal']['current']
    attack = df[df['request_label'] == 'attack']['current']
    ax2.hist(normal, bins=15, alpha=0.7, label='Normal', color='#2ecc71')
    ax2.hist(attack, bins=15, alpha=0.7, label='Attack', color='#e74c3c')
    ax2.axvline(x=80, color='orange', linestyle='--', label='Threshold')
    ax2.set_xlabel('Current (A)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Current Distribution: Normal vs Attack')
    ax2.legend()
    
    # Plot 3: Frequency Distribution
    ax3 = axes[1, 0]
    normal = df[df['request_label'] == 'normal']['frequency']
    attack = df[df['request_label'] == 'attack']['frequency']
    ax3.hist(normal, bins=15, alpha=0.7, label='Normal', color='#2ecc71')
    ax3.hist(attack, bins=15, alpha=0.7, label='Attack', color='#e74c3c')
    ax3.axvline(x=59.95, color='green', linestyle='--', label='Normal Range')
    ax3.axvline(x=60.05, color='green', linestyle='--')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Grid Frequency: Normal vs Attack')
    ax3.legend()
    
    # Plot 4: EV Demand Distribution
    ax4 = axes[1, 1]
    normal = df[df['request_label'] == 'normal']['ev_demand']
    attack = df[df['request_label'] == 'attack']['ev_demand']
    ax4.hist(normal, bins=15, alpha=0.7, label='Normal', color='#2ecc71')
    ax4.hist(attack, bins=15, alpha=0.7, label='Attack', color='#e74c3c')
    ax4.axvline(x=60, color='orange', linestyle='--', label='Suspicious Threshold')
    ax4.set_xlabel('EV Demand (kW)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('EV Charging Demand: Normal vs Attack')
    ax4.legend()
    
    plt.suptitle('Cyber-Attack Pattern Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attack_pattern_analysis.png', dpi=150)
    plt.close()
    
    print(f"Saved: {output_dir}/attack_pattern_analysis.png")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ML-Enhanced Cybersecure EV Charging - Intrusion Detection System")
    print("=" * 70)
    
    # Train on normal data (grid stability)
    print("\n" + "-" * 70)
    print("SCENARIO 1: Normal Data - Grid Stability Prediction")
    print("-" * 70)
    normal_results, _ = train_ids_models(
        'data/dataset.csv', 
        'results', 
        'Grid Stability'
    )
    
    # Train on attack data (intrusion detection)
    print("\n" + "-" * 70)
    print("SCENARIO 2: Attack Data - Intrusion Detection")
    print("-" * 70)
    attack_results, classes = train_ids_models(
        'data_attack/attack_dataset.csv', 
        'results_attack', 
        'Intrusion Detection'
    )
    
    # Generate comparison plots
    print("\n" + "-" * 70)
    print("Generating Comparison Visualizations")
    print("-" * 70)
    plot_comparison(normal_results, attack_results, 'results_attack')
    plot_attack_patterns('data_attack/attack_dataset.csv', 'results_attack')
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Why LR & SVM for Smart Grid Cybersecurity")
    print("=" * 70)
    print("""
    Normal Data (Grid Stability):
    - LR Accuracy:  {:.2%}
    - SVM Accuracy: {:.2%}
    
    Attack Data (Intrusion Detection):
    - LR Accuracy:  {:.2%}
    - SVM Accuracy: {:.2%}
    
    Key Findings:
    1. Both models effectively detect cyber-attacks in EV charging requests
    2. Attack patterns show clear differences from normal behavior
    3. Models can be deployed in Federated IDS for privacy-preserving detection
    """.format(
        normal_results['LR']['accuracy'],
        normal_results['SVM']['accuracy'],
        attack_results['LR']['accuracy'],
        attack_results['SVM']['accuracy']
    ))
    
    print("Output Files:")
    print("  - results/                  → Normal data results")
    print("  - results_attack/           → Attack data results")
    print("  - results_attack/normal_vs_attack_comparison.png")
    print("  - results_attack/attack_pattern_analysis.png")
    print("=" * 70)

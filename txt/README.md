# ML-Enhanced Cybersecure EV Charging Control in Smart Grid

## Situation

Smart grids with distributed EV charging stations face critical cybersecurity threats:

| Threat | Description | Impact |
|--------|-------------|--------|
| **False Data Injection (FDI)** | Attackers manipulate sensor readings | Grid operators make wrong decisions |
| **Replay Attack** | Repeated identical requests from scripts | System overwhelmed with fake requests |
| **DoS Attack** | Rapid fluctuating requests | Denial of service to legitimate users |
| **Unauthorized Access** | Suspicious charging requests at odd hours | Potential grid overload or theft |

**Problem:** Without detection, attackers can overload the grid, cause blackouts, or steal electricity.

## Task

Build a **Federated Intrusion Detection System (FIDS)** using Machine Learning to:

1. Detect cyber-attacks in EV charging requests in real-time
2. Classify requests as **Normal** or **Attack**
3. Protect grid stability while preserving user privacy
4. Enable safe RL-based charging optimization

## Action

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMART GRID CONTROL CENTER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   EV Charging Request                                            │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────────┐                                           │
│   │   IDS Module    │  ◄── LR/SVM Models                        │
│   │  (LR + SVM)     │      Trained via Federated Learning       │
│   └────────┬────────┘                                           │
│            │                                                     │
│     ┌──────┴──────┐                                             │
│     │             │                                              │
│     ▼             ▼                                              │
│  NORMAL        ATTACK                                            │
│     │             │                                              │
│     ▼             ▼                                              │
│  ┌──────┐    ┌──────────┐                                       │
│  │  RL  │    │  BLOCK   │                                       │
│  │Agent │    │ & ALERT  │                                       │
│  └──┬───┘    └──────────┘                                       │
│     │                                                            │
│     ▼                                                            │
│  Optimal Charging Schedule                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
EV_SmartGrid_ML/
├── data/                       # Normal operation data
│   ├── dataset.csv             # Training data (stable/unstable)
│   ├── simulink_data.csv       # Raw Simulink signals
│   └── processed_data.csv      # Processed features
│
├── data_attack/                # Attack/malicious data (SEPARATE)
│   ├── attack_dataset.csv      # Mixed normal + attack samples
│   └── pure_attack_data.csv    # Pure attack patterns
│
├── python/
│   ├── load_data.py            # Load Simulink data
│   ├── feature_engineering.py  # Feature extraction
│   ├── train_lr.py             # Train Logistic Regression
│   ├── train_svm.py            # Train SVM
│   ├── generate_attack_data.py # Generate attack patterns
│   ├── train_ids.py            # Train IDS & compare results
│   └── visualize_results.py    # Generate plots
│
├── results/                    # Normal data results
│   ├── lr_accuracy.txt
│   ├── svm_accuracy.txt
│   ├── confusion_lr.png
│   ├── confusion_svm.png
│   └── model_comparison.png
│
├── results_attack/             # Attack detection results (SEPARATE)
│   ├── ids_results.txt
│   ├── normal_vs_attack_comparison.png
│   └── attack_pattern_analysis.png
│
├── models/
│   ├── logistic_regression.pkl
│   └── svm_model.pkl
│
└── simulink/
    ├── ieee13bus.slx
    └── power_signals.mat
```

### Why LR and SVM for Smart Grid Cybersecurity?

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **Logistic Regression** | Fast, interpretable, low resource | Edge deployment on EV chargers |
| **SVM (RBF Kernel)** | Handles non-linear patterns | Complex attack detection |

**Benefits:**
- Both models work in **Federated Learning** - only model updates shared, not raw data
- Lightweight enough to run on local EV charging stations
- High accuracy in distinguishing normal vs attack patterns

## Results

### Graph Explanations

#### 1. Confusion Matrix (confusion_lr.png, confusion_svm.png)

```
                 Predicted
              Normal  Attack
Actual Normal   TN      FP
       Attack   FN      TP
```

| Cell | Meaning | Good/Bad |
|------|---------|----------|
| **TN (True Negative)** | Normal correctly identified | Good |
| **TP (True Positive)** | Attack correctly detected | Good |
| **FP (False Positive)** | Normal flagged as attack | Bad (annoying) |
| **FN (False Negative)** | Attack missed as normal | Bad (dangerous) |

**Our Results:**
- LR: 11 TN, 8 TP, 0 FP, 1 FN → 95% accuracy
- SVM: 11 TN, 8 TP, 0 FP, 1 FN → 95% accuracy

**Interpretation:** Only 1 attack was missed (FN=1). In cybersecurity, minimizing FN is critical.

#### 2. Model Comparison (model_comparison.png)

Shows side-by-side accuracy, precision, recall, and F1 score for both models.

| Metric | LR | SVM | Meaning |
|--------|-----|-----|---------|
| **Accuracy** | 95% | 95% | Overall correct predictions |
| **Precision** | 95.42% | 95.42% | Of predicted attacks, how many were real |
| **Recall** | 95% | 95% | Of real attacks, how many were detected |
| **F1 Score** | 94.96% | 94.96% | Balance of precision and recall |

#### 3. Correlation Matrix (correlation_matrix.png)

Shows relationships between features:
- **High correlation (close to 1):** Features move together
- **Low correlation (close to 0):** Features are independent

**Key Insights:**
- `active_power` ↔ `current`: High correlation (expected, P = V × I)
- `ev_demand` ↔ `total_load`: High correlation (EVs add to load)
- `frequency` ↔ `voltage`: Negative correlation (grid stress indicator)

#### 4. Data Distribution (data_distribution.png)

Histograms showing value ranges for each feature:
- **Normal distribution:** Bell curve shape → stable operation
- **Skewed distribution:** Indicates anomalies or attack patterns

#### 5. Attack Pattern Analysis (results_attack/attack_pattern_analysis.png)

Compares normal vs attack data distributions:

| Feature | Normal Range | Attack Range |
|---------|--------------|--------------|
| Voltage | 0.95 - 1.05 p.u. | 0.65 - 0.70 or 1.25 - 1.35 p.u. |
| Current | 35 - 80 A | 150 - 300 A |
| Frequency | 59.98 - 60.02 Hz | 57.5 - 62.5 Hz |
| EV Demand | 5 - 40 kW | 100 - 250 kW |

**Green bars:** Normal behavior
**Red bars:** Attack patterns

#### 6. Normal vs Attack Comparison (results_attack/normal_vs_attack_comparison.png)

Shows how models perform on:
- **Normal data:** Grid stability prediction (stable vs unstable)
- **Attack data:** Intrusion detection (normal vs attack)

### Model Performance Summary

| Scenario | Model | Accuracy | Purpose |
|----------|-------|----------|---------|
| Grid Stability | LR | 95.00% | Predict stable/unstable grid |
| Grid Stability | SVM | 95.00% | Predict stable/unstable grid |
| Intrusion Detection | LR | ~90-95% | Detect cyber-attacks |
| Intrusion Detection | SVM | ~90-95% | Detect cyber-attacks |

## How It Works (Real-World Example)

**Scenario:** An EV charging station receives many charging requests. Some attackers try to overload the grid by sending fake high-power requests.

**Step 1:** Request arrives at charging station
```
Request: {voltage: 0.68, current: 220A, ev_demand: 180kW, time: 3:00 AM}
```

**Step 2:** IDS analyzes features
- Voltage 0.68 → Outside normal range (0.95-1.05) → **Suspicious**
- Current 220A → Abnormally high → **Suspicious**
- EV Demand 180kW → Unrealistic → **Suspicious**
- Time 3:00 AM → Unusual for high demand → **Suspicious**

**Step 3:** ML model classifies
```
LR Model: P(attack) = 0.92 → ATTACK
SVM Model: Classification = ATTACK
```

**Step 4:** Action taken
- Request BLOCKED
- Alert sent to operator
- Attack logged for analysis

**Step 5:** Legitimate requests processed
- RL agent optimizes charging schedule
- Grid remains stable

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib

# 2. Generate attack data
python python/generate_attack_data.py

# 3. Train IDS models and compare
python python/train_ids.py

# 4. View results
# - results/           → Normal data analysis
# - results_attack/    → Attack detection results
```

## Folder Organization

| Folder | Contains | Purpose |
|--------|----------|---------|
| `data/` | Normal operation data | Grid stability training |
| `data_attack/` | Malicious/attack data | IDS training and testing |
| `results/` | Normal analysis outputs | Baseline performance |
| `results_attack/` | Attack analysis outputs | IDS performance |

**Important:** Attack data is kept SEPARATE to avoid contaminating normal training data.

## Requirements

- Python 3.8+
- MATLAB R2021a+ (for Simulink)
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, joblib

## Key Terms

| Term | Definition |
|------|------------|
| **IDS** | Intrusion Detection System - identifies attacks |
| **FIDS** | Federated IDS - distributed, privacy-preserving |
| **FDI** | False Data Injection - manipulated sensor data |
| **DoS** | Denial of Service - overwhelming system with requests |
| **RL** | Reinforcement Learning - learns optimal actions |
| **TN/TP/FP/FN** | Confusion matrix cells |

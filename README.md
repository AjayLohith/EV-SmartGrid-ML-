# EV SmartGrid ML Project

## Situation

Electric vehicles (EVs) are rapidly increasing on power grids. When multiple EVs charge simultaneously, the power grid can become **unstable** causing voltage drops and frequency changes. Grid operators need a way to predict instability before it occurs.

## Task

Build a Machine Learning system that:
1. Collects electrical signals from a simulated power grid (IEEE 13-bus model)
2. Processes current waveforms into meaningful features
3. Trains classification models to predict grid stability
4. Achieves high accuracy in distinguishing stable vs unstable conditions

## Action

### Project Structure

```
EV_SmartGrid_ML/
├── simulink/
│   ├── ieee13bus.slx           # Power grid simulation model
│   └── power_signals.mat       # Exported electrical signals
├── python/
│   ├── load_data.py            # Load and process data
│   ├── feature_engineering.py  # Create ML features
│   ├── train_lr.py             # Train Logistic Regression
│   ├── train_svm.py            # Train Support Vector Machine
│   └── visualize_results.py    # Generate result plots
├── data/
│   ├── dataset.csv             # Training data
│   ├── simulink_data.csv       # Raw simulation data
│   └── processed_data.csv      # Processed ML-ready data
├── results/
│   ├── lr_accuracy.txt         # LR performance metrics
│   ├── svm_accuracy.txt        # SVM performance metrics
│   ├── confusion_lr.png        # LR confusion matrix
│   ├── confusion_svm.png       # SVM confusion matrix
│   └── model_comparison.png    # Model comparison chart
├── models/
│   ├── logistic_regression.pkl # Saved LR model
│   └── svm_model.pkl           # Saved SVM model
└── README.md
```

### Step 1: Run Simulink Model

1. Open MATLAB
2. Open `simulink/ieee13bus.slx`
3. Click **Run** button
4. Wait for status to show "Ready"

**Variables created in MATLAB Workspace:**
- `Iabc9and` - Current signals at Bus 9
- `Iabs3and7` - Current signals at Bus 3 and 7
- `simout` - EV State of Charge data
- `tout` - Time vector

### Step 2: Verify Data in MATLAB

```matlab
% View signal structure
Iabc9and.signals(1)
Iabc9and.signals(2)

% Plot to verify data
plot(Iabc9and.time, Iabc9and.signals(1).values)
title('Current Signal at Bus 9')
xlabel('Time (s)')
ylabel('Current (A)')
grid on
```

### Step 3: Export Data for Python

```matlab
% Extract signals from struct
time = Iabc9and.time;
I_bus9_1 = Iabc9and.signals(1).values;
I_bus9_2 = Iabc9and.signals(2).values;
I_bus37_1 = Iabs3and7.signals(1).values;
I_bus37_2 = Iabs3and7.signals(2).values;
EV_SoC = simout;

% Save as MAT file
save('D:/EV_SmartGrid_ML/simulink/power_signals.mat', ...
    'time', 'I_bus9_1', 'I_bus9_2', 'I_bus37_1', 'I_bus37_2', 'EV_SoC', 'tout')

% Save as CSV file
T = table(time, I_bus9_1, I_bus9_2, I_bus37_1, I_bus37_2);
writetable(T, 'D:/EV_SmartGrid_ML/data/simulink_data.csv')
```

### Step 4: Install Python Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
```

| Library | Purpose |
|---------|---------|
| numpy | Numerical computations |
| pandas | Data manipulation |
| matplotlib | Plotting graphs |
| seaborn | Statistical visualizations |
| scikit-learn | Machine Learning algorithms |
| scipy | Load MATLAB .mat files |
| joblib | Save/load trained models |

### Step 5: Process Data

```bash
python python/load_data.py
```

**Processing Pipeline:**
1. Load CSV file with raw current waveforms
2. Calculate RMS (Root Mean Square) values
3. Calculate Active Power (P) and Reactive Power (Q)
4. Generate stability labels based on thresholds
5. Save processed data to `processed_data.csv`

**Output Features:**

| Feature | Description | Unit |
|---------|-------------|------|
| time | Timestamp | seconds |
| I_bus9_1_rms | RMS current at Bus 9 | Amps |
| I_bus9_1_P | Active power | kW |
| I_bus9_1_Q | Reactive power | kVAR |
| total_P | Total active power | kW |
| grid_stability | Classification label | stable/unstable |

### Step 6: Train Machine Learning Models

**Logistic Regression:**
```bash
python python/train_lr.py
```
- Linear classifier that draws a decision boundary
- Fast training and inference
- Good baseline model

**Support Vector Machine:**
```bash
python python/train_svm.py
```
- Non-linear classifier using RBF kernel
- Creates complex decision boundaries
- Better for non-linear patterns

### Step 7: Generate Visualizations

```bash
python python/visualize_results.py
```

**Output Files:**

| File | Description |
|------|-------------|
| confusion_lr.png | Logistic Regression confusion matrix |
| confusion_svm.png | SVM confusion matrix |
| model_comparison.png | Side-by-side accuracy comparison |

## Result

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 95.00% | 95.42% | 95.00% | 94.96% |
| SVM (RBF Kernel) | 95.00% | 95.42% | 95.00% | 94.96% |

**Interpretation:** Out of 20 test samples, 19 were predicted correctly (1 misclassified).

### Data Flow Diagram

```
Simulink Model (ieee13bus.slx)
         │
         ▼
Current Waveforms (56,956 samples)
         │
         ▼
Export to CSV (simulink_data.csv)
         │
         ▼
Python Processing (RMS, Power calculations)
         │
         ▼
Processed Data (570 samples after downsampling)
         │
         ▼
Train ML Models (80% train, 20% test)
         │
         ▼
Predictions (stable / unstable)
```

## Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib

# Process data
python python/load_data.py

# Train models
python python/train_lr.py
python python/train_svm.py

# Generate plots
python python/visualize_results.py
```

## Requirements

- Python 3.8+
- MATLAB R2021a+ (for Simulink simulation)

## Key Terms

| Term | Definition |
|------|------------|
| Machine Learning | Algorithms that learn patterns from data |
| Feature | Input variable used for prediction |
| Label | Output variable (stable/unstable) |
| RMS | Root Mean Square - average signal magnitude |
| Accuracy | Percentage of correct predictions |
| Confusion Matrix | Table showing prediction vs actual values |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| File not found | Verify MATLAB exported data before running Python |
| Import error | Run `pip install <library_name>` |
| Low accuracy | Check data quality or adjust hyperparameters |

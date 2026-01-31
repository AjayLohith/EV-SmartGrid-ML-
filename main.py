"""
EV SmartGrid ML - Main Runner
==============================
Single entry point to run all project scenarios.

Usage:
    python main.py              # Show menu
    python main.py --normal     # Run normal grid analysis
    python main.py --attack     # Run attack detection
    python main.py --compare    # Compare both scenarios
    python main.py --all        # Run everything
"""

import os
import sys
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 70)
    print("   ML-Enhanced Cybersecure EV Charging Control in Smart Grid")
    print("   Federated Intrusion Detection System (FIDS)")
    print("=" * 70)

def print_menu():
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SELECT SCENARIO                              │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   [1] NORMAL GRID      - Train models on normal operation data │
    │                          Predict: stable vs unstable            │
    │                                                                 │
    │   [2] ATTACK DETECTION - Train IDS on attack data              │
    │                          Predict: normal vs attack              │
    │                                                                 │
    │   [3] COMPARE BOTH     - Side-by-side comparison               │
    │                          Shows difference in detection          │
    │                                                                 │
    │   [4] GENERATE ATTACK  - Create new attack dataset             │
    │                          Simulates FDI, Replay, DoS attacks    │
    │                                                                 │
    │   [5] VISUALIZE        - Generate all plots                    │
    │                                                                 │
    │   [6] RUN ALL          - Execute complete pipeline             │
    │                                                                 │
    │   [0] EXIT                                                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)

def run_script(script_name, description):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {description}")
    print(f"  File: python/{script_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, f"python/{script_name}"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode == 0

def scenario_normal():
    """Run normal grid stability analysis"""
    print("\n" + "=" * 70)
    print("  SCENARIO: NORMAL GRID OPERATION")
    print("  Data: data/dataset.csv")
    print("  Task: Predict grid stability (stable vs unstable)")
    print("=" * 70)
    
    print("\n[Step 1/3] Loading and processing data...")
    run_script("load_data.py", "Data Loading")
    
    print("\n[Step 2/3] Training Logistic Regression...")
    run_script("train_lr.py", "Logistic Regression Training")
    
    print("\n[Step 3/3] Training SVM...")
    run_script("train_svm.py", "SVM Training")
    
    print("\n" + "-" * 70)
    print("  NORMAL SCENARIO COMPLETE")
    print("  Results saved in: results/")
    print("  - lr_accuracy.txt")
    print("  - svm_accuracy.txt")
    print("  - confusion_lr.png")
    print("  - confusion_svm.png")
    print("-" * 70)

def scenario_attack():
    """Run attack detection analysis"""
    print("\n" + "=" * 70)
    print("  SCENARIO: CYBER-ATTACK DETECTION")
    print("  Data: data_attack/attack_dataset.csv")
    print("  Task: Detect intrusions (normal vs attack)")
    print("=" * 70)
    
    # Check if attack data exists
    if not os.path.exists("data_attack/attack_dataset.csv"):
        print("\n[!] Attack dataset not found. Generating...")
        run_script("generate_attack_data.py", "Attack Data Generation")
    
    print("\n[Step 1/1] Training IDS models on attack data...")
    run_script("train_ids.py", "Intrusion Detection System Training")
    
    print("\n" + "-" * 70)
    print("  ATTACK SCENARIO COMPLETE")
    print("  Results saved in: results_attack/")
    print("  - ids_results.txt")
    print("  - normal_vs_attack_comparison.png")
    print("  - attack_pattern_analysis.png")
    print("-" * 70)

def scenario_compare():
    """Compare normal vs attack scenarios"""
    print("\n" + "=" * 70)
    print("  SCENARIO: COMPARISON (Normal vs Attack)")
    print("=" * 70)
    
    # Run both scenarios
    scenario_normal()
    scenario_attack()
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    NORMAL vs ATTACK                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  NORMAL GRID (data/dataset.csv)                                │
    │  ├── Purpose: Predict grid stability                           │
    │  ├── Classes: stable, unstable                                 │
    │  ├── Use case: Load balancing, demand prediction               │
    │  └── Results: results/                                         │
    │                                                                 │
    │  ATTACK DATA (data_attack/attack_dataset.csv)                  │
    │  ├── Purpose: Detect cyber-attacks                             │
    │  ├── Classes: normal, attack                                   │
    │  ├── Attack types: FDI, Replay, DoS, Unauthorized              │
    │  └── Results: results_attack/                                  │
    │                                                                 │
    │  KEY DIFFERENCE:                                               │
    │  - Normal data → Grid health monitoring                        │
    │  - Attack data → Cybersecurity intrusion detection             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)

def scenario_generate_attack():
    """Generate new attack dataset"""
    print("\n" + "=" * 70)
    print("  GENERATING ATTACK DATA")
    print("=" * 70)
    run_script("generate_attack_data.py", "Attack Data Generation")

def scenario_visualize():
    """Generate visualizations"""
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    run_script("visualize_results.py", "Visualization Generation")
    
    print("\n  Generated plots:")
    print("  - results/confusion_lr.png")
    print("  - results/confusion_svm.png")
    print("  - results/model_comparison.png")
    print("  - results/correlation_matrix.png")
    print("  - results/data_distribution.png")

def scenario_all():
    """Run complete pipeline"""
    print("\n" + "=" * 70)
    print("  RUNNING COMPLETE PIPELINE")
    print("=" * 70)
    
    print("\n[1/5] Generating attack data...")
    run_script("generate_attack_data.py", "Attack Data Generation")
    
    print("\n[2/5] Processing normal data...")
    run_script("load_data.py", "Data Loading")
    
    print("\n[3/5] Training normal grid models...")
    run_script("train_lr.py", "Logistic Regression")
    run_script("train_svm.py", "SVM")
    
    print("\n[4/5] Training IDS models...")
    run_script("train_ids.py", "Intrusion Detection System")
    
    print("\n[5/5] Generating visualizations...")
    run_script("visualize_results.py", "Visualizations")
    
    print("\n" + "=" * 70)
    print("  ALL TASKS COMPLETE!")
    print("=" * 70)
    print("""
    Output Files:
    
    Normal Grid Results (results/):
    ├── lr_accuracy.txt
    ├── svm_accuracy.txt
    ├── confusion_lr.png
    ├── confusion_svm.png
    ├── model_comparison.png
    ├── correlation_matrix.png
    └── data_distribution.png
    
    Attack Detection Results (results_attack/):
    ├── ids_results.txt
    ├── normal_vs_attack_comparison.png
    └── attack_pattern_analysis.png
    """)

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--normal', '-n', '1']:
            scenario_normal()
        elif arg in ['--attack', '-a', '2']:
            scenario_attack()
        elif arg in ['--compare', '-c', '3']:
            scenario_compare()
        elif arg in ['--generate', '-g', '4']:
            scenario_generate_attack()
        elif arg in ['--visualize', '-v', '5']:
            scenario_visualize()
        elif arg in ['--all', '-all', '6']:
            scenario_all()
        elif arg in ['--help', '-h']:
            print(__doc__)
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
        return
    
    # Interactive menu
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("    Enter choice [0-6]: ").strip()
        
        if choice == '1':
            scenario_normal()
        elif choice == '2':
            scenario_attack()
        elif choice == '3':
            scenario_compare()
        elif choice == '4':
            scenario_generate_attack()
        elif choice == '5':
            scenario_visualize()
        elif choice == '6':
            scenario_all()
        elif choice == '0':
            print("\n    Goodbye!")
            break
        else:
            print("\n    Invalid choice. Please try again.")
        
        input("\n    Press Enter to continue...")

if __name__ == "__main__":
    main()

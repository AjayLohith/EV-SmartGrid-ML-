"""
Real-Time Simulink ↔ Python ML Integration
===========================================
This script connects to Simulink and performs real-time attack detection.

METHOD 2: MATLAB Engine API (Recommended for your setup)
"""

import numpy as np
import pandas as pd
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler

# Try to import MATLAB engine
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("[!] MATLAB Engine not found. Install with:")
    print("    cd <MATLAB_ROOT>/extern/engines/python")
    print("    python setup.py install")


class RealTimeIDS:
    """Real-Time Intrusion Detection System"""
    
    def __init__(self, model_path='models/'):
        """Load trained ML models"""
        print("=" * 60)
        print("Initializing Real-Time IDS")
        print("=" * 60)
        
        # Load models
        self.lr_model = joblib.load(f'{model_path}/logistic_regression.pkl')
        self.svm_model = joblib.load(f'{model_path}/svm_model.pkl')
        print("[✓] ML models loaded")
        
        # Feature scaler (fitted on training data statistics)
        self.scaler = StandardScaler()
        # These are approximate values from training data
        self.scaler.mean_ = np.array([1.0, 60.0, 120.0, 50.0, 60.0, 25.0, 200.0])
        self.scaler.scale_ = np.array([0.05, 25.0, 50.0, 20.0, 0.1, 15.0, 80.0])
        
        # Detection thresholds
        self.attack_threshold = 0.5
        self.alert_count = 0
        self.total_requests = 0
        
        # Normal ranges for rule-based backup
        self.normal_ranges = {
            'voltage': (0.90, 1.10),
            'current': (0, 150),
            'active_power': (0, 300),
            'reactive_power': (0, 150),
            'frequency': (59.5, 60.5),
            'ev_demand': (0, 80),
            'total_load': (0, 500)
        }
        
        print("[✓] IDS initialized and ready")
        print("=" * 60)
    
    def extract_features(self, data_dict):
        """Extract features from incoming data"""
        features = np.array([
            data_dict.get('voltage', 1.0),
            data_dict.get('current', 50.0),
            data_dict.get('active_power', 100.0),
            data_dict.get('reactive_power', 40.0),
            data_dict.get('frequency', 60.0),
            data_dict.get('ev_demand', 20.0),
            data_dict.get('total_load', 150.0)
        ]).reshape(1, -1)
        
        return self.scaler.transform(features)
    
    def rule_based_check(self, data_dict):
        """Backup rule-based anomaly detection"""
        anomalies = []
        
        for feature, (min_val, max_val) in self.normal_ranges.items():
            value = data_dict.get(feature, 0)
            if value < min_val or value > max_val:
                anomalies.append(f"{feature}={value:.2f} (expected {min_val}-{max_val})")
        
        return anomalies
    
    def detect(self, data_dict):
        """
        Main detection function
        
        Input: Dictionary with sensor readings
        Output: Detection result with confidence
        """
        self.total_requests += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract and scale features
        features = self.extract_features(data_dict)
        
        # ML-based detection
        lr_prob = self.lr_model.predict_proba(features)[0]
        lr_prediction = self.lr_model.predict(features)[0]
        
        svm_prediction = self.svm_model.predict(features)[0]
        
        # Rule-based backup
        rule_anomalies = self.rule_based_check(data_dict)
        
        # Combined decision
        is_attack = False
        confidence = 0.0
        attack_type = "unknown"
        
        # If ML says attack
        if lr_prediction == 1 or svm_prediction == 1:
            is_attack = True
            confidence = max(lr_prob[1] if len(lr_prob) > 1 else 0.5, 0.5)
        
        # If rules detect anomaly
        if len(rule_anomalies) >= 2:
            is_attack = True
            confidence = max(confidence, 0.7)
        
        # Determine attack type based on patterns
        if is_attack:
            attack_type = self._classify_attack_type(data_dict)
            self.alert_count += 1
        
        # Build result
        result = {
            'timestamp': timestamp,
            'is_attack': is_attack,
            'confidence': confidence,
            'attack_type': attack_type if is_attack else 'none',
            'lr_prediction': 'attack' if lr_prediction == 1 else 'normal',
            'svm_prediction': 'attack' if svm_prediction == 1 else 'normal',
            'rule_anomalies': rule_anomalies,
            'action': 'BLOCK' if is_attack else 'ALLOW',
            'input_data': data_dict
        }
        
        return result
    
    def _classify_attack_type(self, data_dict):
        """Classify the type of attack based on patterns"""
        voltage = data_dict.get('voltage', 1.0)
        current = data_dict.get('current', 50.0)
        frequency = data_dict.get('frequency', 60.0)
        ev_demand = data_dict.get('ev_demand', 20.0)
        
        # FDI: Extreme values
        if voltage < 0.8 or voltage > 1.2 or current > 200 or frequency < 58 or frequency > 62:
            return "FDI (False Data Injection)"
        
        # Unauthorized: High demand at unusual times
        if ev_demand > 100:
            return "Unauthorized Access"
        
        # DoS: Multiple anomalies
        return "Potential DoS/Replay"
    
    def print_alert(self, result):
        """Print formatted alert"""
        if result['is_attack']:
            print("\n" + "!" * 70)
            print("!!! ATTACK DETECTED !!!")
            print("!" * 70)
            print(f"  Time:       {result['timestamp']}")
            print(f"  Type:       {result['attack_type']}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print(f"  LR Model:   {result['lr_prediction']}")
            print(f"  SVM Model:  {result['svm_prediction']}")
            print(f"  Action:     {result['action']}")
            if result['rule_anomalies']:
                print(f"  Anomalies:  {', '.join(result['rule_anomalies'])}")
            print("!" * 70 + "\n")
        else:
            print(f"[{result['timestamp']}] Normal request - ALLOWED")
    
    def get_stats(self):
        """Get detection statistics"""
        return {
            'total_requests': self.total_requests,
            'attacks_detected': self.alert_count,
            'attack_rate': self.alert_count / max(self.total_requests, 1) * 100
        }


class SimulinkConnector:
    """Connect to MATLAB/Simulink for real-time data"""
    
    def __init__(self):
        self.eng = None
        self.connected = False
    
    def connect(self):
        """Start MATLAB engine"""
        if not MATLAB_AVAILABLE:
            print("[!] MATLAB Engine not available")
            return False
        
        print("[*] Starting MATLAB engine...")
        try:
            self.eng = matlab.engine.start_matlab()
            self.connected = True
            print("[✓] Connected to MATLAB")
            return True
        except Exception as e:
            print(f"[!] Failed to connect: {e}")
            return False
    
    def connect_to_existing(self):
        """Connect to already running MATLAB"""
        if not MATLAB_AVAILABLE:
            return False
        
        print("[*] Connecting to existing MATLAB session...")
        try:
            self.eng = matlab.engine.connect_matlab()
            self.connected = True
            print("[✓] Connected to existing MATLAB")
            return True
        except Exception as e:
            print(f"[!] No existing MATLAB found: {e}")
            return self.connect()
    
    def get_workspace_variables(self):
        """Get current values from MATLAB workspace"""
        if not self.connected:
            return None
        
        try:
            data = {
                'voltage': float(self.eng.eval("voltage")),
                'current': float(self.eng.eval("current")),
                'active_power': float(self.eng.eval("active_power")),
                'reactive_power': float(self.eng.eval("reactive_power")),
                'frequency': float(self.eng.eval("frequency")),
                'ev_demand': float(self.eng.eval("ev_demand")),
                'total_load': float(self.eng.eval("total_load"))
            }
            return data
        except Exception as e:
            print(f"[!] Error reading variables: {e}")
            return None
    
    def get_simulink_signals(self, model_name='ieee13bus'):
        """Get signals from running Simulink model"""
        if not self.connected:
            return None
        
        try:
            # Get simulation time
            sim_time = self.eng.eval(f"get_param('{model_name}', 'SimulationTime')")
            
            # Read To Workspace blocks or Scope data
            # Adjust these variable names based on your model
            data = {
                'time': float(sim_time),
                'voltage': float(self.eng.eval("Iabc9and.signals(1).values(end)")),
                'current': float(self.eng.eval("Iabc9and.signals(2).values(end)")),
                'frequency': 60.0,  # From your model
                'ev_demand': float(self.eng.eval("simout(end)")),
            }
            
            # Calculate derived values
            data['active_power'] = abs(data['voltage'] * data['current'])
            data['reactive_power'] = data['active_power'] * 0.4
            data['total_load'] = data['active_power'] + data['ev_demand']
            
            return data
        except Exception as e:
            print(f"[!] Error reading Simulink: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from MATLAB"""
        if self.eng:
            self.eng.quit()
            self.connected = False
            print("[✓] Disconnected from MATLAB")


def demo_without_matlab():
    """Demo the IDS without MATLAB connection"""
    print("\n" + "=" * 70)
    print("DEMO MODE: Simulating real-time detection without MATLAB")
    print("=" * 70)
    
    ids = RealTimeIDS()
    
    # Simulate some requests
    test_cases = [
        # Normal requests
        {'voltage': 1.02, 'current': 45.2, 'active_power': 85.6, 
         'reactive_power': 32.1, 'frequency': 60.01, 'ev_demand': 12.5, 
         'total_load': 156.3},
        
        {'voltage': 0.98, 'current': 52.3, 'active_power': 95.4,
         'reactive_power': 38.2, 'frequency': 59.95, 'ev_demand': 15.8,
         'total_load': 168.5},
        
        # FDI Attack - extreme voltage
        {'voltage': 0.65, 'current': 287.96, 'active_power': 443.46,
         'reactive_power': 243.9, 'frequency': 57.5, 'ev_demand': 180.14,
         'total_load': 886.92},
        
        # Normal
        {'voltage': 1.01, 'current': 42.8, 'active_power': 78.4,
         'reactive_power': 28.9, 'frequency': 60.00, 'ev_demand': 8.3,
         'total_load': 142.7},
        
        # Replay Attack - repeated values
        {'voltage': 0.95, 'current': 85.5, 'active_power': 178.69,
         'reactive_power': 98.28, 'frequency': 59.85, 'ev_demand': 45.0,
         'total_load': 357.39},
        
        # Unauthorized - high demand
        {'voltage': 0.88, 'current': 143.9, 'active_power': 278.6,
         'reactive_power': 153.23, 'frequency': 59.6, 'ev_demand': 150.23,
         'total_load': 557.19},
    ]
    
    print("\nProcessing simulated requests...\n")
    
    for i, data in enumerate(test_cases, 1):
        print(f"\n--- Request #{i} ---")
        result = ids.detect(data)
        ids.print_alert(result)
        time.sleep(0.5)  # Simulate delay between requests
    
    # Print statistics
    stats = ids.get_stats()
    print("\n" + "=" * 70)
    print("DETECTION STATISTICS")
    print("=" * 70)
    print(f"  Total Requests:    {stats['total_requests']}")
    print(f"  Attacks Detected:  {stats['attacks_detected']}")
    print(f"  Attack Rate:       {stats['attack_rate']:.1f}%")
    print("=" * 70)


def real_time_with_matlab():
    """Real-time detection with MATLAB connection"""
    print("\n" + "=" * 70)
    print("REAL-TIME MODE: Connecting to MATLAB/Simulink")
    print("=" * 70)
    
    # Initialize
    ids = RealTimeIDS()
    connector = SimulinkConnector()
    
    # Connect to MATLAB
    if not connector.connect_to_existing():
        print("[!] Cannot connect to MATLAB. Running demo mode instead.")
        demo_without_matlab()
        return
    
    print("\n[*] Starting real-time monitoring...")
    print("[*] Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Get current data from Simulink
            data = connector.get_simulink_signals()
            
            if data:
                result = ids.detect(data)
                ids.print_alert(result)
            else:
                print("[!] No data received")
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\n[*] Stopping monitoring...")
    
    finally:
        connector.disconnect()
        stats = ids.get_stats()
        print(f"\nFinal Stats: {stats['attacks_detected']} attacks detected "
              f"out of {stats['total_requests']} requests")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   REAL-TIME INTRUSION DETECTION SYSTEM")
    print("   For EV SmartGrid Cybersecurity")
    print("=" * 70)
    
    print("\nSelect mode:")
    print("  [1] Demo mode (no MATLAB required)")
    print("  [2] Real-time with MATLAB")
    
    choice = input("\nEnter choice [1/2]: ").strip()
    
    if choice == '2':
        real_time_with_matlab()
    else:
        demo_without_matlab()

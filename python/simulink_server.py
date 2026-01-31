"""
TCP/UDP Server for Real-Time Simulink Communication
====================================================
This script creates a UDP server that receives data from Simulink
and sends back attack detection results in real-time.

Run this BEFORE starting your Simulink model.
"""

import socket
import struct
import numpy as np
import joblib
import time
import threading
from sklearn.preprocessing import StandardScaler

class UDPServer:
    """UDP Server for Simulink communication"""
    
    def __init__(self, receive_port=5000, send_port=5001, simulink_ip='127.0.0.1'):
        self.receive_port = receive_port
        self.send_port = send_port
        self.simulink_ip = simulink_ip
        
        # Create sockets
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.bind(('0.0.0.0', receive_port))
        self.recv_socket.settimeout(1.0)  # 1 second timeout
        
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Load ML models
        print("[*] Loading ML models...")
        self.lr_model = joblib.load('models/logistic_regression.pkl')
        self.svm_model = joblib.load('models/svm_model.pkl')
        
        # Scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([1.0, 60.0, 120.0, 50.0, 60.0, 25.0, 200.0])
        self.scaler.scale_ = np.array([0.05, 25.0, 50.0, 20.0, 0.1, 15.0, 80.0])
        
        # Stats
        self.running = False
        self.request_count = 0
        self.attack_count = 0
        
        print(f"[✓] UDP Server initialized")
        print(f"    Receiving on port: {receive_port}")
        print(f"    Sending to: {simulink_ip}:{send_port}")
    
    def detect(self, data):
        """Run ML detection on incoming data"""
        # Data format: [voltage, current, active_power, reactive_power, frequency, ev_demand, total_load]
        features = np.array(data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        lr_pred = self.lr_model.predict(features_scaled)[0]
        svm_pred = self.svm_model.predict(features_scaled)[0]
        
        # Combined decision
        is_attack = 1 if (lr_pred == 1 or svm_pred == 1) else 0
        
        # Calculate confidence
        try:
            lr_prob = self.lr_model.predict_proba(features_scaled)[0]
            confidence = lr_prob[1] if len(lr_prob) > 1 else 0.5
        except:
            confidence = 0.5 if is_attack else 0.0
        
        return is_attack, confidence
    
    def start(self):
        """Start the UDP server"""
        self.running = True
        print("\n" + "=" * 60)
        print("UDP Server Running - Waiting for Simulink data...")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        while self.running:
            try:
                # Receive data from Simulink
                data_bytes, addr = self.recv_socket.recvfrom(1024)
                
                # Unpack data (7 doubles = 56 bytes)
                if len(data_bytes) >= 56:
                    data = struct.unpack('7d', data_bytes[:56])
                else:
                    # Try single precision floats
                    data = struct.unpack(f'{len(data_bytes)//4}f', data_bytes)
                
                self.request_count += 1
                
                # Detect attack
                is_attack, confidence = self.detect(data)
                
                if is_attack:
                    self.attack_count += 1
                    print(f"[!] ATTACK DETECTED - Confidence: {confidence*100:.1f}%")
                    print(f"    Data: V={data[0]:.2f}, I={data[1]:.2f}, "
                          f"f={data[4]:.2f}Hz, EV={data[5]:.2f}kW")
                else:
                    print(f"[✓] Normal - V={data[0]:.2f}, I={data[1]:.2f}")
                
                # Send response back to Simulink
                response = struct.pack('2d', float(is_attack), confidence)
                self.send_socket.sendto(response, (self.simulink_ip, self.send_port))
                
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[!] Error: {e}")
        
        self.stop()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        self.recv_socket.close()
        self.send_socket.close()
        
        print("\n" + "=" * 60)
        print("Server Stopped")
        print(f"  Total requests: {self.request_count}")
        print(f"  Attacks detected: {self.attack_count}")
        print("=" * 60)


class TCPServer:
    """TCP Server for reliable Simulink communication"""
    
    def __init__(self, port=5000):
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen(1)
        
        # Load ML models
        print("[*] Loading ML models...")
        self.lr_model = joblib.load('models/logistic_regression.pkl')
        self.svm_model = joblib.load('models/svm_model.pkl')
        
        # Scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([1.0, 60.0, 120.0, 50.0, 60.0, 25.0, 200.0])
        self.scaler.scale_ = np.array([0.05, 25.0, 50.0, 20.0, 0.1, 15.0, 80.0])
        
        print(f"[✓] TCP Server listening on port {port}")
    
    def detect(self, data):
        """Run ML detection"""
        features = np.array(data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        lr_pred = self.lr_model.predict(features_scaled)[0]
        svm_pred = self.svm_model.predict(features_scaled)[0]
        
        is_attack = 1 if (lr_pred == 1 or svm_pred == 1) else 0
        
        try:
            lr_prob = self.lr_model.predict_proba(features_scaled)[0]
            confidence = lr_prob[1] if len(lr_prob) > 1 else 0.5
        except:
            confidence = 0.5 if is_attack else 0.0
        
        return is_attack, confidence
    
    def start(self):
        """Start TCP server"""
        print("\n" + "=" * 60)
        print("TCP Server Running - Waiting for connection...")
        print("=" * 60)
        
        try:
            client_socket, addr = self.server_socket.accept()
            print(f"[✓] Connected to {addr}")
            
            request_count = 0
            attack_count = 0
            
            while True:
                data_bytes = client_socket.recv(1024)
                if not data_bytes:
                    break
                
                # Unpack data
                data = struct.unpack('7d', data_bytes[:56])
                request_count += 1
                
                # Detect
                is_attack, confidence = self.detect(data)
                
                if is_attack:
                    attack_count += 1
                    print(f"[!] ATTACK #{attack_count} - Confidence: {confidence*100:.1f}%")
                
                # Send response
                response = struct.pack('2d', float(is_attack), confidence)
                client_socket.send(response)
            
            print(f"\nSession ended. Attacks: {attack_count}/{request_count}")
            
        except KeyboardInterrupt:
            pass
        finally:
            self.server_socket.close()


def file_watcher_mode():
    """Watch a file for new data and process it"""
    import os
    
    print("\n" + "=" * 60)
    print("FILE WATCHER MODE")
    print("Watching: data/realtime_data.csv")
    print("=" * 60)
    
    # Load models
    lr_model = joblib.load('models/logistic_regression.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    
    scaler = StandardScaler()
    scaler.mean_ = np.array([1.0, 60.0, 120.0, 50.0, 60.0, 25.0, 200.0])
    scaler.scale_ = np.array([0.05, 25.0, 50.0, 20.0, 0.1, 15.0, 80.0])
    
    watch_file = 'data/realtime_data.csv'
    last_modified = 0
    
    print("\n[*] Waiting for file updates...")
    print("[*] Export data from MATLAB to trigger detection")
    print("[*] Press Ctrl+C to stop\n")
    
    try:
        while True:
            if os.path.exists(watch_file):
                current_modified = os.path.getmtime(watch_file)
                
                if current_modified > last_modified:
                    last_modified = current_modified
                    print(f"\n[*] File updated at {time.strftime('%H:%M:%S')}")
                    
                    # Read and process
                    import pandas as pd
                    df = pd.read_csv(watch_file)
                    
                    feature_cols = ['voltage', 'current', 'active_power', 
                                   'reactive_power', 'frequency', 'ev_demand', 'total_load']
                    feature_cols = [c for c in feature_cols if c in df.columns]
                    
                    X = df[feature_cols].values
                    X_scaled = scaler.transform(X)
                    
                    # Detect
                    lr_preds = lr_model.predict(X_scaled)
                    svm_preds = svm_model.predict(X_scaled)
                    
                    attacks = np.logical_or(lr_preds == 1, svm_preds == 1)
                    
                    print(f"[*] Processed {len(df)} samples")
                    print(f"[*] Attacks detected: {attacks.sum()}")
                    
                    if attacks.sum() > 0:
                        print("\n[!] ATTACKS FOUND AT INDICES:")
                        attack_indices = np.where(attacks)[0]
                        for idx in attack_indices[:10]:  # Show first 10
                            print(f"    Row {idx}: {df.iloc[idx][feature_cols].to_dict()}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Stopped watching")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   SIMULINK-PYTHON COMMUNICATION SERVER")
    print("=" * 60)
    
    print("\nSelect mode:")
    print("  [1] UDP Server (real-time, fast)")
    print("  [2] TCP Server (reliable)")
    print("  [3] File Watcher (easy setup)")
    
    choice = input("\nEnter choice [1/2/3]: ").strip()
    
    if choice == '1':
        server = UDPServer()
        server.start()
    elif choice == '2':
        server = TCPServer()
        server.start()
    else:
        file_watcher_mode()

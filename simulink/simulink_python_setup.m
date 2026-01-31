%% =========================================================================
%  SIMULINK-PYTHON INTEGRATION SETUP
%  Add these blocks to your ieee13bus.slx model for real-time ML detection
%% =========================================================================

%% METHOD 1: Export Data to Python (Easiest)
%% =========================================
% Add these "To Workspace" blocks in your Simulink model:
%
%   Signal              Variable Name    Save Format
%   ----------------    -------------    -----------
%   Voltage             voltage_out      Array
%   Current             current_out      Array  
%   Active Power        power_out        Array
%   Frequency           freq_out         Array
%   EV Demand           ev_demand_out    Array
%
% After simulation, run Python:
%   python python/realtime_ids.py


%% METHOD 2: MATLAB Function Block for Real-Time Detection
%% ========================================================
% Add a MATLAB Function block to your Simulink model with this code:

function [is_attack, attack_type, confidence] = detectAttack(voltage, current, power, frequency, ev_demand)
    % Simple rule-based attack detection (runs inside Simulink)
    
    is_attack = 0;
    attack_type = 0;  % 0=normal, 1=FDI, 2=DoS, 3=Unauthorized
    confidence = 0;
    
    % Normal ranges
    voltage_min = 0.90; voltage_max = 1.10;
    current_max = 150;
    freq_min = 59.5; freq_max = 60.5;
    demand_max = 80;
    
    anomaly_count = 0;
    
    % Check voltage
    if voltage < voltage_min || voltage > voltage_max
        anomaly_count = anomaly_count + 1;
        if voltage < 0.8 || voltage > 1.2
            attack_type = 1;  % FDI
        end
    end
    
    % Check current
    if current > current_max
        anomaly_count = anomaly_count + 1;
        if current > 200
            attack_type = 1;  % FDI
        end
    end
    
    % Check frequency
    if frequency < freq_min || frequency > freq_max
        anomaly_count = anomaly_count + 1;
        if frequency < 58 || frequency > 62
            attack_type = 1;  % FDI
        end
    end
    
    % Check EV demand
    if ev_demand > demand_max
        anomaly_count = anomaly_count + 1;
        if ev_demand > 150
            attack_type = 3;  % Unauthorized
        end
    end
    
    % Determine if attack
    if anomaly_count >= 2
        is_attack = 1;
        confidence = min(anomaly_count / 4, 1.0);
    end
end


%% METHOD 3: Python System Object (Advanced)
%% =========================================
% Create a System Object that calls Python ML model

classdef PythonMLDetector < matlab.System
    % Calls Python ML model for attack detection
    
    properties (Access = private)
        pyModule
        pyDetector
    end
    
    methods (Access = protected)
        function setupImpl(obj)
            % Initialize Python connection
            if count(py.sys.path, '') == 0
                insert(py.sys.path, int32(0), '');
            end
            obj.pyModule = py.importlib.import_module('realtime_ids');
            obj.pyDetector = obj.pyModule.RealTimeIDS();
        end
        
        function [is_attack, confidence] = stepImpl(obj, voltage, current, power, freq, demand)
            % Create data dictionary
            data = py.dict(pyargs(...
                'voltage', voltage, ...
                'current', current, ...
                'active_power', power, ...
                'frequency', freq, ...
                'ev_demand', demand, ...
                'total_load', power + demand ...
            ));
            
            % Call Python detector
            result = obj.pyDetector.detect(data);
            
            is_attack = double(result{'is_attack'});
            confidence = double(result{'confidence'});
        end
    end
end


%% METHOD 4: TCP/UDP Communication (Real-Time Production)
%% ======================================================
% Simulink sends data via UDP, Python receives and responds

% --- SIMULINK SIDE ---
% Add UDP Send block:
%   Remote IP: 127.0.0.1
%   Remote Port: 5000
%   Data: [voltage, current, power, frequency, ev_demand]

% Add UDP Receive block:
%   Local Port: 5001
%   Data: [is_attack, confidence]


%% SETUP INSTRUCTIONS
%% ==================

disp('=========================================================')
disp('SIMULINK-PYTHON INTEGRATION SETUP')
disp('=========================================================')
disp(' ')
disp('STEP 1: Install MATLAB Engine for Python')
disp('   cd "C:\Program Files\MATLAB\R2021a\extern\engines\python"')
disp('   python setup.py install')
disp(' ')
disp('STEP 2: Modify your Simulink model')
disp('   a) Add "To Workspace" blocks for signals')
disp('   b) OR Add MATLAB Function block with detection code')
disp('   c) OR Add UDP Send/Receive blocks')
disp(' ')
disp('STEP 3: Run the integration')
disp('   Option A: Post-simulation analysis')
disp('      1. Run Simulink model')
disp('      2. Export data to CSV')
disp('      3. Run: python python/train_ids.py')
disp(' ')
disp('   Option B: Real-time with MATLAB Engine')
disp('      1. Start MATLAB with: matlab.engine.shareEngine')
disp('      2. Run: python python/realtime_ids.py')
disp('      3. Select mode 2')
disp(' ')
disp('   Option C: Demo without MATLAB')
disp('      1. Run: python python/realtime_ids.py')
disp('      2. Select mode 1')
disp('=========================================================')


%% QUICK TEST: Export current workspace data
%% ==========================================

% If you have already run your Simulink model, use this to export:
disp(' ')
disp('To export your current data:')
disp('   1. Run your Simulink model (ieee13bus.slx)')
disp('   2. Execute these commands:')
disp(' ')
disp('   % Extract last N samples')
disp('   N = 100;')
disp('   time_vec = Iabc9and.time(end-N+1:end);')
disp('   voltage = ones(N,1);  % Assuming per-unit')
disp('   current = abs(Iabc9and.signals(1).values(end-N+1:end));')
disp('   frequency = 60 * ones(N,1);')
disp('   ev_demand = simout(end-N+1:end);')
disp('   ')
disp('   % Calculate power')
disp('   active_power = voltage .* current;')
disp('   reactive_power = active_power * 0.4;')
disp('   total_load = active_power + ev_demand;')
disp('   ')
disp('   % Save to CSV')
disp('   T = table(time_vec, voltage, current, active_power, ...')
disp('             reactive_power, frequency, ev_demand, total_load);')
disp('   writetable(T, ''data/realtime_data.csv'')')
disp('   ')
disp('   % Then run Python')
disp('   % python python/realtime_ids.py')

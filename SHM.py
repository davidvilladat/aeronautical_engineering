import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.svm import OneClassSVM
import logging

# Configure logging for debugging and error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_synthetic_data(num_sensors=4, duration=10, fs=100, anomaly_ratio=0.05):
    """
    Generate synthetic vibration data for multiple sensors.
    
    :param num_sensors: Number of sensors (e.g., placed on wings/fuselage).
    :param duration: Duration of the recording in seconds.
    :param fs: Sampling frequency (Hz).
    :param anomaly_ratio: Proportion of data points that will contain anomalies.
    :return: A dictionary containing time vector and sensor data.
    """
    try:
        t = np.linspace(0, duration, int(duration * fs), endpoint=False)
        # Dictionary to store data for each sensor
        data = {}
        
        for sensor_id in range(num_sensors):
            # Simulate normal data: random noise around a small sinusoidal baseline
            normal_data = 0.1 * np.sin(2 * np.pi * 5 * t) + 0.02 * np.random.randn(len(t))
            
            # Introduce anomalies randomly
            num_anomalies = int(len(t) * anomaly_ratio)
            anomaly_indices = np.random.choice(len(t), size=num_anomalies, replace=False)
            normal_data[anomaly_indices] += 0.5 * np.random.randn(num_anomalies)  # random spikes
            
            data[f"sensor_{sensor_id}"] = normal_data
        
        return {"time": t, "data": data}
    
    except Exception as e:
        logging.error(f"Error generating synthetic data: {e}")
        return None

def low_pass_filter(signal, cutoff=20, fs=100, order=4):
    """
    Apply a low-pass Butterworth filter to the signal.
    
    :param signal: The input signal array.
    :param cutoff: The cutoff frequency (Hz).
    :param fs: Sampling frequency (Hz).
    :param order: Order of the Butterworth filter.
    :return: Filtered signal.
    """
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    except Exception as e:
        logging.error(f"Error filtering signal: {e}")
        return signal  # Return the original signal if filtering fails

def preprocess_data(data_dict):
    """
    Preprocess the collected data by filtering and normalizing.
    
    :param data_dict: Dictionary with time and sensor data.
    :return: Dictionary with the same structure but processed signals.
    """
    try:
        processed_data = {}
        time = data_dict["time"]
        processed_data["time"] = time
        
        new_sensor_data = {}
        for sensor, readings in data_dict["data"].items():
            # Apply a low-pass filter
            filtered = low_pass_filter(readings, cutoff=20, fs=100, order=4)
            
            # Normalize the data
            mean_val = np.mean(filtered)
            std_val = np.std(filtered)
            if std_val == 0:
                normalized = filtered  # Avoid division by zero
            else:
                normalized = (filtered - mean_val) / std_val
            
            new_sensor_data[sensor] = normalized
        
        processed_data["data"] = new_sensor_data
        return processed_data
    
    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}")
        return data_dict  # Return unprocessed data in case of error

def extract_features(data_dict):
    """
    Extract features from the preprocessed data.
    Features: mean, std, and key frequency components via FFT.
    
    :param data_dict: Dictionary with time and sensor data.
    :return: A dictionary of extracted features:
             - 'feature_matrix': NxM matrix (N = # of samples, M = # of features per sample)
             - 'sensors': List of sensor names
             - 'time': Time array
    """
    try:
        time = data_dict["time"]
        sensor_data = data_dict["data"]
        sensors = list(sensor_data.keys())
        
        # We'll stack features across sensors for each point in time.
        # e.g., mean, std, selected FFT magnitude bins for each sensor
        feature_list = []
        
        for i in range(len(time)):
            features_at_t = []
            for sensor in sensors:
                # Single value at time i
                val = sensor_data[sensor][i]
                # Since we're extracting point-by-point features, we combine
                # it with a short window or the entire signal as needed.
                # For simplicity, let's consider the entire signal for FFT-based feature
                # (but in reality, you'd use a window around 'i').
                
                # Basic statistical features (for demonstration):
                mean_val = np.mean(sensor_data[sensor])
                std_val = np.std(sensor_data[sensor])
                
                # FFT-based feature (take magnitude of the first few frequency bins)
                fft_vals = np.fft.rfft(sensor_data[sensor])
                fft_magnitude = np.abs(fft_vals)
                
                # We'll pick the first few bins as features
                fft_feature_bins = fft_magnitude[:5]  # e.g., first 5 freq bins
                
                # Combine features
                sensor_features = [val, mean_val, std_val] + list(fft_feature_bins)
                features_at_t.extend(sensor_features)
            
            feature_list.append(features_at_t)
        
        feature_matrix = np.array(feature_list)
        return {
            "feature_matrix": feature_matrix,
            "sensors": sensors,
            "time": time
        }
    
    except Exception as e:
        logging.error(f"Error in extract_features: {e}")
        return None

def detect_anomalies(feature_dict):
    """
    Detect anomalies using a One-Class SVM.
    
    :param feature_dict: Dictionary containing 'feature_matrix', 'time', and 'sensors'.
    :return: A dictionary with 'time', 'anomaly_scores', and 'anomalies'.
    """
    try:
        feature_matrix = feature_dict["feature_matrix"]
        time = feature_dict["time"]
        
        # Train One-Class SVM on the features
        # For demonstration, we assume entire data set is used for training. 
        # In practical scenarios, you might need a separate baseline "normal" dataset.
        oc_svm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
        oc_svm.fit(feature_matrix)
        
        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = oc_svm.predict(feature_matrix)
        anomaly_scores = oc_svm.decision_function(feature_matrix)
        
        # Identify anomaly points
        anomalies = np.where(predictions == -1)[0]
        
        return {
            "time": time,
            "anomaly_scores": anomaly_scores,
            "anomalies": anomalies
        }
    
    except Exception as e:
        logging.error(f"Error in detect_anomalies: {e}")
        return None

def visualize_data(data_dict, feature_dict, anomaly_dict):
    """
    Generate visualizations for the sensor data, extracted features, and anomalies.
    
    :param data_dict: Original data dictionary with 'time' and 'data'.
    :param feature_dict: Feature dictionary with 'feature_matrix' and 'time'.
    :param anomaly_dict: Output of the anomaly detection with 'anomalies' and 'time'.
    """
    try:
        time = data_dict["time"]
        sensors = list(data_dict["data"].keys())
        
        # 1. Plot raw (or preprocessed) sensor signals
        plt.figure(figsize=(12, 6))
        for sensor in sensors:
            plt.plot(time, data_dict["data"][sensor], label=sensor)
        plt.title("Sensor Readings Over Time (Preprocessed)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 2. Plot anomaly scores
        anomaly_scores = anomaly_dict["anomaly_scores"]
        anomalies = anomaly_dict["anomalies"]
        
        plt.figure(figsize=(12, 4))
        plt.plot(time, anomaly_scores, label="Anomaly Score")
        plt.scatter(time[anomalies], anomaly_scores[anomalies], color='red', label="Anomalies")
        plt.title("Anomaly Scores Over Time (One-Class SVM)")
        plt.xlabel("Time (s)")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        logging.error(f"Error in visualize_data: {e}")

def generate_report(anomaly_dict):
    """
    Generate a summary report of detected anomalies.
    
    :param anomaly_dict: Dictionary with anomaly indices and times.
    :return: A string report summary.
    """
    try:
        anomalies = anomaly_dict["anomalies"]
        time = anomaly_dict["time"]
        
        if len(anomalies) == 0:
            report = "No anomalies detected. Structural integrity is within normal limits."
        else:
            report = (
                f"Detected {len(anomalies)} anomalies at time indices:\n"
                + ", ".join([f"{time[idx]:.2f}s" for idx in anomalies])
                + "\n\nPotential Implications:\n"
                  "• Possible wear or damage in structural components.\n"
                  "• High stress points on wings or fuselage.\n\n"
                  "Recommended Actions:\n"
                  "• Schedule detailed inspection for affected areas.\n"
                  "• Perform necessary maintenance or part replacements.\n"
                  "• Continue monitoring for changes in anomaly frequency or severity.\n"
            )
        return report
    
    except Exception as e:
        logging.error(f"Error in generate_report: {e}")
        return "Error generating report."

def main():
    # Step 1: Data Acquisition
    data = generate_synthetic_data(num_sensors=4, duration=10, fs=100, anomaly_ratio=0.05)
    if data is None:
        logging.error("Data acquisition failed. Exiting.")
        return
    
    # Step 2: Data Processing (Filtering & Normalization)
    processed_data = preprocess_data(data)
    
    # Step 3: Feature Extraction
    features = extract_features(processed_data)
    if features is None:
        logging.error("Feature extraction failed. Exiting.")
        return
    
    # Step 4: Anomaly Detection
    anomalies = detect_anomalies(features)
    if anomalies is None:
        logging.error("Anomaly detection failed. Exiting.")
        return
    
    # Step 5: Visualization
    visualize_data(processed_data, features, anomalies)
    
    # Step 6: Reporting
    report = generate_report(anomalies)
    print("===== SHM Anomaly Report =====")
    print(report)
    print("==============================")

if __name__ == "__main__":
    main()

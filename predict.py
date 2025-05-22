import pandas as pd
import joblib

# Load model
model = joblib.load('model/trained_model.pkl')

# New sample input
sample_data = pd.DataFrame([{
    'network_packet_size': 520,
    'protocol_type': 'TCP',
    'login_attempts': 3,
    'session_duration': 600.5,
    'encryption_used': 'AES',
    'ip_reputation_score': 0.2,
    'failed_logins': 1,
    'browser_type': 'Chrome',
    'unusual_time_access': 0
}])

# Predict
prediction = model.predict(sample_data)
print("Prediction (0 = No Attack, 1 = Attack):", prediction[0])

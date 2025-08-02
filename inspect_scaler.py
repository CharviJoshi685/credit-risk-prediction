import joblib

scaler = joblib.load("scaler.pkl")

# Try to print feature names if available
if hasattr(scaler, 'feature_names_in_'):
    print("Feature names in scaler:")
    print(scaler.feature_names_in_)
else:
    print("Scaler does not have feature_names_in_ attribute.")

# Print scaler attributes for inspection
print("Scaler attributes:")
print(dir(scaler))

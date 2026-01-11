import joblib 


# Load the model
model = joblib.load("random_forest_model.pkl")

# Example prediction (use an appropriate input format based on your model)
sample_data = [[1500]]  # Replace with actual input features
prediction = model.predict(sample_data)

print("Prediction:", prediction)  # Should output 0 or 1 (Legitimate or Fraudulent)


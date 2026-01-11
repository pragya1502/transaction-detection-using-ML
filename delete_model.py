import os

if os.path.exists("random_forest_model.pkl"):
    os.remove("random_forest_model.pkl")
    print("✅ Old model deleted.")
else:
    print("ℹ️ Model file does not exist.")

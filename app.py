from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd  # ✅ Use pandas for DataFrame

# Load the trained model
model = joblib.load("random_forest_model.pkl")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return "<h2>Welcome to the UPI Scam Detection API</h2>"

@app.route("/submit_transaction", methods=["POST"])
def submit_transaction():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Extract required fields
        transaction_type = data.get("type")
        amount = data.get("amount")
        nameOrig = data.get("nameOrig")
        nameDest = data.get("nameDest")
        newbalanceOrig = data.get("newbalanceOrig")

        # Validate presence
        if None in [transaction_type, amount, nameOrig, nameDest, newbalanceOrig]:
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Create DataFrame with exact feature names used during training
        df_input = pd.DataFrame([{
            "type": transaction_type,
            "amount": float(amount),
            "nameOrig": nameOrig,
            "nameDest": nameDest,
            "newbalanceOrig": float(newbalanceOrig)
        }])

        # Predict
        prediction = model.predict(df_input)[0]
        status = "Fraudulent" if prediction == 1 else "Legitimate"

        return jsonify({"status": status})

    except Exception as e:
        print("⚠️ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)

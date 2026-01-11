import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

print("🚦 Script started...")
start_time = time.time()

# Load the dataset
print("📥 Loading dataset...")
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Define sample size
DESIRED_SAMPLE_SIZE = 200000

# Stratified sampling to maintain fraud ratio
print("🔍 Performing stratified sampling...")
fraud_df = df[df["isFraud"] == 1]
nonfraud_df = df[df["isFraud"] == 0].sample(n=DESIRED_SAMPLE_SIZE - len(fraud_df), random_state=42)
df_sampled = pd.concat([fraud_df, nonfraud_df])

# Shuffle the sampled dataset
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Selected features and target
features = ["type", "amount", "nameOrig", "nameDest", "newbalanceOrig"]
target = "isFraud"

X = df_sampled[features]
y = df_sampled[target]

# Define categorical and numerical features
categorical_features = ["type", "nameOrig", "nameDest"]
numerical_features = ["amount", "newbalanceOrig"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Split the dataset
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
print("🚀 Starting model training...")
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "random_forest_model.pkl")

end_time = time.time()
duration = end_time - start_time
print(f"✅ Model trained and saved as 'random_forest_model.pkl'")
print(f"⏱️ Training completed in {duration:.2f} seconds")

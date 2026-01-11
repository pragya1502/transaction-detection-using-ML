import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Ensure the file is in the same directory

# Inspect data
print(df.head())  # Prints the first few rows
print(df.info())  # Provides data type and missing value details

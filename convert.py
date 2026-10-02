import pandas as pd

# Read the Parquet file
df = pd.read_parquet("Obfuscated-MalMem2022.parquet")

print("Dataset Shape:", df.shape)
print(df.head())

# Save as CSV
df.to_csv("Obfuscated-MalMem2022.csv", index=False)

print("✅ CSV created successfully!")
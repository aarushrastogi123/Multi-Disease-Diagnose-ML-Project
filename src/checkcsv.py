import pandas as pd

# Load the CSV (path relative to this file)
df = pd.read_csv('data/train.csv')

# Print first few rows and column names
print(df.head())
print("\nColumns in CSV:", df.columns)

files = ["data/train.csv", "data/val_labels.csv", "data/test.csv"]

for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\n✅ {f} loaded successfully!")
        print(df.head())
    except Exception as e:
        print(f"\n❌ {f} failed: {e}")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reading the data
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Load data into dataframes
df_red = pd.read_csv(url_red, sep=';')
df_white = pd.read_csv(url_white, sep=';')

# Add the `is_white` binary feature such that 0 = Red and 1 = White
df_red['is_white'] = 0
df_white['is_white'] = 1

# Combine the two dataframes into one ignoring the indexing as it's not useful to preserve it
df_wine = pd.concat([df_red, df_white], axis=0, ignore_index=True)

# Form the feature matrix X and label vector y
X = df_wine.drop(columns=['quality'])
y = df_wine['quality'].to_numpy()

# Splitting the dataset into training (60 %), validation (20 %) and test (20 %) sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Normalize the data to zero mean and unit variance
# We only use training data when calculating mu and sigma as we want to prevent data leakage
# These values will be used with training and validation data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Generate EDA statistics
print("--- DATASET SUMMARY ---")
print(f"Total samples (N): {len(df_wine)}")
print(f"Total white wine samples: {len(df_white)}")
print(f"Total red wine samples: {len(df_red)}")
print(f"Feature vector dimension (d): {X.shape[1]}")
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}\n")

print(f"--- FEATURE SUMMARY STATISTICS ---")
print(df_wine.describe().T[['mean', 'std', 'min', 'max']])

# Generate Pearson Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_wine.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Pearson Correlation Matrix - Combined Wine Quality")
plt.tight_layout()
plt.savefig("../assets/correlation_matrix.png", dpi=300)
plt.show()
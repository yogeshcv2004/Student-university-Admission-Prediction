import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset (rename the file if needed)
df = pd.read_csv("admission_data.csv")

# Convert categorical variables
df['hsc_b'] = df['hsc_b'].apply(lambda x: 1 if x == 'Central' else 0)
df['commerce'] = df['hsc_s'].apply(lambda x: 1 if x == 'Commerce' else 0)
df['science'] = df['hsc_s'].apply(lambda x: 1 if x == 'Science' else 0)

# Select input features and target
X = df[['gender', 'hsc_p', 'hsc_b', 'commerce', 'science', 'kcet_score', 'university_rating']]
y = df['admission']

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
with open("admission_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")

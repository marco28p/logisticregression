import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load CSV
df = pd.read_csv("values.csv")

# Clean column names (remove spaces)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Convert date to numeric (optional: ordinal)
df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].map(pd.Timestamp.toordinal)

# Features and target
X = df.drop("score", axis=1)
y = df["score"]

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Optional: predict on new data
def predict_score(new_data):
    """
    new_data should be a list in this order:
    [date, heart_rate, cadence, power, stride_length, vertical_oscillation, gct]
    """
    new_df = pd.DataFrame([new_data], columns=X.columns)

    # Convert date same way
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df["date"] = new_df["date"].map(pd.Timestamp.toordinal)

    new_scaled = scaler.transform(new_df)
    prediction = model.predict(new_scaled)

    return prediction[0]


# Example usage
example = ["3.01.2026", 150, 165, 360, 1.35, 10.3, 250]
print("\nExample prediction:", predict_score(example))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("data/ad_click_data.csv")

# Features and target
X = df[["age", "time_on_site", "pages_viewed", "previous_clicks"]]
y = df["clicked"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Build pipeline (scaling + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate basic accuracy
accuracy = pipeline.score(X_test, y_test)

print("✅ Model trained successfully")
print(f"📊 Test Accuracy: {accuracy:.3f}")

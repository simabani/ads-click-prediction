import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("data/ad_click_data.csv")

X = df[["age", "time_on_site", "pages_viewed", "previous_clicks"]]
y = df["clicked"]

# Train-test split
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "models/ad_click_model.pkl")

print("✅ Model saved to models/ad_click_model.pkl")

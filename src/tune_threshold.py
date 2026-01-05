import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/ad_click_data.csv")

X = df[["age", "time_on_site", "pages_viewed", "previous_clicks"]]
y = df["clicked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Get probabilities
y_probs = pipeline.predict_proba(X_test)[:, 1]

# Try different thresholds
for threshold in [0.3, 0.5, 0.7]:
    y_pred = (y_probs >= threshold).astype(int)
    print(f"\n📊 Threshold = {threshold}")
    print(classification_report(y_test, y_pred))

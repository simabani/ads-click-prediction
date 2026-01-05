import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/ad_click_data.csv")

X = df[["age", "time_on_site", "pages_viewed", "previous_clicks"]]
y = df["clicked"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Classification report
print("\n📊 Classification Report")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Clicked", "Clicked"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

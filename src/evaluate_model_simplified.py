import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# 1️⃣ Load data
df = pd.read_csv("data/ad_click_data.csv")

X = df[["age", "time_on_site", "pages_viewed", "previous_clicks"]]
y = df["clicked"]

# 2️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3️⃣ Build and train model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

# 4️⃣ Predictions
y_pred = pipeline.predict(X_test)

# 5️⃣ Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

labels = np.array([
    [f"True Negative\n(Not Clicked → Not Clicked)\n{tn}",
     f"False Positive\n(Not Clicked → Clicked)\n{fp}"],
    [f"False Negative\n(Clicked → Not Clicked)\n{fn}",
     f"True Positive\n(Clicked → Clicked)\n{tp}"]
])

# 6️⃣ Plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i, j],
                ha="center", va="center", fontsize=11)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted: Not Clicked", "Predicted: Clicked"])
ax.set_yticklabels(["Actual: Not Clicked", "Actual: Clicked"])

ax.set_title("Confusion Matrix (Ad Click Prediction)")
plt.colorbar(im)
plt.tight_layout()
plt.show()

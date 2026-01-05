import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/ad_click_data.csv")

print("\n📊 Dataset Overview")
print(df.head())

print("\n📈 Dataset Info")
print(df.info())

print("\n🎯 Target Distribution (clicked)")
print(df["clicked"].value_counts(normalize=True))

# Plot class balance
sns.countplot(x="clicked", data=df)
plt.title("Clicked vs Not Clicked")
plt.xticks([0, 1], ["Not Clicked", "Clicked"])
plt.show()

# Feature distributions by click
features = ["time_on_site", "pages_viewed", "previous_clicks", "age"]

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="clicked", y=feature, data=df)
    plt.title(f"{feature} vs Clicked")
    plt.xticks([0, 1], ["Not Clicked", "Clicked"])
    plt.show()


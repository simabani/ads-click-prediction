import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 1000

# Generate features
age = np.random.randint(18, 65, size=n_samples)
time_on_site = np.round(np.random.exponential(scale=3, size=n_samples), 2)
pages_viewed = np.random.randint(1, 15, size=n_samples)
previous_clicks = np.random.poisson(lam=1.5, size=n_samples)

# Click probability logic (ground truth)
click_probability = (
    0.05
    + 0.04 * time_on_site
    + 0.03 * pages_viewed
    + 0.15 * previous_clicks
)

click_probability = np.clip(click_probability, 0, 0.95)

clicked = np.random.binomial(1, click_probability)

# Create DataFrame
df = pd.DataFrame({
    "age": age,
    "time_on_site": time_on_site,
    "pages_viewed": pages_viewed,
    "previous_clicks": previous_clicks,
    "clicked": clicked
})

# Save dataset
df.to_csv("data/ad_click_data.csv", index=False)

print("✅ Dataset created: data/ad_click_data.csv")
print(df.head())

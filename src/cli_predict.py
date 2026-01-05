import joblib
import pandas as pd

# Load saved model
pipeline = joblib.load("models/ad_click_model.pkl")

print("\n📢 Ad Click Prediction CLI")
print("Type 'exit' anytime to quit\n")

while True:
    try:
        age = input("Age: ")
        if age.lower() == "exit":
            break

        time_on_site = input("Time on site (minutes): ")
        if time_on_site.lower() == "exit":
            break

        pages_viewed = input("Pages viewed: ")
        if pages_viewed.lower() == "exit":
            break

        previous_clicks = input("Previous ad clicks: ")
        if previous_clicks.lower() == "exit":
            break

        # Create input DataFrame
        user_data = pd.DataFrame({
            "age": [float(age)],
            "time_on_site": [float(time_on_site)],
            "pages_viewed": [int(pages_viewed)],
            "previous_clicks": [int(previous_clicks)]
        })

        # Predict probability
        click_prob = pipeline.predict_proba(user_data)[0][1]

        # Apply threshold (default 0.5)
        prediction = "CLICK" if click_prob >= 0.5 else "NO CLICK"

        print(f"\n🔮 Click Probability: {click_prob:.2f}")
        print(f"📌 Prediction: {prediction}\n")

    except ValueError:
        print("\n⚠️ Invalid input. Please enter numbers.\n")

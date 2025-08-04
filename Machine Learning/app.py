from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset and model
df = pd.read_csv('Beverages.csv')
df = df.dropna()
df['Safe/Unsafe'] = df['Safe/Unsafe'].map({'Safe': 1, 'Unsafe': 0})

# Prepare the model
features = ['Sugar(g/100ml)', 'Caffeine(mg/100ml)']
X = df[features]
y = df['Safe/Unsafe']

clf = RandomForestClassifier(n_estimators=30, random_state=42)
clf.fit(X, y)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    beverage_details = None
    if request.method == "POST":
        drink_input = request.form.get("drink_name").lower()
        result = df[df['Brand'].str.lower().str.contains(drink_input)]

        if not result.empty:
            beverage_details = []
            for index, row in result.iterrows():
                input_data = [[row['Sugar(g/100ml)'], row['Caffeine(mg/100ml)']]]
                prediction = clf.predict(input_data)[0]
                status = "Safe" if prediction == 1 else "Unsafe"
                beverage_details.append({
                    'brand': row['Brand'],
                    'type': row['Type'],
                    'sugar': row['Sugar(g/100ml)'],
                    'caffeine': row['Caffeine(mg/100ml)'],
                    'sweeteners': row['Artificial Sweeteners'],
                    'colourants': row['Artificial Colourants'],
                    'location': row['Manufacturer Location'],
                    'status': status
                })
        else:
            beverage_details = "❌ Drink not found in the database."

    return render_template("index.html", beverage_details=beverage_details)

if __name__ == "__main__":
    app.run(debug=True)

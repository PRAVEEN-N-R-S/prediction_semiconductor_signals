from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

@app.route('/')
def home():
    return render_template("index.html", features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []

        for feature in features:
            value = request.form.get(feature)
            input_data.append(float(value))

        data = np.array(input_data).reshape(1, -1)
        data = scaler.transform(data)

        prediction = model.predict(data)

        result = "PASS ✅" if prediction[0] == -1 else "FAIL ❌"

        return render_template("index.html", features=features, prediction_text=result)

    except Exception as e:
        return render_template("index.html", features=features, prediction_text="Error: " + str(e))


# ✅ MOVE THIS ABOVE app.run
@app.route('/upload', methods=['POST'])
def upload():
    import matplotlib.pyplot as plt
    try:
        file = request.files['file']
        
        if file.filename == '':
            return render_template("index.html", features=features,
                                   prediction_text="No file selected!")

        data = pd.read_csv(file)

        data = data[features]

        data_scaled = scaler.transform(data)

        predictions = model.predict(data_scaled)

        result = ["PASS" if p == -1 else "FAIL" for p in predictions]

        data['Prediction'] = result

        # Count PASS / FAIL
        counts = data['Prediction'].value_counts()

        # Plot graph
        plt.figure()
        counts.plot(kind='bar')
        plt.title("Pass vs Fail Count")
        plt.xlabel("Result")
        plt.ylabel("Count")

        # Save image
        plt.savefig("static/result.png")
        plt.close()

        data.to_csv("output.csv", index=False)

        return render_template("index.html", features=features,
                               prediction_text="✅ File processed!", graph=True)

    except Exception as e:
        return render_template("index.html", features=features,
                               prediction_text="Error: " + str(e))


# ✅ ALWAYS LAST
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
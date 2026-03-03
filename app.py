from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_features = np.array(input_features).reshape(1, -1)

    # 🔥 APPLY SCALING
    final_features = scaler.transform(final_features)

    prediction = model.predict(final_features)

    stages = {
        0: "Normal",
        1: "Stage-1",
        2: "Stage-2",
        3: "Crisis"
    }

    output = stages[prediction[0]]

    return render_template("index.html", prediction_text=f"Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
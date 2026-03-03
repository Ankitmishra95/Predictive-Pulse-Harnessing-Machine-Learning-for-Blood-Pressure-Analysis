from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model, label_encoders = pickle.load(open("hypertension_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []

    for column, value in request.form.items():
        if column in label_encoders:
            encoded_value = label_encoders[column].transform([value])[0]
            input_data.append(encoded_value)
        else:
            input_data.append(value)

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)

    # Convert number back to stage name
    stage_name = label_encoders["Stages"].inverse_transform(prediction)[0]

    return render_template("index.html",
                           prediction_text="Predicted Stage: " + stage_name)

if __name__ == "__main__":
    app.run(debug=True)
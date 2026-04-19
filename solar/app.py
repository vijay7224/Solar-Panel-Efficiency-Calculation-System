from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        ambient = float(request.form["ambient"])
        module = float(request.form["module"])
        irradiation = float(request.form["irradiation"])
        dc_power = float(request.form["dc_power"])

        if irradiation == 0:
            irradiation = 0.01

        prediction = model.predict([[ambient, module, irradiation, dc_power]])
        result = round(prediction[0], 2)


    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
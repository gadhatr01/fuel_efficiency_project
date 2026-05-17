import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("auto-mpg.csv")

# Remove car name column if present
if 'car name' in df.columns:
    df.drop('car name', axis=1, inplace=True)

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Convert all columns to float
df = df.astype(float)

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop("mpg", axis=1)

y = df["mpg"]

# =====================================================
# TRAIN MODEL
# =====================================================

model = LinearRegression()

model.fit(X, y)

# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template("home.html")

# =====================================================
# RESULT PAGE
# =====================================================

@app.route("/result", methods=["POST"])
def result():

    cylinders = float(request.form["cylinders"])

    displacement = float(request.form["displacement"])

    horsepower = float(request.form["horsepower"])

    weight = float(request.form["weight"])

    acceleration = float(request.form["acceleration"])

    model_year = float(request.form["model_year"])

    origin = float(request.form["origin"])

    features = np.array([[
        cylinders,
        displacement,
        horsepower,
        weight,
        acceleration,
        model_year,
        origin
    ]])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)
"""
app.py - Digital Eye Strain Risk Prediction Web App
"""

import pickle
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = "eye_strain_secret_2025"  # required for session

ADMIN_PASSWORD = "admin123"

# Load model once at startup
with open("best_eye_strain_model.pkl", "rb") as f:
    payload = pickle.load(f)

MODEL          = payload["model"]
ENCODERS       = payload["encoders"]
TARGET_ENCODER = payload["target_encoder"]
MODEL_NAME     = payload["model_name"]

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

RECOMMENDATIONS = {
    0: [
        "Great job! Keep maintaining healthy screen habits.",
        "Continue taking regular breaks every 30-45 minutes.",
        "Maintain your current sleep schedule of 7-9 hours.",
    ],
    1: [
        "Follow the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec.",
        "Reduce daily screen time below 6 hours where possible.",
        "Consider enabling blue light filter on all devices.",
        "Increase break frequency to at least once per hour.",
    ],
    2: [
        "Consult an eye specialist if symptoms persist.",
        "Immediately reduce screen time and take frequent breaks.",
        "Use blue light blocking glasses and enable night mode.",
        "Ensure proper lighting — avoid screens in dark rooms.",
        "Aim for 7-9 hours of sleep to allow eye recovery.",
    ],
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            features = [
                float(request.form["age"]),
                int(ENCODERS["Gender"].transform([request.form["gender"]])[0]),
                float(request.form["screen_time"]),
                int(ENCODERS["Primary_Device"].transform([request.form["device"]])[0]),
                int(ENCODERS["Screen_Brightness"].transform([request.form["brightness"]])[0]),
                int(ENCODERS["Break_Frequency"].transform([request.form["break_freq"]])[0]),
                float(request.form["sleep_hours"]),
                int(ENCODERS["Blue_Light_Filter"].transform([request.form["blue_light"]])[0]),
                int(ENCODERS["Eye_Symptom"].transform([request.form["symptom"]])[0]),
            ]
            prediction = MODEL.predict([features])[0]
            risk_label  = RISK_LABELS[prediction]
            tips        = RECOMMENDATIONS[prediction]
            return render_template("result.html",
                                   risk=risk_label,
                                   risk_level=prediction,
                                   tips=tips,
                                   form_data=request.form)
        except (KeyError, ValueError) as e:
            return render_template("predict.html", error=f"Invalid input: {e}")
    return render_template("predict.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prevention")
def prevention():
    return render_template("prevention.html")

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    # Already logged in — go straight to model info
    if session.get("admin_logged_in"):
        return redirect(url_for("model_info"))
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("model_info"))
        error = "Invalid password. Try again."
    return render_template("admin_login.html", error=error)

@app.route("/admin-logout")
def admin_logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/model-info")
def model_info():
    # Guard: redirect to login if not authenticated
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    accuracies = {
        "Decision Tree": "97.50%",
        "Random Forest": "97.50%",
        "KNN":           "75.00%",
        "SVM":           "65.00%",
    }
    return render_template("model_info.html",
                           model_name=MODEL_NAME,
                           accuracies=accuracies)

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)

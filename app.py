"""
app.py
------
Disease Prediction System — Flask Web App (with Login/Register via MongoDB)

train.py যে ৫টা .pkl ফাইল বানিয়েছে, এই app সেগুলো load করে
user-এর দেওয়া symptom লিস্ট থেকে disease predict করে এবং
সেই disease-এর description + precautions দেখায়।

এখন prediction দেখার আগে user-কে অবশ্যই Register/Login করতে হবে।
User-এর data (name, email, phone, height, weight, age, blood group)
এবং prediction history MongoDB-তে save হয়।

Run করার আগে এই ফাইলগুলো app.py-এর same folder-এ থাকতে হবে:
  disease_model.pkl, label_encoder.pkl, symptom_weights.pkl,
  all_symptoms.pkl, disease_info.pkl

এবং MongoDB Atlas (online) ব্যবহার করা হচ্ছে। Connection string env variable
থেকে আসে, কোডে hardcode করা নেই (security-এর জন্য)।

Install করুন:
  pip install pymongo "pymongo[srv]" dnspython python-dotenv

app.py-এর same folder-এ একটা .env ফাইল বানান, ভেতরে এক লাইন:
  MONGO_URI=mongodb+srv://<user>:<pass>@cluster0.xxxxx.mongodb.net/?appName=Cluster0

(.env ফাইলে quote ("") ব্যবহার করবেন না — সরাসরি লিখুন, যেমন উপরে দেখানো হলো।)
"""

import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from translations import symptom_bn, disease_bn

# .env ফাইল থেকে env variable লোড করা হচ্ছে
# (.env ফাইলে MONGO_URI=... লেখা থাকলে এই লাইন সেটা পড়ে env variable বানিয়ে দেয়)
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-to-a-random-secret-key")

# -------------------------------------------------
# MongoDB Atlas কানেকশন
# -------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI environment variable সেট করা নেই। "
        "Atlas connection string কে MONGO_URI নামে env variable হিসেবে সেট করুন "
        "(কোডে সরাসরি password লিখবেন না)।"
    )

client = MongoClient(MONGO_URI, server_api=ServerApi("1"))

# কানেকশন ঠিক আছে কিনা যাচাই (startup-এই ধরা পড়বে যদি ভুল URI/পাসওয়ার্ড/IP allowlist হয়)
try:
    client.admin.command("ping")
    print("✅ MongoDB Atlas-এর সাথে কানেকশন সফল হয়েছে।")
except Exception as e:
    raise RuntimeError(
        f"MongoDB Atlas-এর সাথে কানেক্ট করা যায়নি: {e}\n"
        "চেক করুন: (1) MONGO_URI সঠিক কিনা, (2) Atlas-এর Network Access-এ আপনার IP allow করা আছে কিনা, "
        "(3) ইউজারনেম/পাসওয়ার্ড সঠিক কিনা।"
    )

db = client["disease_prediction_db"]
users_col = db["users"]
history_col = db["predictions"]

# email-এ duplicate registration আটকানোর জন্য unique index
users_col.create_index("email", unique=True)

# -------------------------------------------------
# train.py থেকে তৈরি হওয়া সব ফাইল Load করা
# -------------------------------------------------
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_weights = joblib.load("symptom_weights.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")          # sorted list, training column order
disease_info = joblib.load("disease_info.pkl")          # {disease: {description, precautions}}


# ফর্মে দেখানোর জন্য সুন্দর (readable) নাম -> আসল symptom key ম্যাপ করা
# যেমন "skin_rash" -> "Skin Rash (ত্বকে র‍্যাশ)"
def to_display_name(symptom):
    english = symptom.replace("_", " ").title()
    bangla = symptom_bn(symptom)
    return f"{english} ({bangla})" if bangla else english


symptom_display_map = {s: to_display_name(s) for s in all_symptoms}
# Dropdown-এ readable নাম অনুযায়ী sort করে দেখানো
sorted_symptoms = sorted(all_symptoms, key=lambda s: symptom_display_map[s])


# -------------------------------------------------
# Auth helper
# -------------------------------------------------
def login_required(view_func):
    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


# -------------------------------------------------
# Register
# -------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "name": request.form.get("name", "").strip(),
            "email": request.form.get("email", "").strip().lower(),
            "phone": request.form.get("phone", "").strip(),
            "age": request.form.get("age", "").strip(),
            "height": request.form.get("height", "").strip(),
            "weight": request.form.get("weight", "").strip(),
            "blood_group": request.form.get("blood_group", "").strip(),
        }
        password = request.form.get("password", "")

        # বেসিক validation
        if not all([form_data["name"], form_data["email"], form_data["phone"],
                    form_data["age"], form_data["height"], form_data["weight"],
                    form_data["blood_group"], password]):
            error = "অনুগ্রহ করে সব ফিল্ড পূরণ করুন।"
        elif users_col.find_one({"email": form_data["email"]}):
            error = "এই ইমেইল দিয়ে আগেই একটি অ্যাকাউন্ট আছে। Login করুন।"
        else:
            try:
                age = int(form_data["age"])
                height = float(form_data["height"])
                weight = float(form_data["weight"])
            except ValueError:
                error = "বয়স/উচ্চতা/ওজন সঠিকভাবে দিন।"
                age = height = weight = None

            if error is None:
                user_doc = {
                    "name": form_data["name"],
                    "email": form_data["email"],
                    "phone": form_data["phone"],
                    "age": age,
                    "height": height,
                    "weight": weight,
                    "blood_group": form_data["blood_group"],
                    "password_hash": generate_password_hash(password),
                    "created_at": datetime.utcnow(),
                }
                result = users_col.insert_one(user_doc)

                # রেজিস্ট্রেশনের পরই সরাসরি লগইন করিয়ে দেওয়া
                session["user_id"] = str(result.inserted_id)
                session["user_name"] = user_doc["name"]
                return redirect(url_for("home"))

    return render_template("register.html", error=error, form_data=form_data)


# -------------------------------------------------
# Login
# -------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    email = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = users_col.find_one({"email": email})

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = str(user["_id"])
            session["user_name"] = user["name"]
            return redirect(url_for("home"))
        else:
            error = "ইমেইল বা পাসওয়ার্ড সঠিক নয়।"

    return render_template("login.html", error=error, email=email)


# -------------------------------------------------
# Logout
# -------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------------------------------------
# Prediction history
# -------------------------------------------------
@app.route("/history")
@login_required
def history():
    records = list(
        history_col.find({"user_id": session["user_id"]}).sort("created_at", -1)
    )
    for r in records:
        r["created_at"] = r["created_at"].strftime("%d %B %Y, %I:%M %p")
    return render_template("history.html", history=records, user_name=session.get("user_name"))


# -------------------------------------------------
# Home / Predict (লগইন করা থাকলেই দেখা যাবে)
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    prediction = None
    prediction_bn = ""
    description = ""
    precautions = []
    selected_symptoms = []

    if request.method == "POST":
        # Form থেকে selected symptoms নেওয়া (multi-select <select multiple> থেকে আসবে)
        selected_symptoms = request.form.getlist("symptoms")

        if selected_symptoms:
            # train.py-এর মতো একই column order-এ (all_symptoms) feature vector বানানো
            input_vector = np.zeros(len(all_symptoms))
            for symptom in selected_symptoms:
                if symptom in all_symptoms:
                    idx = all_symptoms.index(symptom)
                    input_vector[idx] = symptom_weights.get(symptom, 1)

            input_vector = input_vector.reshape(1, -1)
            input_df = pd.DataFrame(input_vector, columns=all_symptoms)

            # Prediction করা
            pred_encoded = model.predict(input_df)[0]
            prediction = label_encoder.inverse_transform([pred_encoded])[0]
            prediction_bn = disease_bn(prediction)

            # ওই disease-এর description ও precaution বের করা
            info = disease_info.get(prediction, {})
            description = info.get("description", "")
            precautions = info.get("precautions", [])

            # প্রেডিকশন history-তে save করা
            symptoms_display = ", ".join(symptom_display_map.get(s, s) for s in selected_symptoms)
            history_col.insert_one({
                "user_id": session["user_id"],
                "symptoms": selected_symptoms,
                "symptoms_display": symptoms_display,
                "prediction": prediction,
                "prediction_bn": prediction_bn,
                "created_at": datetime.utcnow(),
            })
        else:
            prediction = "অনুগ্রহ করে অন্তত একটি symptom select করুন।"

    return render_template(
        "index.html",
        symptoms=sorted_symptoms,
        symptom_display_map=symptom_display_map,
        prediction=prediction,
        prediction_bn=prediction_bn,
        description=description,
        precautions=precautions,
        selected_symptoms=selected_symptoms,
        user_name=session.get("user_name"),
    )


if __name__ == "__main__":
    app.run(debug=True)
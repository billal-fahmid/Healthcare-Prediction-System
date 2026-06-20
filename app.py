# """
# app.py
# ------
# Disease Prediction System — Flask Web App

# train.py যে ৫টা .pkl ফাইল বানিয়েছে, এই app সেগুলো load করে
# user-এর দেওয়া symptom লিস্ট থেকে disease predict করে এবং
# সেই disease-এর description + precautions দেখায়।

# Run করার আগে এই ফাইলগুলো app.py-এর same folder-এ থাকতে হবে:
#   disease_model.pkl, label_encoder.pkl, symptom_weights.pkl,
#   all_symptoms.pkl, disease_info.pkl
# """

# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # -------------------------------------------------
# # train.py থেকে তৈরি হওয়া সব ফাইল Load করা
# # -------------------------------------------------
# model = joblib.load("disease_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")
# symptom_weights = joblib.load("symptom_weights.pkl")
# all_symptoms = joblib.load("all_symptoms.pkl")          # sorted list, training column order
# disease_info = joblib.load("disease_info.pkl")          # {disease: {description, precautions}}

# # ফর্মে দেখানোর জন্য সুন্দর (readable) নাম -> আসল symptom key ম্যাপ করা
# # যেমন "skin_rash" -> "Skin Rash"
# def to_display_name(symptom):
#     return symptom.replace("_", " ").title()

# symptom_display_map = {s: to_display_name(s) for s in all_symptoms}
# # Dropdown-এ readable নাম অনুযায়ী sort করে দেখানো
# sorted_symptoms = sorted(all_symptoms, key=lambda s: symptom_display_map[s])


# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = None
#     description = ""
#     precautions = []
#     selected_symptoms = []

#     if request.method == "POST":
#         # Form থেকে selected symptoms নেওয়া (multi-select <select multiple> থেকে আসবে)
#         selected_symptoms = request.form.getlist("symptoms")

#         if selected_symptoms:
#             # train.py-এর মতো একই column order-এ (all_symptoms) feature vector বানানো
#             input_vector = np.zeros(len(all_symptoms))
#             for symptom in selected_symptoms:
#                 if symptom in all_symptoms:
#                     idx = all_symptoms.index(symptom)
#                     input_vector[idx] = symptom_weights.get(symptom, 1)

#             input_vector = input_vector.reshape(1, -1)
#             input_df = pd.DataFrame(input_vector, columns=all_symptoms)

#             # Prediction করা
#             pred_encoded = model.predict(input_df)[0]
#             prediction = label_encoder.inverse_transform([pred_encoded])[0]

#             # ওই disease-এর description ও precaution বের করা
#             info = disease_info.get(prediction, {})
#             description = info.get("description", "")
#             precautions = info.get("precautions", [])
#         else:
#             prediction = "অনুগ্রহ করে অন্তত একটি symptom select করুন।"

#     return render_template(
#         "index.html",
#         symptoms=sorted_symptoms,
#         symptom_display_map=symptom_display_map,
#         prediction=prediction,
#         description=description,
#         precautions=precautions,
#         selected_symptoms=selected_symptoms,
#     )


# if __name__ == "__main__":
#     app.run(debug=True)


"""
app.py
------
Disease Prediction System — Flask Web App

train.py যে ৫টা .pkl ফাইল বানিয়েছে, এই app সেগুলো load করে
user-এর দেওয়া symptom লিস্ট থেকে disease predict করে এবং
সেই disease-এর description + precautions দেখায়।

Run করার আগে এই ফাইলগুলো app.py-এর same folder-এ থাকতে হবে:
  disease_model.pkl, label_encoder.pkl, symptom_weights.pkl,
  all_symptoms.pkl, disease_info.pkl
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from translations import symptom_bn, disease_bn

app = Flask(__name__)

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


@app.route("/", methods=["GET", "POST"])
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
    )


if __name__ == "__main__":
    app.run(debug=True)

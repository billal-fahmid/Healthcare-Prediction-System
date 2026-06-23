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

Master Admin: .env-এ ADMIN_EMAIL ও ADMIN_PASSWORD সেট করলে app চালু হওয়ার
সময় (না থাকলে) সেই account auto-create হবে। Admin সেই একই /login পেজ দিয়েই
লগইন করবে, কিন্তু লগইন করার পর সরাসরি /admin (Admin Dashboard)-এ চলে যাবে,
যেখানে সব user-এর তথ্য ও সবার prediction history দেখা যাবে।

User বা Admin — দুজনেই কোনো prediction result-এর পাশে থাকা "Download PDF"
বাটনে ক্লিক করে সেই রিপোর্টের PDF download করতে পারবে।

Run করার আগে এই ফাইলগুলো app.py-এর same folder-এ থাকতে হবে:
  disease_model.pkl, label_encoder.pkl, symptom_weights.pkl,
  all_symptoms.pkl, disease_info.pkl

এবং MongoDB Atlas (online) ব্যবহার করা হচ্ছে। Connection string env variable
থেকে আসে, কোডে hardcode করা নেই (security-এর জন্য)।

Install করুন:
  pip install pymongo "pymongo[srv]" dnspython python-dotenv reportlab

app.py-এর same folder-এ একটা .env ফাইল বানান, ভেতরে এইরকম লাইনগুলো:
  MONGO_URI=mongodb+srv://<user>:<pass>@cluster0.xxxxx.mongodb.net/?appName=Cluster0
  ADMIN_EMAIL=diseaseAdmin
  ADMIN_PASSWORD=diseaseAdminPass

(.env ফাইলে quote ("") ব্যবহার করবেন না — সরাসরি লিখুন, যেমন উপরে দেখানো হলো।)

লক্ষ্য করুনঃ PDF রিপোর্টে শুধু English টেক্সট দেখানো হয় (disease name,
description, precautions)। Bangla টেক্সট (prediction_bn) PDF-এ অন্তর্ভুক্ত
করা হয়নি, কারণ ডিফল্ট PDF font বাংলা গ্লিফ সাপোর্ট করে না — সেটা দেখাতে
চাইলে একটা বাংলা TTF font (যেমন NotoSansBengali) ফাইলে include করে
reportlab-এ register করতে হবে।
"""

import os
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from translations import symptom_bn, disease_bn
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

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
# Master Admin account — .env-এ ADMIN_EMAIL/ADMIN_PASSWORD দিলে
# app চালু হওয়ার সময় (না থাকলে) auto-create হয়ে যাবে
# -------------------------------------------------
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

if ADMIN_EMAIL and ADMIN_PASSWORD:
    existing_admin = users_col.find_one({"email": ADMIN_EMAIL.strip().lower()})
    if not existing_admin:
        users_col.insert_one({
            "name": "Master Admin",
            "email": ADMIN_EMAIL.strip().lower(),
            "phone": "",
            "age": None,
            "height": None,
            "weight": None,
            "blood_group": "",
            "password_hash": generate_password_hash(ADMIN_PASSWORD),
            "is_admin": True,
            "created_at": datetime.utcnow(),
        })
        print(f"✅ Admin account তৈরি হয়েছে: {ADMIN_EMAIL}")
    elif not existing_admin.get("is_admin"):
        # ইমেইল আগে থেকেই normal user হিসেবে আছে — সেটাকে admin বানিয়ে দেওয়া হলো
        users_col.update_one({"_id": existing_admin["_id"]}, {"$set": {"is_admin": True}})
        print(f"✅ বিদ্যমান অ্যাকাউন্টকে admin করা হয়েছে: {ADMIN_EMAIL}")

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

symptom_index_map = {symptom: idx for idx, symptom in enumerate(all_symptoms)}


def compute_prediction_for_symptoms(selected_symptoms):
    prediction = None
    prediction_bn = ""
    description = ""
    precautions = []

    if selected_symptoms:
        input_vector = np.zeros(len(all_symptoms))
        for symptom in selected_symptoms:
            idx = symptom_index_map.get(symptom)
            if idx is not None:
                input_vector[idx] = symptom_weights.get(symptom, 1)

        input_vector = input_vector.reshape(1, -1)
        input_df = pd.DataFrame(input_vector, columns=all_symptoms)

        pred_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        prediction_bn = disease_bn(prediction)

        info = disease_info.get(prediction, {})
        description = info.get("description", "")
        precautions = info.get("precautions", [])

    return prediction, prediction_bn, description, precautions


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


def admin_required(view_func):
    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        if not session.get("is_admin"):
            return redirect(url_for("home"))
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
            session["is_admin"] = bool(user.get("is_admin", False))

            if session["is_admin"]:
                return redirect(url_for("admin_dashboard"))
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
# Admin Dashboard — সব user ও prediction দেখার জন্য
# -------------------------------------------------
@app.route("/admin")
@admin_required
def admin_dashboard():
    users = list(users_col.find({"is_admin": {"$ne": True}}).sort("created_at", -1))
    total_predictions = history_col.count_documents({})

    for u in users:
        u["id_str"] = str(u["_id"])
        u["prediction_count"] = history_col.count_documents({"user_id": u["id_str"]})
        u["created_at_display"] = u["created_at"].strftime("%d %B %Y") if u.get("created_at") else "—"

    return render_template(
        "admin_dashboard.html",
        users=users,
        total_users=len(users),
        total_predictions=total_predictions,
        admin_name=session.get("user_name"),
    )


@app.route("/admin/user/<user_id>")
@admin_required
def admin_user_detail(user_id):
    user = users_col.find_one({"_id": ObjectId(user_id)})
    if not user:
        return redirect(url_for("admin_dashboard"))

    records = list(history_col.find({"user_id": user_id}).sort("created_at", -1))
    for r in records:
        r["id_str"] = str(r["_id"])
        r["created_at"] = r["created_at"].strftime("%d %B %Y, %I:%M %p")

    user["created_at_display"] = user["created_at"].strftime("%d %B %Y") if user.get("created_at") else "—"

    return render_template(
        "admin_user_detail.html",
        user=user,
        history=records,
        admin_name=session.get("user_name"),
    )


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
        r["id_str"] = str(r["_id"])
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

            # প্রেডিকশন history-তে save করা (description/precautions-ও রাখা হলো, যাতে PDF-এ ব্যবহার করা যায়)
            symptoms_display = ", ".join(symptom_display_map.get(s, s) for s in selected_symptoms)
            inserted = history_col.insert_one({
                "user_id": session["user_id"],
                "symptoms": selected_symptoms,
                "symptoms_display": symptoms_display,
                "prediction": prediction,
                "prediction_bn": prediction_bn,
                "description": description,
                "precautions": precautions,
                "created_at": datetime.utcnow(),
            })
            prediction_id = str(inserted.inserted_id)
        else:
            prediction = "অনুগ্রহ করে অন্তত একটি symptom select করুন।"
            prediction_id = None
    else:
        prediction_id = None

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
        prediction_id=prediction_id,
    )


# -------------------------------------------------
# AJAX prediction endpoint for realtime symptom updates
# -------------------------------------------------
@app.route("/predict-json", methods=["POST"])
@login_required
def predict_json():
    payload = request.get_json(silent=True) or {}
    selected_symptoms = payload.get("symptoms") if isinstance(payload.get("symptoms"), list) else []

    prediction, prediction_bn, description, precautions = compute_prediction_for_symptoms(selected_symptoms)
    response = {
        "prediction": prediction,
        "prediction_bn": prediction_bn,
        "description": description,
        "precautions": precautions,
    }
    flask_response = jsonify(response)
    flask_response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return flask_response


# -------------------------------------------------
# Prediction PDF Download
# -------------------------------------------------
@app.route("/download-pdf/<prediction_id>")
@login_required
def download_pdf(prediction_id):
    record = history_col.find_one({"_id": ObjectId(prediction_id)})
    if not record:
        return redirect(url_for("history"))

    # নিজের prediction অথবা admin হলেই PDF download করতে পারবে — অন্যজনের রিপোর্ট নয়
    is_owner = record.get("user_id") == session.get("user_id")
    if not (is_owner or session.get("is_admin")):
        return redirect(url_for("home"))

    patient = users_col.find_one({"_id": ObjectId(record["user_id"])})
    patient_name = patient.get("name", "—") if patient else "—"
    patient_email = patient.get("email", "—") if patient else "—"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=28 * mm, bottomMargin=28 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
    )

    def draw_page(canvas, doc):
        width, height = A4
        canvas.saveState()

        # Logo badge in the top-left corner
        logo_width = 40
        logo_height = 20
        logo_x = doc.leftMargin
        logo_y = height - doc.topMargin + 8
        canvas.setFillColor(colors.HexColor("#1f4aa8"))
        canvas.roundRect(logo_x, logo_y - logo_height, logo_width, logo_height, 5, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(logo_x + 8, logo_y - 14, "HPS")
        canvas.setFont("Helvetica", 6)
        canvas.drawString(logo_x + logo_width + 8, logo_y - 14, "Healthcare Prediction")

        # Watermark in the center
        canvas.setFillColor(colors.HexColor("#d1d5db"))
        canvas.setFont("Helvetica-Bold", 48)
        canvas.translate(width / 2, height / 2)
        canvas.rotate(45)
        canvas.drawCentredString(0, 0, "CONFIDENTIAL")
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawCentredString(0, -40, "FOR PRIVATE USE ONLY")

        canvas.restoreState()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=18,
        textColor=colors.HexColor("#16233f"), spaceAfter=4, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#6b7280"), spaceAfter=18, alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#1f4aa8"), spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"], fontSize=10.5, leading=15,
    )
    disease_style = ParagraphStyle(
        "DiseaseStyle", parent=styles["Heading1"], fontSize=16,
        textColor=colors.HexColor("#1e8449"), spaceAfter=4,
    )
    warn_style = ParagraphStyle(
        "WarnStyle", parent=styles["Normal"], fontSize=9, leading=13,
        textColor=colors.HexColor("#92400e"),
    )

    elements = []
    elements.append(Paragraph("Disease Prediction Report", title_style))
    elements.append(Paragraph(
        "Generated by Disease Prediction System — for preliminary reference only",
        subtitle_style,
    ))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e6ed"), thickness=1))
    elements.append(Spacer(1, 12))

    # Patient info table
    info_data = [
        ["Patient Name", patient_name],
        ["Email", patient_email],
        ["Report Date", record["created_at"].strftime("%d %B %Y, %I:%M %p")],
    ]
    info_table = Table(info_data, colWidths=[120, 360])
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b7280")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#eef1f6")),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 10))

    # Predicted disease
    elements.append(Paragraph("Predicted Disease", heading_style))
    elements.append(Paragraph(record.get("prediction", "—"), disease_style))

    # Symptoms
    elements.append(Paragraph("Reported Symptoms", heading_style))
    symptoms_text = record.get("symptoms_display", "—")
    elements.append(Paragraph(symptoms_text, body_style))

    # Description
    if record.get("description"):
        elements.append(Paragraph("Description", heading_style))
        elements.append(Paragraph(record["description"], body_style))

    # Precautions
    if record.get("precautions"):
        elements.append(Paragraph("Precautions", heading_style))
        for p in record["precautions"]:
            elements.append(Paragraph(f"&ndash; {p}", body_style))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e6ed"), thickness=1))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "This is only a preliminary indication. Please consult a qualified doctor for "
        "an accurate diagnosis and treatment.",
        warn_style,
    ))

    doc.build(elements, onFirstPage=draw_page, onLaterPages=draw_page)
    buffer.seek(0)

    filename = f"disease_prediction_{record['created_at'].strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
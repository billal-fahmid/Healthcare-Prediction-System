"""
train.py
--------
Disease Prediction System — Model Training Script

এই স্ক্রিপ্ট ৪টি dataset ব্যবহার করে:
  1. dataset.csv              -> Disease + Symptom_1..Symptom_17 (training data)
  2. Symptom-severity.csv     -> প্রতিটি Symptom-এর severity weight
  3. symptom_Description.csv  -> প্রতিটি Disease-এর description (app.py-তে দেখানোর জন্য)
  4. symptom_precaution.csv   -> প্রতিটি Disease-এর precautions (app.py-তে দেখানোর জন্য)

Approach:
  প্রতিটি symptom-কে তার severity weight দিয়ে replace করা হয় (categorical -> numeric),
  তারপর RandomForestClassifier দিয়ে Disease predict করার model train করা হয়।

Output files (pickle):
  disease_model.pkl       -> trained classifier
  label_encoder.pkl       -> Disease নামের LabelEncoder
  symptom_weights.pkl     -> {symptom_name: weight} dict
  all_symptoms.pkl        -> সব unique symptom-এর sorted list (app.py form-এ dropdown বানাতে)
  disease_info.pkl        -> {disease: {"description": ..., "precautions": [...]}} dict
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# 1) Helper: symptom string normalize করা
#    (CSV-তে symptom নামে inconsistent space/underscore আছে,
#     যেমন "dischromic _patches" vs "dischromic_patches")
# -------------------------------------------------
def normalize_symptom(s):
    if pd.isna(s):
        return None
    return str(s).strip().lower().replace(" ", "").replace("__", "_")


# -------------------------------------------------
# 2) Dataset গুলো Load করা
# -------------------------------------------------
df = pd.read_csv("dataset.csv")
severity_df = pd.read_csv("Symptom-severity.csv")
description_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

print("dataset.csv shape:", df.shape)
print("Total diseases:", df["Disease"].nunique())

# Disease column-এর extra space clean করা (যেমন "Diabetes " -> "Diabetes")
df["Disease"] = df["Disease"].str.strip()

symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]

# -------------------------------------------------
# 3) Symptom -> severity weight dictionary বানানো
# -------------------------------------------------
severity_df["Symptom"] = severity_df["Symptom"].apply(normalize_symptom)
symptom_weights = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# সব Symptom column normalize করে নেওয়া
for col in symptom_cols:
    df[col] = df[col].apply(normalize_symptom)

# পুরো dataset-এ থাকা সব unique symptom (dropdown-এর জন্য কাজে লাগবে app.py-তে)
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())
all_symptoms = sorted(all_symptoms)
print("Total unique symptoms:", len(all_symptoms))

# severity file-এ নেই এমন symptom থাকলে ডিফল্ট weight ১ ধরে নেওয়া হলো
for s in all_symptoms:
    if s not in symptom_weights:
        symptom_weights[s] = 1
        print(f"Warning: '{s}' not found in Symptom-severity.csv, default weight=1 set")

# -------------------------------------------------
# 4) Feature Engineering
#    প্রতিটি symptom column-কে তার severity weight দিয়ে replace করা,
#    আর প্রতিটি unique symptom-এর জন্য একটি column বানানো (multi-hot + weight)
# -------------------------------------------------
X = pd.DataFrame(0, index=df.index, columns=all_symptoms, dtype=float)

for col in symptom_cols:
    for idx, symptom in df[col].items():
        if symptom is not None and symptom in symptom_weights:
            X.at[idx, symptom] = symptom_weights[symptom]

y_raw = df["Disease"]

# -------------------------------------------------
# 5) Label Encoding (Disease নাম -> সংখ্যা)
# -------------------------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# -------------------------------------------------
# 6) Train / Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# 7) Model Train করা
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None,
)
model.fit(X_train, y_train)

# -------------------------------------------------
# 8) Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# -------------------------------------------------
# 9) Description + Precaution লুকআপ ডিকশনারি বানানো
#    (app.py পরে prediction-এর সাথে এগুলো দেখাতে পারবে)
# -------------------------------------------------
description_df["Disease"] = description_df["Disease"].str.strip()
precaution_df["Disease"] = precaution_df["Disease"].str.strip()

disease_info = {}
for disease in label_encoder.classes_:
    desc_row = description_df[description_df["Disease"] == disease]
    prec_row = precaution_df[precaution_df["Disease"] == disease]

    description = desc_row["Description"].values[0] if len(desc_row) else ""
    precaution_cols = [c for c in precaution_df.columns if c.startswith("Precaution_")]
    precautions = []
    if len(prec_row):
        precautions = [
            p for p in prec_row[precaution_cols].values[0].tolist()
            if isinstance(p, str) and p.strip() != ""
        ]

    disease_info[disease] = {
        "description": description,
        "precautions": precautions,
    }

# -------------------------------------------------
# 10) সব কিছু .pkl ফাইলে Save করা
# -------------------------------------------------
joblib.dump(model, "disease_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(symptom_weights, "symptom_weights.pkl")
joblib.dump(all_symptoms, "all_symptoms.pkl")
joblib.dump(disease_info, "disease_info.pkl")

print("\nSaved files:")
print(" - disease_model.pkl")
print(" - label_encoder.pkl")
print(" - symptom_weights.pkl")
print(" - all_symptoms.pkl")
print(" - disease_info.pkl")
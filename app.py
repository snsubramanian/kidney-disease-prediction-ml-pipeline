
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Kidney Disease Classification")
st.write("Machine Learning Prediction Website")

st.warning(
    "Educational use only. This model is not clinically validated "
    "and must not be used for medical diagnosis or treatment decisions."
)

# --------------------------------
# Load trained model
# --------------------------------
MODEL_PATH = Path(__file__).parent / "kidney_disease_best_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# --------------------------------
# Patient input form
# --------------------------------
st.subheader("Enter Patient Details")

with st.form("patient_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=48)
        bp = st.number_input("Blood Pressure (bp)", min_value=0, value=80)
        sg = st.selectbox("Specific Gravity (sg)", [
            1.005, 1.010, 1.015, 1.020, 1.025
        ])
        al = st.selectbox("Albumin (al)", [0, 1, 2, 3, 4, 5])
        su = st.selectbox("Sugar (su)", [0, 1, 2, 3, 4, 5])

    with col2:
        rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps (pcc)", ["notpresent", "present"])
        ba = st.selectbox("Bacteria (ba)", ["notpresent", "present"])
        bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0, value=121)
        bu = st.number_input("Blood Urea (bu)", min_value=0.0, value=36.0)
        sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, value=1.2)

    with col3:
        sod = st.number_input("Sodium (sod)", min_value=0.0, value=138.0)
        pot = st.number_input("Potassium (pot)", min_value=0.0, value=4.5)
        hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, value=13.0)
        pcv = st.number_input("Packed Cell Volume (pcv)", min_value=0, value=40)
        wc = st.number_input("White Blood Cell Count (wc)", min_value=0, value=8000)
        rc = st.number_input("Red Blood Cell Count (rc)", min_value=0.0, value=4.5)

    col4, col5 = st.columns(2)

    with col4:
        htn = st.selectbox("Hypertension (htn)", ["yes", "no"])
        dm = st.selectbox("Diabetes Mellitus (dm)", ["yes", "no"])
        cad = st.selectbox("Coronary Artery Disease (cad)", ["yes", "no"])

    with col5:
        appet = st.selectbox("Appetite (appet)", ["good", "poor"])
        pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"])
        ane = st.selectbox("Anemia (ane)", ["yes", "no"])

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# --------------------------------
# Prediction
# --------------------------------
if submitted:

    patient = pd.DataFrame([{
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "su": su,
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "wc": wc,
        "rc": rc,
        "rbc": rbc,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane
    }])

    try:
        prediction = model.predict(patient)[0]

        st.subheader("Prediction Result")

        if str(prediction).strip().lower() == "ckd":
            st.error("Model prediction: CKD")
        elif str(prediction).strip().lower() == "notckd":
            st.success("Model prediction: Not CKD")
        else:
            st.info(f"Model prediction: {prediction}")

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(patient)[0]

                st.subheader("Model Probability Estimates")

                probability_df = pd.DataFrame({
                    "Class": model.classes_,
                    "Estimated probability": probabilities
                })

                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True
                )
            except Exception:
                st.info("Probability estimates are unavailable.")

        st.caption(
            "This prediction is not a medical diagnosis. Consult a qualified "
            "healthcare professional for medical interpretation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(
            "Check that the input column names, order, data types, and "
            "categories match the features used to train your saved pipeline."
        )

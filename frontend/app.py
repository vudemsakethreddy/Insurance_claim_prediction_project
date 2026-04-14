import streamlit as st
import requests
from PIL import Image
import io

BACKEND_URL = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")

st.title("Insurance Claim Prediction (Streamlit + FastAPI)")

st.subheader("1) Train Model")
if st.button("Train Now"):
    with st.spinner("Training..."):
        r = requests.post(f"{BACKEND_URL}/train")
        if r.status_code == 200:
            st.success("Training complete!")
            st.json(r.json())
        else:
            st.error(r.text)

st.divider()

st.subheader("2) Predict")

col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Age", "25")
    gender = st.text_input("Gender (male/female or 0/1)", "male")
    bmi = st.text_input("BMI", "27.5")
    children = st.text_input("Children", "0")

with col2:
    smoker = st.text_input("Smoker (yes/no or 0/1)", "no")
    region = st.text_input("Region (e.g. southwest)", "southwest")
    charges = st.text_input("Charges", "3000")

if st.button("Predict"):
    payload = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges,
    }
    r = requests.post(f"{BACKEND_URL}/predict", json=payload)
    if r.status_code == 200:
        st.success("Prediction result:")
        st.json(r.json())
    else:
        st.error(r.text)

st.divider()

st.subheader("3) Visualization (.png) from Backend")

st.caption("These images are served by FastAPI from /plots and saved in static/plots/*.png")

plot_files = [
    "01_target_distribution.png",
    "02_hist_age.png",
    "02_hist_bmi.png",
    "02_hist_charges.png",
    "05_corr_heatmap.png",
    "08_confusion_matrix_best.png",
    "09_roc_curve_best.png",
]

cols = st.columns(2)
i = 0
for f in plot_files:
    url = f"{BACKEND_URL}/plots/{f}"
    try:
        img_res = requests.get(url, timeout=3)
        if img_res.status_code == 200 and img_res.content:
            img = Image.open(io.BytesIO(img_res.content))
            with cols[i % 2]:
                st.image(img, caption=f, use_container_width=True)
            i += 1
    except Exception:
        pass
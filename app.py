import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Smart Factory AI Monitoring", layout="wide")

st.title("🏭 Smart Factory AI Monitoring System")

model = YOLO("models/best.pt")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)

    results = model(image)

    annotated = results[0].plot()

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated, caption="Detection Result", use_container_width=True)

    counter = Counter()

    for box in results[0].boxes:

        counter[model.names[int(box.cls)]] += 1

    workers = counter["Person"]
    helmets = counter["Hardhat"]
    vests = counter["Safety Vest"]
    vehicles = counter["vehicle"]
    machinery = counter["machinery"]

    helmet_violation = max(0, workers - helmets)
    vest_violation = max(0, workers - vests)

    col1, col2, col3 = st.columns(3)

    col1.metric("Workers", workers)
    col2.metric("Helmets", helmets)
    col3.metric("Safety Vest", vests)

    col1.metric("Vehicles", vehicles)
    col2.metric("Machinery", machinery)
    col3.metric("Violations", helmet_violation + vest_violation)

    df = pd.DataFrame({

        "Category":[
            "Workers",
            "Helmets",
            "Safety Vest",
            "Vehicles",
            "Machinery",
            "Helmet Violation",
            "Vest Violation"
        ],

        "Count":[
            workers,
            helmets,
            vests,
            vehicles,
            machinery,
            helmet_violation,
            vest_violation
        ]

    })

    st.bar_chart(df.set_index("Category"))
    st.dataframe(df)
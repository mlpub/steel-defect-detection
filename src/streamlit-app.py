import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import json
import requests


API_URL = "https://steel-defect-app-901723644506.asia-southeast1.run.app/predict"

st.set_page_config(page_title="Steel Defect Detection")

st.title("Steel Defect Detection Demo")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="content")

    if st.button("Predict"):
        # convert image to base64
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        #print(img_str)  # For debugging

        headers = {"Content-Type": "application/json"}
        data = json.dumps({"image": img_str})

        response = requests.post(API_URL, headers=headers, data=data)
        #print("Status code:", response.status_code)
        #print("Raw response:", response.text)
        json_result = json.loads(response.text)
        defect_prob = json_result.get("defect_prob")
        defect_status = json_result.get("defect_status")

        result = f"Defect Probability: {defect_prob:.4f}\nDefect Status: {defect_status}"
        if (defect_status == 0):
            result = result + "\nNo Defect ✅"
        else:
            result = result + "\nDefect ❌"
        st.text(result)

        #st.success(response.text)


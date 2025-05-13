import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests

st.title("ID Document Scanner")
st.write("This is a simplified version to test dependencies")

# Test camera input
st.subheader("Camera Test")
camera_input = st.camera_input("Take a picture")

if camera_input:
    st.success("Camera working correctly!")
    
    # Test image processing
    image = Image.open(camera_input)
    st.image(image, caption="Captured Image")
    
    # Convert to OpenCV format
    import io
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Test grayscale conversion (simple OpenCV test)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image")
    
    # Test Tesseract OCR
    try:
        text = pytesseract.image_to_string(gray)
        st.subheader("Extracted Text")
        st.text(text)
    except Exception as e:
        st.error(f"OCR Error: {e}")

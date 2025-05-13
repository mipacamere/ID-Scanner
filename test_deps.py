import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests

st.title("Dependencies Test")
st.write("If you can see this, all dependencies are correctly installed!")

# Test OpenCV
st.write("OpenCV version:", cv2.__version__)

# Test pytesseract
try:
    pytesseract.get_tesseract_version()
    st.write("Tesseract is properly installed!")
except Exception as e:
    st.error(f"Tesseract error: {e}")

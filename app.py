import streamlit as st
import io
import cv2
import numpy as np
import pytesseract
import requests
import json
import re
from PIL import Image
import base64
import os
from datetime import datetime

# App title and description
st.set_page_config(page_title="ID Document Scanner", layout="wide")
st.title("ID Document Scanner")
st.write("Scan ID documents, extract information, and submit to API")

# Initialize session state for storing data across reruns
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}
if 'selected_doc' not in st.session_state:
    st.session_state.selected_doc = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'first_name': '',
        'last_name': '',
        'date_of_birth': '',
        'place_of_birth': '',
        'id_number': ''
    }
if 'api_url' not in st.session_state:
    st.session_state.api_url = ''
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'headers' not in st.session_state:
    st.session_state.headers = '{\n    "Content-Type": "application/json"\n}'
if 'tabs' not in st.session_state:
    st.session_state.tabs = 0

# Create tabs
tab1, tab2, tab3 = st.tabs(["Scan & Upload", "Results & Form", "API Settings"])

# Function to preprocess image for better OCR
def preprocess_image(image):
    # Convert to grayscale if image has colors
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Dilation to make text clearer
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(opening, kernel, iterations=1)
    
    return processed

# Functions to extract information from OCR text
def extract_name(text):
    patterns = [
        r'Name:?\s*([A-Za-z]+)',
        r'First Name:?\s*([A-Za-z]+)',
        r'Given Name:?\s*([A-Za-z]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return ""

def extract_surname(text):
    patterns = [
        r'Surname:?\s*([A-Za-z]+)',
        r'Last Name:?\s*([A-Za-z]+)',
        r'Family Name:?\s*([A-Za-z]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return ""

def extract_dob(text):
    patterns = [
        r'Date of Birth:?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
        r'DOB:?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
        r'Born:?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return ""

def extract_pob(text):
    patterns = [
        r'Place of Birth:?\s*([A-Za-z\s]+)',
        r'POB:?\s*([A-Za-z\s]+)',
        r'Born in:?\s*([A-Za-z\s]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_id_number(text):
    patterns = [
        r'ID:?\s*([A-Za-z0-9]+)',
        r'ID Number:?\s*([A-Za-z0-9]+)',
        r'Document No:?\s*([A-Za-z0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return ""

def extract_information(image):
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(image)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(pil_img)
    
    # Parse extracted text for relevant information
    data = {
        'first_name': extract_name(text),
        'last_name': extract_surname(text),
        'date_of_birth': extract_dob(text),
        'place_of_birth': extract_pob(text),
        'id_number': extract_id_number(text)
    }
    
    return data, text

# Function to process the uploaded documents
def process_documents():
    with st.spinner('Processing documents...'):
        st.session_state.processed_images = []
        st.session_state.extracted_data = {}
        
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(st.session_state.image_paths):
            try:
                # Update progress
                progress = (i + 1) / len(st.session_state.image_paths)
                progress_bar.progress(progress)
                
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                uploaded_file.seek(0)  # Reset file pointer
                
                # Process image
                processed = preprocess_image(image)
                st.session_state.processed_images.append(processed)
                
                # Extract information
                data, raw_text = extract_information(processed)
                
                # Store extracted data
                doc_name = uploaded_file.name
                st.session_state.extracted_data[doc_name] = {
                    'data': data,
                    'raw_text': raw_text,
                    'image': image,
                    'processed': processed
                }
                
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
        
        if st.session_state.extracted_data:
            # Select first document by default
            st.session_state.selected_doc = list(st.session_state.extracted_data.keys())[0]
            
            # Update form with first document data
            doc_data = st.session_state.extracted_data[st.session_state.selected_doc]['data']
            for field, value in doc_data.items():
                st.session_state.form_data[field] = value
                
            # Switch to results tab
            st.session_state.tabs = 1
            st.rerun()
        else:
            st.error("No information could be extracted from the documents")

# Function to submit data to API
def submit_data():
    with st.spinner('Submitting data to API...'):
        try:
            # Parse headers
            headers = {}
            try:
                if st.session_state.headers.strip():
                    headers = json.loads(st.session_state.headers)
            except json.JSONDecodeError:
                st.warning("Invalid JSON format for headers. Using default headers.")
                headers = {"Content-Type": "application/json"}
            
            # Add API key if provided
            if st.session_state.api_key:
                headers["Authorization"] = f"Bearer {st.session_state.api_key}"
            
            # Make API request
            response = requests.post(
                st.session_state.api_url,
                json=st.session_state.form_data,
                headers=headers
            )
            
            # Handle response
            if response.status_code in [200, 201]:
                st.success("Data successfully submitted to API!")
            else:
                st.error(f"API Error: {response.status_code}\n{response.text}")
        
        except requests.RequestException as e:
            st.error(f"Failed to connect to API: {str(e)}")

# Function to test API connection
def test_api_connection():
    if not st.session_state.api_url:
        st.warning("Please enter an API endpoint URL")
        return
    
    with st.spinner('Testing API connection...'):
        try:
            # Parse headers
            headers = {}
            try:
                if st.session_state.headers.strip():
                    headers = json.loads(st.session_state.headers)
            except json.JSONDecodeError:
                st.warning("Invalid JSON format for headers. Using default headers.")
                headers = {"Content-Type": "application/json"}
            
            # Add API key if provided
            if st.session_state.api_key:
                headers["Authorization"] = f"Bearer {st.session_state.api_key}"
            
            # Make a simple GET request to test connection
            response = requests.get(
                st.session_state.api_url,
                headers=headers,
                timeout=5
            )
            
            st.success(f"Connection successful! Status code: {response.status_code}")
            
        except requests.RequestException as e:
            st.error(f"Connection failed: {str(e)}")

# Tab 1: Scan & Upload
with tab1:
    st.header("Upload or Take Photos of ID Documents")
    
    # File upload
    uploaded_files = st.file_uploader("Upload ID Documents", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.image_paths = uploaded_files
    
    # Camera capture
    st.subheader("Or Capture from Camera")
    camera_col1, camera_col2 = st.columns(2)
    
    with camera_col1:
        camera_image = st.camera_input("Take a picture of your ID")
        
        if camera_image:
            # Add camera image to image_paths
            if camera_image not in st.session_state.image_paths:
                st.session_state.image_paths.append(camera_image)
    
    # Show list of uploaded documents
    if st.session_state.image_paths:
        st.subheader("Uploaded Documents")
        for i, file in enumerate(st.session_state.image_paths):
            st.write(f"{i+1}. {file.name}")
        
        # Process button
        if st.button("Process Documents", key="process_btn"):
            process_documents()

# Tab 2: Results & Form
with tab2:
    st.header("Extracted Information")
    
    if st.session_state.extracted_data:
        # Document selection
        doc_options = list(st.session_state.extracted_data.keys())
        selected_doc = st.selectbox("Select Document", doc_options, 
                                 index=doc_options.index(st.session_state.selected_doc) if st.session_state.selected_doc in doc_options else 0)
        
        if selected_doc != st.session_state.selected_doc:
            st.session_state.selected_doc = selected_doc
            # Update form with selected document data
            doc_data = st.session_state.extracted_data[selected_doc]['data']
            for field, value in doc_data.items():
                st.session_state.form_data[field] = value
            st.rerun()
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            original = st.session_state.extracted_data[selected_doc]['image']
            st.image(original, channels="BGR", use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            processed = st.session_state.extracted_data[selected_doc]['processed']
            st.image(processed, use_column_width=True)
        
        # Show raw OCR text
        with st.expander("Show Raw OCR Text"):
            st.text(st.session_state.extracted_data[selected_doc]['raw_text'])
        
        # Form for extracted data
        st.subheader("Extracted Information (Edit if needed)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.form_data['first_name'] = st.text_input("First Name", 
                                             value=st.session_state.form_data['first_name'])
            st.session_state.form_data['last_name'] = st.text_input("Last Name", 
                                            value=st.session_state.form_data['last_name'])
            st.session_state.form_data['date_of_birth'] = st.text_input("Date of Birth", 
                                                value=st.session_state.form_data['date_of_birth'])
        
        with col2:
            st.session_state.form_data['place_of_birth'] = st.text_input("Place of Birth", 
                                                value=st.session_state.form_data['place_of_birth'])
            st.session_state.form_data['id_number'] = st.text_input("ID Number", 
                                            value=st.session_state.form_data['id_number'])
        
        # Submit button
        if st.button("Submit Data to API"):
            if not st.session_state.api_url:
                st.warning("Please configure API endpoint in the API Settings tab")
                st.session_state.tabs = 2  # Switch to API tab
                st.rerun()
            else:
                submit_data()
    else:
        st.info("No documents processed yet. Please go to the Scan & Upload tab to upload and process documents.")

# Tab 3: API Settings
with tab3:
    st.header("API Configuration")
    
    st.session_state.api_url = st.text_input("API Endpoint URL", value=st.session_state.api_url)
    st.session_state.api_key = st.text_input("API Key (if required)", value=st.session_state.api_key, type="password")
    
    st.subheader("Additional Headers (JSON format)")
    st.session_state.headers = st.text_area("Headers", value=st.session_state.headers, height=150)
    
    if st.button("Test API Connection"):
        test_api_connection()
    
    # Help information
    with st.expander("API Help"):
        st.markdown("""
        ### API Information
        
        This application submits the extracted ID information to your specified API endpoint.
        
        - Make sure your API accepts POST requests with JSON data
        - The data will be sent in the following format:
        
        ```json
        {
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "01/01/1990",
            "place_of_birth": "New York",
            "id_number": "ABC123456"
        }
        ```
        
        - If your API requires different field names, you can modify them in the code
        """)

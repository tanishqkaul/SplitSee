import os
import io
import cv2
import numpy as np
import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image

# Ensure the GOOGLE_APPLICATION_CREDENTIALS secret is set
google_credentials = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

if google_credentials is None:
    st.error("The GOOGLE_APPLICATION_CREDENTIALS secret is not set.")
else:
    # Set up credentials using the service account info from st.secrets
    credentials = service_account.Credentials.from_service_account_info(google_credentials)

# Set up Google Cloud Vision client
client = vision.ImageAnnotatorClient(credentials=credentials)

def analyze_image(uploaded_image):
    """Analyze the uploaded image using Google Vision API to detect texts.
    
    Args:
        uploaded_image: The uploaded image file.
    
    Returns:
        texts: The detected texts from the image.
    """
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to bytes
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='PNG')
    content = image_byte_array.getvalue()
    
    # Use Vision API to analyze the image
    vision_image = vision.Image(content=content)
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations
    
    return texts

def extract_texts(texts):
    """Extract and display detected texts from the image.
    
    Args:
        texts: The detected texts from the image.
    """
    for text in texts:
        st.write(f"Description: {text.description}")
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        st.write(f"Vertices: {vertices}")

def segment_visual_elements(uploaded_image):
    """Segment and display visual elements from the uploaded image.
    
    Args:
        uploaded_image: The uploaded image file.
    """
    image = Image.open(uploaded_image)
    open_cv_image = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(open_cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    st.image(open_cv_image, caption='Segmented Image', use_column_width=True)

def main():
    """Main function to run the Streamlit app."""
    st.title('Image Text and Visual Element Extractor')
    
    uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        st.write('Analyzing Image...')
        texts = analyze_image(uploaded_image)
        
        st.write('Extracted Texts:')
        extract_texts(texts)
        
        st.write('Segmenting Visual Elements...')
        segment_visual_elements(uploaded_image)

if __name__ == '__main__':
    main()

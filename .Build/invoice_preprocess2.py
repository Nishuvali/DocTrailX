import cv2
import re
import numpy as np
import csv  
import os  
from google.cloud import vision
from datetime import datetime, date 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract\tesseract.exe'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\doctrailx-cc7380fa08bb.json"

def preprocess_invoice(image_path, display=False):
    # Load image
    image = cv2.imread(image_path) 

    # Grayscale conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding (Experiment here)
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=9, C=8)
    print(thresh.dtype)  # Should be 'uint8'

    # Optional: Thresholding Alternatives
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    
    kernel = np.ones((1, 3), np.uint8)  # Start with a 3x3 kernel
    thresh = cv2.erode(thresh, kernel, iterations=1)  # Adjust iterations

    # Noise Removal (experiment with filters if needed)
    blurred_image = cv2.medianBlur(thresh, 5)  

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Relaxed Contour Filtering 
    cnts = [c for c in cnts if cv2.contourArea(c) > 200]  # Lowered threshold

    # Detect and draw bounding boxes 
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if display:
        cv2.imshow("Original", image)
        cv2.imshow("Preprocessed", thresh) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

    return image, thresh, cnts

def write_to_csv(invoice_data, csv_file_path):
    """Writes invoice data to a CSV file."""

    with open(csv_file_path, 'a', newline='') as csvfile: 
        fieldnames = ['invoice_number', 'date', 'total', 'vendor']  # Example
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If it's a new file, write the header row
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(invoice_data)
        
def process_response(response):
    invoice_data = {}

    for block in response.get('blocks', []):
        block_text = block.get('text')  

        # Invoice Number
        if 'Invoice Number:' in block_text:
            invoice_data['invoice_number'] = block_text.split(':')[-1].strip()

        if re.search(r'\d{2}/\d{2}/\d{4}', block_text):
            potential_date = re.search(r'\d{2}/\d{2}/\d{4}', block_text).group()
    try:
        # Attempt to parse the date (adjust strptime format if needed)
        invoice_data['due_date'] = datetime.datetime.strptime(potential_date, '%m/%d/%Y').date() 
    except ValueError:
        # Handle incorrect date formats if necessary
        print(f"Invalid date format found: {potential_date}") 

        # Total Amount
        if 'Total:' in block_text:
            total_str = block_text.split(':')[-1].strip()
            invoice_data['total_amount'] = re.sub(r'[^\d\.]', '', total_str)
        elif re.search(r'\$?\d+\.\d{2}', block_text):  # Currency-based search
            invoice_data['total_amount'] = block_text  # ... (Handle potential variations)

    return invoice_data


image_path = "C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\Sample 4.jpg" 

# Preprocess the image
image, thresh, cnts = preprocess_invoice(image_path) 

# Google Cloud Vision - Image Loading and API Call
client = vision.ImageAnnotatorClient()
with open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)
response = client.text_detection(image=image) 

# Extract data from the response
invoice_data = process_response(response)

# Output to CSV
write_to_csv(invoice_data, "invoice_data.csv")
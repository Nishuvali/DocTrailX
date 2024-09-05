import cv2
import re
import numpy as np
import csv  
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract\tesseract.exe'

def preprocess_invoice(image_path, display=False):
    # Load image
    image = cv2.imread(image_path) 

    # Grayscale conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding (Experiment here)
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, blockSize=11, C=5)

    # Optional: Thresholding Alternatives
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    
    kernel = np.ones((1, 1), np.uint8)  # Start with a 3x3 kernel
    thresh = cv2.erode(thresh, kernel, iterations=1)  # Adjust iterations

    # Noise Removal (experiment with filters if needed)
    #blurred_image = cv2.medianBlur(thresh, 5)  

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
'''
def find_template_matches(image, template, threshold=0.8):
    """Performs template matching and returns matching regions."""
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    return zip(*locations[::-1])  # Convert to x, y coordinates
'''
'''
invoice_number_template = cv2.imread('C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\invoice_number.jpg', 0) 
'''
# Example Usage
image_path = "C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\Sample 4.jpg" 
preprocessed_image, thresh, cnts = preprocess_invoice(image_path, display=True)  # Get 'thresh'

'''
# Template Matching on the preprocessed 'thresh' image
matches = find_template_matches(thresh, invoice_number_template, threshold=0.9)

# Draw Bounding Boxes (use the original 'preprocessed_image' for display)
for pt in matches:
    x, y = pt
    w, h = invoice_number_template.shape[::-1]
    cv2.rectangle(preprocessed_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('Image with Template Matches', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# ... after loading and preprocessing the image:
cv2.drawContours(preprocessed_image, cnts, -1, (0, 255, 0), 2) # Draw all contours
cv2.imshow("Bounding Boxes", preprocessed_image)
cv2.waitKey(0) 


def extract_text_from_image(image):
    """Performs OCR on an image."""
    text = pytesseract.image_to_string(image)
    return text

#image_path = "C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\Sample 4.jpg" 
#preprocessed_image, thresh, cnts = preprocess_invoice(image_path, display=True) 

# OCR within bounding boxes with potential refinements
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)

    # Refinement 1: Add a small padding
    x -= 5  # Expand box slightly (left)
    y -= 5  # Expand box slightly (top)
    w += 10 
    h += 10

    # Refinement 2: (Optional) Apply morphology before OCR
    roi = thresh[y:y+h, x:x+w]  
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    roi = cv2.erode(roi, kernel, iterations=1)  # Or use cv2.dilate
    print(x, y, w, h)
    roi_text = extract_text_from_image(roi)
    print(roi_text)
    
def extract_invoice_data(image_path, display=False):
    """
    Extracts invoice data from an image.

    Args:
        image_path (str): Path to the invoice image.
        display (bool, optional): Whether to display the preprocessed images. Defaults to False.

    Returns:
        dict: A dictionary containing the extracted invoice data.
    """

    # Preprocess the image
    image, thresh, cnts = preprocess_invoice(image_path, display)

    # Initialize the invoice data dictionary
    invoice_data = {}

    # Iterate over each bounding box
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # Extract text from the bounding box region
        roi = thresh[y:y+h, x:x+w]
        roi_text = extract_text_from_image(roi)

        # Field Identification Logic 
        if is_invoice_number(x, y, roi_text, image.shape[1], image.shape[0]): 
            invoice_data["invoice_number"] = roi_text
        elif is_date(x, y, roi_text, image.shape[1], image.shape[0]):  
            invoice_data["date"] = roi_text
        elif is_total_amount(x, y, roi_text, image.shape[1], image.shape[0]):
            invoice_data["total"] = roi_text
        elif is_vendor(x, y, roi_text, image.shape[1], image.shape[0]):
            invoice_data["vendor"] = roi_text
        # ... add more field checks as needed

    return invoice_data

def is_invoice_number(x, y, text, image_width, image_height):
    """Checks if the text and location likely represent the invoice number."""

    if x < image_width * 0.3 and y < image_height * 0.2:  # Check left-top 
        if re.match(r'INV-\d+', text):  # Adjust pattern if needed
            return True
    return False

def is_date(x, y, text,image_width, image_height):
    """Checks if the text and location likely represent the invoice date."""

    if x < image_width * 0.3 and y < image_height * 0.2:  # Check left-top 
        if re.match(r'\d{2}/\d{2}/\d{4}', text) or re.match(r'\d{4}-\d{2}-\d{2}', text): 
            return True
    return False

def is_total_amount(x, y, text,image_width, image_height):
    """Checks for the total invoice amount."""

    if x > image_width * 0.7 and y > image_height * 0.8:  # Check right-bottom
        if re.match(r'\$?\d+\.\d{2}', text):  # Adjust pattern if needed
            return True
    return False

def is_vendor(x, y, text, image_width, image_height):
    """Checks if the text is likely the vendor/supplier name."""

    if x < image_width * 0.3 and y < image_height * 0.3:  # Example: Check top-left
        # You might use keywords or other checks here
        return True 
    return False

def classify_invoice(invoice_data):
    keywords = {
        "Office Supplies": ["pens", "paper", "stapler", "printer"], 
        "IT Services": ["software", "web hosting", "consulting", "cloud"],
        "Marketing Expenses": ["advertising", "promotion", "social media"],
        "Grocery Expenses": ["food", "beverages", "supermarket"], 
        "Medical Bills": ["hospital", "doctor", "pharmacy", "medicine"]
    }
    vendors = {
    "Office Supplies": ["ABC Staples", "Paper Plus", "Office Depot", "SupplyMe"],
    "IT Services": ["Tech Solutions", "WebDev Corp", "CloudExperts"],
    "Marketing Expenses": ["SocialBoost Inc", "AdSense Media"],
    "Grocery Expenses": ["Whole Foods", "Green Grocer", "Farmer's Market"], 
    "Medical Bills": ["City Hospital",  "Dr. Johnson's Clinic", "Main Street Pharmacy"],
    }


    # 1. Keyword-Based Check
    for category, words in keywords.items():
        if any(word in invoice_data['text'].lower() for word in words):  
            return category

    # 2. Vendor-Based Check 
    if 'vendor' in invoice_data: 
        for category, vendor_list in vendors.items():
            if invoice_data['vendor'].lower() in vendor_list:
                return category

    # 3. Heuristics (Optional - Example)
    if 'invoice_number' in invoice_data and invoice_data['invoice_number'].startswith("MED-"):
        return "Medical Bills"

    return "Uncategorized" 

def write_to_csv(invoice_data, csv_file_path):
    """Writes invoice data to a CSV file."""

    with open(csv_file_path, 'a', newline='') as csvfile: 
        fieldnames = ['invoice_number', 'date', 'total', 'vendor']  # Example
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If it's a new file, write the header row
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(invoice_data)

# After extracting from an image
image_path = "C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\Sample 4.jpg" 
invoice_data = extract_invoice_data(image_path) 

# Output to CSV
write_to_csv(invoice_data, "invoice_data.csv")
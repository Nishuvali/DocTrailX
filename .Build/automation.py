import os
import time
import schedule
from invoice_preprocess import preprocess_invoice, extract_invoice_data
import sqlite3
import time
import signal
import csv
import smtplib
from email.message import EmailMessage

# Configuration
INVOICE_FOLDER = "C:\\Users\\saini\\OneDrive\\Documents\\DocTrailX\\NewInvoices"  # Update with your path
DATABASE_FILE = "invoice_data.db"  # SQLite database (adjust as needed)
REPORT_INTERVAL = 60  # Generate report every hour
EMAIL_THRESHOLD = 1000  # Send email notification if invoice total exceeds this amount
RECIPIENT_EMAIL = input("Enter the recipient email address: ")

def create_email(subject, body, to_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = 'nishwanthvaliveti@gmail.com'
    msg['To'] = to_email
    return msg

def send_email(subject, body, to_email):
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login('nishwanthvaliveti@gmail.com', 'byge xepl kdxs artk')
        server.send_message(create_email(subject, body, to_email))

# --- Database Setup (uncomment and modify if you're using a database) ---
def create_database_table():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            invoice_number TEXT PRIMARY KEY,
            date TEXT,
            total TEXT,
            vendor TEXT
        )
    """)
    conn.commit()
    conn.close()

create_database_table()  # Initialize the database table

# Test email
subject = "Test Email"
body = "This is a test email."
to_email = "sanjutherippergaming@gmail.comb"

send_email(subject, body, to_email)

# --- Main Functions ---

def process_new_invoice(image_path):
    # Load and preprocess the image 
    image, cropped, cnts = preprocess_invoice(image_path, display=False)  

    # Extract the invoice data
    invoice_data = extract_invoice_data(image, display=False)

    # Store in a database (modify if you're NOT using a database)
    store_invoice_data(invoice_data)

def check_for_invoices():
    print("Checking for new invoices...")
    new_files = [f for f in os.listdir(INVOICE_FOLDER) if f.endswith(('.jpg', '.png'))]
    for file in new_files:
        image_path = os.path.join(INVOICE_FOLDER, file)
        process_new_invoice(image_path)


def store_invoice_data(invoice_data):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Check if the invoice amount exceeds a threshold (e.g., $1000)
        if float(invoice_data['total'].replace(',', '')) > EMAIL_THRESHOLD:
            subject = "Invoice Amount Alert"
            body = f"Invoice {invoice_data['invoice_number']} has an amount of ${invoice_data['total']}. Please review."
            send_email(subject, body, RECIPIENT_EMAIL)

        cursor.execute("""
            INSERT INTO invoices VALUES (?, ?, ?, ?)
        """, (invoice_data['invoice_number'], invoice_data['date'], 
              invoice_data['total'], invoice_data['vendor']))
        conn.commit()
    except Exception as e:
        print(f"Error storing invoice data: {e}")
    finally:
        conn.close()
        
def generate_csv_report():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM invoices")
    rows = cursor.fetchall()
    conn.close()

    csv_file = "invoice_data.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['invoice_number', 'date', 'total', 'vendor']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({'invoice_number': row[0], 'date': row[1], 'total': row[2], 'vendor': row[3]})

    print(f"CSV report saved to {csv_file}")

# --- Scheduling --- 
# --- Scheduling ---  
schedule.every(1).minutes.do(check_for_invoices)  # Adjust frequency
schedule.every(REPORT_INTERVAL).minutes.do(generate_csv_report)  # Generate report every hour

print("Scheduler is running. Press CTRL+C to stop.")

def signal_handler(signal, frame):
    print("Script execution has been interrupted by the user.")
    # Perform any necessary cleanup here
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("Script execution has been interrupted by the user.")
    # Perform any necessary cleanup here
    exit(0)
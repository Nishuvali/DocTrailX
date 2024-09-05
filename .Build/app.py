from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
from uuid import uuid4
from validators import email
import os
import logging
from invoice_preprocess import preprocess_invoice, extract_invoice_data, process_invoice

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['LOG_FILE'] = 'app.log'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize the logger
logging.basicConfig(filename=app.config['LOG_FILE'], level=logging.DEBUG)

# Add a function to validate the email address format
def validate_email(email):
    if not email(email):
        logging.error('Invalid email address.')
        return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'invoice' not in request.files:
            logging.error('No file was uploaded.')
            return jsonify({
                'status': 'error',
                'message': 'No file was uploaded.'
            }), 400

        invoice = request.files['invoice']
        email = request.form['email']

        # Check if the email address is valid
        if not validate_email(email):
            logging.error('Invalid email address.')
            return jsonify({
                'status': 'error',
                'message': 'Invalid email address.'
            }), 400

        # Save the file to a temporary location
        invoice_filename = str(uuid4()) + '.' + secure_filename(invoice.filename)
        invoice.save(os.path.join(app.config['UPLOAD_FOLDER'], invoice_filename))

        # Preprocess the invoice and extract data
        preprocessed_image, cropped, cnts = preprocess_invoice(os.path.join(app.config['UPLOAD_FOLDER'], invoice_filename), display=False)
        invoice_data = extract_invoice_data(preprocessed_image, display=False)

        # Process the invoice data and store it in the database
        process_invoice(invoice_data, os.path.join(app.config['UPLOAD_FOLDER'], invoice_filename))

        # Return the results to the front-end interface
        return jsonify({
            'status': 'success',
            'message': f'Invoice {invoice_filename} processed successfully. Check your email for the results.',
            'email': email
        })

if __name__ == '__main__':
    app.run(debug=True)
import os
import json
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from celery import Celery, Task
from database import init_db, save_processed_document, get_processed_document, get_history
from processor import process_documents_task
from flask_swagger_ui import get_swaggerui_blueprint
from math import ceil
import fitz  # PyMuPDF for PDF creation
from PIL import Image
import io

load_dotenv()


def make_celery(app):
    class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.import_name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    return celery_app


app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config.update(
    CELERY=dict(
        broker_url="redis://redis:6379/0",
        result_backend="redis://redis:6379/0",
        task_ignore_result=False,
    )
)
celery = make_celery(app)


def normalize_image_for_pdf(image_bytes):
    """
    Helper function to ensure images are in a PDF-compatible format (JPEG).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Could not normalize image for PDF creation: {e}")
        return None


def create_pdf_from_images(image_bytes_list):
    """
    Creates a multi-page PDF in memory from a list of image bytes.
    """
    doc = fitz.open()
    for img_bytes in image_bytes_list:
        normalized_bytes = normalize_image_for_pdf(img_bytes)
        if not normalized_bytes:
            continue
        try:
            img_doc = fitz.open(stream=normalized_bytes, filetype="jpg")
            page = doc.new_page(width=img_doc[0].rect.width, height=img_doc[0].rect.height)
            page.insert_image(img_doc[0].rect, stream=normalized_bytes)
            img_doc.close()
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
            continue

    if len(doc) == 0:
        return None

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# --- Swagger UI API Documentation ---
SWAGGER_URL = '/api/docs'
API_URL = '/api/spec'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "OCR AI API"}
)
app.register_blueprint(swaggerui_blueprint)


@app.route('/api/spec')
def api_spec():
    """Dynamically serves the swagger.json file."""
    with open(os.path.join(app.static_folder, 'swagger.json')) as f:
        swagger_spec = json.load(f)

    swagger_spec['host'] = request.host
    scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
    swagger_spec['schemes'] = [scheme]
    swagger_spec['basePath'] = "/"

    return jsonify(swagger_spec)


@app.before_request
def setup():
    if not hasattr(app, 'db_initialized'):
        init_db()
        app.db_initialized = True


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        doc_type = request.form.get('doc_type')
        front_file = request.files.get('front_image')
        back_file = request.files.get('back_image')

        if not doc_type or not front_file or not back_file or front_file.filename == '' or back_file.filename == '':
            flash('Please select a document type and upload both front and back images.')
            return redirect(request.url)

        front_bytes = front_file.read()
        back_bytes = back_file.read()

        pdf_bytes = create_pdf_from_images([front_bytes, back_bytes])

        if not pdf_bytes:
            flash(
                'Could not create PDF from the provided images. Please try again with standard image formats (JPG, PNG).')
            return redirect(request.url)

        file_contents_dict = {
            "file_0": ("combined_id_card.pdf", pdf_bytes)
        }

        task = process_documents_task.delay(file_contents_dict, doc_type)
        return redirect(url_for('processing_page', task_id=task.id))

    return render_template('index.html')


# --- API ENDPOINT (Unchanged: Still accepts generic multi-file uploads) ---
@app.route('/api/v1/extract', methods=['POST'])
def api_extract():
    if 'files' not in request.files:
        return jsonify({"error": "No 'files' part in the request"}), 400

    files = request.files.getlist('files')
    doc_type = request.form.get('doc_type', 'Unknown')

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400

    file_contents_dict = {f"file_{i}": (f.filename, f.read()) for i, f in enumerate(files)}

    task = process_documents_task.delay(file_contents_dict, doc_type)

    return jsonify({
        "message": "Processing started.",
        "task_id": task.id,
        "status_url": url_for('task_status', task_id=task.id, _external=True)
    }), 202


# --- HISTORY PAGE ---
@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    history_items, total_count = get_history(page, per_page)

    last_page = ceil(total_count / per_page) if total_count > 0 else 1

    return render_template('history.html',
                           history=history_items,
                           page=page,
                           per_page=per_page,
                           last_page=last_page)


# --- TASK STATUS & RESULTS PAGES ---
@app.route('/processing/<task_id>')
def processing_page(task_id):
    return render_template('processing.html', task_id=task_id)


@app.route('/status/<task_id>')
def task_status(task_id):
    task = process_documents_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'status': task.info.get('status', ''), 'result': task.info.get('result')}
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)


@app.route('/results/<int:doc_id>')
def results(doc_id):
    document = get_processed_document(doc_id)
    if not document:
        flash('Document not found!', 'error')
        return redirect(url_for('index'))
    extracted_data = json.dumps(document['extracted_data'], indent=2)
    face_image_b64 = None
    if document['face_image']:
        face_image_b64 = base64.b64encode(document['face_image']).decode('utf-8')
    return render_template('results.html', document=document, extracted_data=extracted_data,
                           face_image_b64=face_image_b64)
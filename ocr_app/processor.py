import cv2
import numpy as np
import requests
import base64
import json
from paddleocr import PaddleOCR
from celery import shared_task
from database import save_processed_document
from PIL import Image
import io
import fitz  # PyMuPDF
from datetime import datetime

# --- Configuration ---
OLLAMA_API_URL = "http://ollama:11434/api/generate"
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
AI_MODEL = "minicpm-v:8b"
OCR_CONFIDENCE_THRESHOLD = 0.80  # Ignore any text PaddleOCR is less than 80% sure about.

# --- PaddleOCR Initialization ---
print("Initializing PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR Initialized.")


def normalize_image(image_bytes):
    """
    Opens any image, converts it to a standard RGB JPEG format,
    and returns the standardized image bytes. This prevents errors
    from unsupported image formats like WEBP, HEIC, etc.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error normalizing image: {e}")
        # Fallback to original bytes if normalization fails
        return image_bytes


def process_file_input(file_bytes, filename):
    """
    Accepts a file (image or PDF) and returns a list of standardized image bytes.
    If the file is a PDF, it converts each page into an image.
    """
    images_bytes = []

    if filename.lower().endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                # Render page to a high-resolution image
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("jpeg")
                images_bytes.append(normalize_image(img_bytes))
            doc.close()
        except Exception as e:
            print(f"Error processing PDF file '{filename}': {e}")
    else:
        # Process as a single image
        images_bytes.append(normalize_image(file_bytes))

    return images_bytes


def preprocess_image_for_ocr(image_bytes):
    """
    Applies pre-processing steps like grayscaling and adaptive thresholding
    to an image to improve OCR quality.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding to handle varying lighting conditions
        processed_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        _, buffer = cv2.imencode('.jpg', processed_img)
        return buffer.tobytes()
    except Exception as e:
        print(f"Could not preprocess image, falling back to original: {e}")
        return image_bytes


def extract_text_with_paddleocr(ordered_image_bytes):
    """Step 1: Use PaddleOCR with pre-processing and confidence filtering."""
    full_text = ""
    for i, img_bytes in enumerate(ordered_image_bytes):
        separator = f"\n--- TEXT FROM PAGE/IMAGE {i + 1} ---\n"
        full_text += separator
        try:
            processed_bytes = preprocess_image_for_ocr(img_bytes)
            result = paddle_ocr.ocr(processed_bytes)
            if result and result[0]:
                high_confidence_texts = [
                    line[1][0] for line in result[0] if line[1][1] > OCR_CONFIDENCE_THRESHOLD
                ]
                full_text += "\n".join(high_confidence_texts)
        except Exception as e:
            print(f"Error during PaddleOCR processing: {e}")
    return full_text


def structure_data_with_master_prompt(raw_text, base64_images, doc_type_hint):
    """
    Step 2: Uses the ultimate "Multi-Template" prompt to let the AI choose the
    best structure for the document it identifies, guided by the user's hint.
    """
    prompt = f"""
    You are a world-class data extraction expert. Your task is to analyze the provided document image(s) and raw OCR text to create a single, perfectly structured JSON output.

    Follow these steps meticulously:
    1.  **Consider Hint**: The user has suggested this document is a "{doc_type_hint}". Use this as a strong hint for your identification.
    2.  **Identify Document**: Examine the images and text to confirm the document type. If the hint seems correct, proceed. If it's clearly wrong (e.g., hint is "Passport" but image is a driver's license), use your own judgment.
    3.  **Select Template**: Choose the single best JSON template from the "Available Templates" list below that matches the identified document.
    4.  **Verify & Extract**: Use the images to visually verify and correct the `Raw OCR Text`. Extract all data needed to populate your chosen template.
    5.  **Populate Standard Fields**: Fill in the main fields of your chosen template. Format dates as YYYY-MM-DD and country codes as 3-letter ISO 3166-1 Alpha-3 codes (e.g., "USA", "PHL", "IND", "ARE"). Use `null` if a field is not present.
    6.  **Populate `additional_data`**: If you find any other important, labeled data that does not fit in the standard fields, add it as a key-value pair inside the `additional_data` object.
    7.  **Final Output**: Your response must be ONLY the single, minified JSON object based on your chosen template. Do not include explanations or markdown.

    --- Document Type Hint (from user) ---
    {doc_type_hint}

    --- Available Templates (Choose ONE) ---

    **Template for "Passport":**
    {{
      "document_type": "passport", "full_name": null, "surname": null, "given_names": null, "passport_number": null,
      "nationality": null, "issuing_country": null, "gender": null, "date_of_birth": null,
      "date_of_issue": null, "expiry_date": null, "place_of_birth": null, "issuing_authority": null,
      "mrz": null, "additional_data": {{}}
    }}

    **Template for "Driving License":**
    {{
      "document_type": "driving_license", "full_name": null, "license_number": null, "nationality": null,
      "gender": null, "date_of_birth": null, "address": null, "date_of_issue": null, "expiry_date": null,
      "vehicle_classes": [], "conditions": null, "agency_code": null, "serial_number": null,
      "additional_data": {{}}
    }}

    **Template for "Aadhaar Card":**
    {{
      "document_type": "aadhaar_card", "full_name": null, "date_of_birth": null, "gender": null,
      "aadhaar_number": null, "virtual_id": null, "address": null,
      "additional_data": {{}}
    }}

    **Template for "Emirates ID":**
    {{
      "document_type": "emirates_id", "full_name": null, "id_number": null, "nationality": null,
      "address": null, "date_of_birth": null, "expiry_date": null,
      "additional_data": {{}}
    }}

    **Template for "Generic ID / Other":**
    {{
      "document_type": "other", "document_title": null, "full_name": null, "id_number": null,
      "address": null, "organization": null, "date_of_issue": null, "expiry_date": null, "date_of_birth": null,
      "additional_data": {{}}
    }}

    --- Raw OCR Text (for guidance, verify against images) ---
    {raw_text}
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": AI_MODEL, "prompt": prompt, "images": base64_images, "stream": False, "format": "json"},
            timeout=600
        )
        response.raise_for_status()
        extracted_data = json.loads(response.json().get('response', '{}'))
        return extracted_data
    except Exception as e:
        return {"error": f"The language model failed to structure the text. Error: {e}"}


def post_process_and_validate(data):
    """A final, deterministic check to clean and standardize the AI's output."""
    if not isinstance(data, dict):
        return data
    for key, value in data.items():
        if "date" in key.lower() and isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%d %b %Y", "%B %d, %Y", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    data[key] = datetime.strptime(value, fmt).strftime("%Y-%m-%d")
                    break
                except (ValueError, TypeError):
                    continue
    return data


@shared_task(bind=True)
def process_documents_task(self, file_contents_dict, doc_type):
    """The main Celery task orchestrating the final, high-accuracy pipeline."""
    try:
        all_image_bytes = []
        original_images_to_save = []

        for key in sorted(file_contents_dict.keys()):
            filename, file_bytes = file_contents_dict[key]
            original_images_to_save.append(file_bytes)
            processed_images = process_file_input(file_bytes, filename)
            all_image_bytes.extend(processed_images)

        if not all_image_bytes:
            raise Exception("No valid images could be processed from the provided file(s).")

        self.update_state(state='PROGRESS', meta={'status': 'Cleaning images & performing high-accuracy OCR...'})
        raw_text = extract_text_with_paddleocr(all_image_bytes)

        self.update_state(state='PROGRESS', meta={'status': 'AI is analyzing and structuring the document...'})
        base64_images = [base64.b64encode(img).decode('utf-8') for img in all_image_bytes]
        structured_data = structure_data_with_master_prompt(raw_text, base64_images, doc_type)

        if "error" in structured_data:
            raise Exception(structured_data["error"])

        self.update_state(state='PROGRESS', meta={'status': 'Validating and formatting final data...'})
        final_data = post_process_and_validate(structured_data)

        self.update_state(state='PROGRESS', meta={'status': 'Detecting faces...'})
        face_image_bytes = detect_and_crop_face(all_image_bytes)

        self.update_state(state='PROGRESS', meta={'status': 'Saving to database...'})
        json_data = json.dumps(final_data)

        final_doc_type = final_data.get("document_type", doc_type).replace("_", " ").title()

        doc_id = save_processed_document(final_doc_type, json_data, original_images_to_save, face_image_bytes)

        return {'status': 'Task Complete!', 'result': doc_id}
    except Exception as e:
        raise e


def detect_and_crop_face(image_bytes_list):
    """Finds a face from any of the provided images."""
    for img_bytes in image_bytes_list:
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                face_crop = img[y:y + h, x:x + w]
                _, buffer = cv2.imencode('.jpg', face_crop)
                return buffer.tobytes()
        except Exception as e:
            print(f"Error during face detection: {e}")
            continue
    return None
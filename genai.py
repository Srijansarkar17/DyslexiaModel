from flask import Flask, request, jsonify, render_template, send_file, session
from groq import Groq
from dotenv import load_dotenv
import os
from PIL import Image
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
import uuid
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")  # Add a secret key for sessions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
    raise ValueError("GROQ_API_KEY environment variable must be set")

groq_client = Groq(api_key=GROQ_API_KEY)

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """
    Preprocess the image (resize, normalize, etc.) if needed.
    """
    try:
        image = image.resize((224, 224))  # Resize to 224x224 (adjust as needed)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def image_to_base64(image):
    """
    Convert PIL image to base64.
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

def generate_description(image):
    """
    Send the image to the Groq model and get the description.
    """
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image)

        # Call the Groq model
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I will write alphabets and numbers and you will have to find the error in that writing from the image. "},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        raise

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Handle image upload, generate description, and return results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Please upload a JPG, JPEG or PNG image."}), 400

    try:
        # Generate a unique filename to prevent collisions
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        
        # Save the uploaded file
        file.save(filepath)
        logger.info(f"Saved file to {filepath}")

        # Load and preprocess the image
        image = Image.open(filepath)
        image = preprocess_image(image)

        # Generate description
        description = generate_description(image)
        logger.info("Generated description for uploaded image")

        # Store the description in the session
        session["image_description"] = description
        session["image_path"] = filepath

        return jsonify({
            "success": True,
            "description": description,
            "filename": unique_filename
        })
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear the session data"""
    session.pop("image_description", None)
    session.pop("image_path", None)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)
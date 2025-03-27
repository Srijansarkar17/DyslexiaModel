from flask import Flask, request, jsonify, render_template, send_from_directory, session
from groq import Groq
from dotenv import load_dotenv
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import base64
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")  # Add a secret key for sessions

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

# Load the CNN model
try:
    model = load_model('Customcnn_model.h5')
    logger.info("Model loaded successfully")
    model.summary()
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Try to get input shape from model
try:
    # Different ways to get input shape
    if hasattr(model, 'input_shape'):
        input_shape = model.input_shape
    elif hasattr(model, 'inputs') and model.inputs:
        input_shape = model.inputs[0].shape
    else:
        # Default shape if we can't determine
        input_shape = (None, 32, 32, 3)
    
    # Remove batch dimension (None,)
    if input_shape[0] is None:
        input_shape = input_shape[1:]
    
    logger.info(f"Model expects input shape: {input_shape}")
except Exception as e:
    logger.error(f"Could not determine input shape: {e}")
    input_shape = (32, 32, 3)
    logger.info(f"Using default input shape: {input_shape}")

# Define the class labels
class_labels = [str(i) for i in range(10)] + \
               [chr(i) for i in range(ord('a'), ord('z')+1)] + \
               [chr(i) for i in range(ord('A'), ord('Z')+1)]

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_cnn(image_path):
    """
    Preprocess the image for the CNN model.
    """
    # Get target dimensions from model
    target_height = 32
    target_width = 32
    target_channels = 3
    
    if len(input_shape) >= 3:
        if None not in input_shape[:2]:
            target_height, target_width = input_shape[:2]
        if input_shape[-1] is not None:
            target_channels = input_shape[-1]
    
    logger.info(f"Target image dimensions: {target_height}x{target_width}x{target_channels}")
    
    # Read the image
    img = cv2.imread(image_path)
    
    # If image couldn't be read properly
    if img is None:
        img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Convert BGR to RGB (OpenCV reads images as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to match the model's expected input shape
    img = cv2.resize(img, (target_width, target_height))
    
    # Handle channel dimensions
    if target_channels == 1 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=-1)
    elif target_channels == 3 and len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Ensure the image has the correct shape
    logger.info(f"Preprocessed image shape: {img.shape}")
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

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

def generate_improvement_suggestions(image, prediction, confidence, accuracy):
    """
    Send the image and prediction data to the Groq model and get writing improvement suggestions.
    """
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image)

        # Create prompt with prediction results
        prompt = f"From the uploaded image, please analyze the handwriting and provide suggestions for improvement. The character was predicted to be '{prediction}' with confidence {confidence} and accuracy {accuracy}."

        # Call the Groq model
        response = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    """
    Process image upload, make CNN prediction, then get Groq suggestions.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Please upload a JPG, JPEG or PNG image."})
    
    try:
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        logger.info(f"Saved file to {file_path}")

        # STEP 1: Preprocess the image for CNN and make prediction
        processed_image = preprocess_image_for_cnn(file_path)
        predictions = model.predict(processed_image)[0]
        
        # Get the top prediction
        pred_index = np.argmax(predictions)
        pred_class = class_labels[pred_index]
        confidence = float(predictions[pred_index]) * 100
        
        # Calculate accuracy (using softmax probabilities)
        softmax_sum = np.sum(predictions)
        accuracy = float(predictions[pred_index] / softmax_sum) * 100 if softmax_sum > 0 else 0
        
        confidence_str = f"{confidence:.2f}%"
        accuracy_str = f"{accuracy:.2f}%"
        
        logger.info(f"CNN Prediction: {pred_class}, Confidence: {confidence_str}, Accuracy: {accuracy_str}")
        
        # STEP 2: Get improvement suggestions from Groq
        # Open with PIL for Groq processing
        pil_image = Image.open(file_path)
        suggestions = generate_improvement_suggestions(pil_image, pred_class, confidence_str, accuracy_str)
        logger.info("Generated writing improvement suggestions")
        
        # Return all results
        return jsonify({
            "success": True,
            "prediction": pred_class,
            "confidence": confidence_str,
            "accuracy": accuracy_str,
            "suggestions": suggestions,
            "image_path": f"/uploads/{unique_filename}"
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing image: {error_details}")
        return jsonify({
            "error": str(e),
            "details": error_details
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
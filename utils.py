import os
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the uploaded file has a valid extension
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to preprocess the image for model prediction
def preprocess_image(image_path, target_size=(150, 150)):
    """Load and preprocess image for model prediction."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# ✅ Updated 38 disease classes
disease_classes = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Healthy", "Cherry - Powdery Mildew",
    "Corn - Cercospora Leaf Spot (Gray Leaf Spot)", "Corn - Common Rust", "Corn - Healthy", "Corn - Northern Leaf Blight",
    "Grape - Black Rot", "Grape - Esca (Black Measles)", "Grape - Healthy", "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot", "Peach - Healthy",
    "Pepper, Bell - Bacterial Spot", "Pepper, Bell - Healthy",
    "Potato - Early Blight", "Potato - Healthy", "Potato - Late Blight",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Healthy", "Strawberry - Leaf Scorch",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Healthy", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites (Two-spotted Spider Mite)",
    "Tomato - Target Spot", "Tomato - Tomato Mosaic Virus", "Tomato - Tomato Yellow Leaf Curl Virus"
]


# ✅ Function to predict the plant disease
def predict_disease(model, image_array):
    """Predict the disease using the loaded model."""
    
    predictions = model.predict(image_array)[0]

    # Debug info
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # Check if the model output matches the number of classes
    model_classes = len(predictions)

    if model_classes != len(disease_classes):
        print(f"⚠️ Mismatch: Model has {model_classes} classes, but {len(disease_classes)} labels are defined.")
        return {
            "plantName": "Unknown",
            "diseaseName": "Class Mismatch",
            "confidence": 0.0
        }

    # Get the predicted disease and confidence score
    predicted_index = np.argmax(predictions)
    confidence = round(predictions[predicted_index] * 100, 2)

    # Get the full label (Plant - Disease)
    full_label = disease_classes[predicted_index]

    # Extract plant and disease names separately
    if " - " in full_label:
        plant_name, disease_name = full_label.split(" - ", 1)
    else:
        plant_name = "Unknown Plant"
        disease_name = full_label

    return {
        "plantName": plant_name,
        "diseaseName": disease_name,   # ✅ Only disease name
        "confidence": confidence
    }


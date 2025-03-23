# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# from utils import allowed_file, preprocess_image, predict_disease
# from tensorflow.keras.models import load_model
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Configure upload folder and allowed extensions
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the trained model
# MODEL_PATH = 'model.h5'

# try:
#     model = load_model(MODEL_PATH)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# # Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Route for the homepage
# @app.route('/')
# def home():
#     """Render the homepage."""
#     return render_template('index.html')


# # Route for image upload and analysis
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """Handle image upload and predict disease."""
    
#     # Validate if the model is loaded
#     if model is None:
#         return jsonify({'error': 'Model not loaded properly'}), 500

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         try:
#             # Preprocess image and predict
#             img_array = preprocess_image(file_path)
#             result = predict_disease(model, img_array)

#             # ✅ Convert NumPy float32/float64 to Python float
#             result = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in result.items()}

#             # ✅ Remove the uploaded image after processing to save space
#             os.remove(file_path)

#             return jsonify(result)

#         except Exception as e:
#             print(f"❌ Error during prediction: {e}")
#             return jsonify({'error': 'Prediction failed'}), 500

#     return jsonify({'error': 'Invalid file format'}), 400


# # Run the Flask server
# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from utils import allowed_file, preprocess_image, predict_disease
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Flask app
app = Flask(__name__)

import os

# ✅ Disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ✅ Suppress TensorFlow warnings
import tensorflow as tf
import logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ✅ Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = 'model.h5'

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# ROUTES
# =========================

# Route for the homepage
@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')


# Route for image upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image upload and predict disease."""
    
    # Validate if the model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save the file
            file.save(file_path)

            # Preprocess image and predict
            img_array = preprocess_image(file_path)
            result = predict_disease(model, img_array)

            # ✅ Convert NumPy float32/float64 to Python float
            result = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in result.items()}

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500

        finally:
            # ✅ Always remove the uploaded image after processing (even on exception)
            if os.path.exists(file_path):
                os.remove(file_path)

        return jsonify(result)

    return jsonify({'error': 'Invalid file format'}), 400


# =========================
# SERVER CONFIGURATION
# =========================

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # ✅ Threaded → Handles multiple requests concurrently



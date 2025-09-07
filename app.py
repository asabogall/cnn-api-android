from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import logging
from datetime import datetime
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar modelo
sys.path.append('.')
from Modelo_Al_ADAM import load_model

app = Flask(__name__)
CORS(app)

# Cargar modelo al inicio
logger.info("Cargando modelo CNN...")
try:
    model = load_model(load_weights=True)
    logger.info("✅ Modelo cargado exitosamente")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"❌ Error cargando modelo: {e}")
    MODEL_LOADED = False
    model = None

# Clases del modelo - ACTUALIZA CON LAS TUYAS
CLASS_NAMES = ['Clase_0', 'Clase_1', 'Clase_2', 'Clase_3', 'Clase_4']

def preprocess_image(image_data):
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((100, 100))
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        raise ValueError(f"Error procesando imagen: {str(e)}")

@app.route('/')
def health_check():
    return jsonify({
        'status': 'online',
        'message': 'CNN API funcionando en Render',
        'model_loaded': MODEL_LOADED,
        'classes_count': len(CLASS_NAMES),
        'platform': 'Render',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED or model is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no disponible'
        }), 503
    
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({
                'success': False,
                'error': 'Campo image requerido'
            }), 400
        
        image_data = request.json['image']
        processed_image = preprocess_image(image_data)
        
        # Predicción
        predictions = model.predict(processed_image, verbose=0)
        predictions = predictions[0]
        
        predicted_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        return jsonify({
            'success': True,
            'result': {
                'predicted_class': CLASS_NAMES[predicted_idx],
                'predicted_class_index': predicted_idx,
                'confidence': round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2)
            },
            'all_probabilities': [
                {
                    'class_name': CLASS_NAMES[i],
                    'probability': round(float(predictions[i]), 4),
                    'percentage': round(float(predictions[i]) * 100, 2)
                } for i in range(len(CLASS_NAMES))
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/classes')
def get_classes():
    return jsonify({
        'classes': CLASS_NAMES,
        'total': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
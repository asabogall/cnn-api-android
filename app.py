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

# Importar tu modelo
sys.path.append('.')
from Modelo_Al_ADAM import load_model

app = Flask(__name__)
CORS(app)

# Cargar modelo una sola vez
logger.info("Cargando modelo CNN...")
model = load_model(load_weights=True)
logger.info("Modelo cargado exitosamente")

# 🔥 IMPORTANTE: ACTUALIZA ESTAS CLASES CON LAS TUYAS REALES
CLASS_NAMES = [
    'Clase_0',  # ← Cambia por tus clases reales
    'Clase_1', 
    'Clase_2', 
    'Clase_3', 
    'Clase_4'
]

def preprocess_image(image_data):
    """Preprocesa imagen desde base64 para el modelo"""
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((100, 100), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error preprocesando imagen: {str(e)}")
        raise ValueError(f"Error procesando imagen: {str(e)}")

@app.route('/', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'online',
        'message': 'CNN API funcionando correctamente',
        'model_ready': True,
        'classes_count': len(CLASS_NAMES),
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para clasificación"""
    start_time = time.time()
    
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({
                'success': False,
                'error': 'Campo "image" requerido en JSON',
                'error_code': 'MISSING_IMAGE_FIELD'
            }), 400
        
        image_data = request.json['image']
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'Imagen vacía',
                'error_code': 'EMPTY_IMAGE'
            }), 400
        
        # Preprocesar imagen
        try:
            processed_image = preprocess_image(image_data)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'error_code': 'IMAGE_PROCESSING_ERROR'
            }), 400
        
        # Realizar predicción
        predictions = model.predict(processed_image, verbose=0)
        predictions = predictions[0]
        
        # Obtener resultados
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        predicted_class = CLASS_NAMES[predicted_class_index]
        
        # Respuesta
        response = {
            'success': True,
            'result': {
                'predicted_class': predicted_class,
                'predicted_class_index': predicted_class_index,
                'confidence': round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2)
            },
            'all_probabilities': [
                {
                    'class_name': CLASS_NAMES[i],
                    'class_index': i,
                    'probability': round(float(predictions[i]), 4),
                    'percentage': round(float(predictions[i]) * 100, 2)
                } for i in range(len(CLASS_NAMES))
            ],
            'metadata': {
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Predicción exitosa: {predicted_class} ({confidence:.3f})")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'error_code': 'INTERNAL_SERVER_ERROR'
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Retorna información sobre las clases del modelo"""
    return jsonify({
        'success': True,
        'classes': [
            {
                'index': i,
                'name': class_name,
                'display_name': class_name
            } for i, class_name in enumerate(CLASS_NAMES)
        ],
        'total_classes': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
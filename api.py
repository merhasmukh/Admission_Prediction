from flask import Flask, request, jsonify
import sys
import os
from datetime import datetime
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from prediction import AdmissionPredictor
except ImportError:
    print("Error: Please run 'python train_models.py' first to train the models.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize predictor
predictor = None

def initialize_predictor():
    """Initialize the predictor on startup"""
    global predictor
    try:
        predictor = AdmissionPredictor()
        print("✓ Admission predictor initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing predictor: {str(e)}")
        return False

@app.before_first_request
def startup():
    """Initialize components before first request"""
    if not initialize_predictor():
        print("Failed to initialize predictor. API may not work properly.")

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'University Admission + Dropout Prediction API',
        'version': '1.0.0',
        'status': 'active',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'predict_confirmation': '/predict/confirm',
            'predict_joining': '/predict/join',
            'predict_dropout': '/predict/dropout',
            'predict_journey': '/predict/journey',
            'batch_predict': '/predict/batch',
            'model_info': '/info',
            'health': '/health'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global predictor
    
    status = {
        'status': 'healthy' if predictor is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'predictor_loaded': predictor is not None
    }
    
    if predictor:
        model_info = predictor.get_model_info()
        status.update({
            'models_loaded': len(model_info['models_loaded']),
            'preprocessor_loaded': model_info['preprocessor_loaded']
        })
    
    return jsonify(status)

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        info = predictor.get_model_info()
        return jsonify({
            'status': 'success',
            'data': info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/confirm', methods=['POST'])
def predict_confirmation():
    """Predict admission confirmation"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        # Get student data from request
        student_data = request.get_json()
        
        if not student_data:
            return jsonify({'error': 'No student data provided'}), 400
        
        # Make prediction
        result, error = predictor.predict_admission_confirmation(student_data)
        
        if error:
            return jsonify({
                'status': 'error',
                'message': error,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        return jsonify({
            'status': 'success',
            'prediction_type': 'admission_confirmation',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/join', methods=['POST'])
def predict_joining():
    """Predict joining decision"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        # Get student data from request
        student_data = request.get_json()
        
        if not student_data:
            return jsonify({'error': 'No student data provided'}), 400
        
        # Make prediction
        result, error = predictor.predict_joining(student_data)
        
        if error:
            return jsonify({
                'status': 'error',
                'message': error,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        return jsonify({
            'status': 'success',
            'prediction_type': 'joining',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/dropout', methods=['POST'])
def predict_dropout():
    """Predict dropout risk"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        # Get student data from request
        student_data = request.get_json()
        
        if not student_data:
            return jsonify({'error': 'No student data provided'}), 400
        
        # Make prediction
        result, error = predictor.predict_dropout(student_data)
        
        if error:
            return jsonify({
                'status': 'error',
                'message': error,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        return jsonify({
            'status': 'success',
            'prediction_type': 'dropout',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/journey', methods=['POST'])
def predict_journey():
    """Predict complete student journey"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        # Get student data from request
        student_data = request.get_json()
        
        if not student_data:
            return jsonify({'error': 'No student data provided'}), 400
        
        # Make prediction
        result = predictor.predict_complete_journey(student_data)
        
        return jsonify({
            'status': 'success',
            'prediction_type': 'complete_journey',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict for multiple students"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        # Get batch data from request
        batch_data = request.get_json()
        
        if not batch_data:
            return jsonify({'error': 'No batch data provided'}), 400
        
        # Expect either 'students' array or direct array
        if 'students' in batch_data:
            students_data = batch_data['students']
        elif isinstance(batch_data, list):
            students_data = batch_data
        else:
            return jsonify({'error': 'Invalid batch data format'}), 400
        
        if not students_data:
            return jsonify({'error': 'No student data in batch'}), 400
        
        # Make batch predictions
        results = []
        for i, student_data in enumerate(students_data):
            try:
                result = predictor.predict_complete_journey(student_data)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'student_info': student_data
                })
        
        return jsonify({
            'status': 'success',
            'prediction_type': 'batch',
            'total_students': len(students_data),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'data': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/risk-profile', methods=['POST'])
def get_risk_profile():
    """Get comprehensive risk profile for a student"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        # Get student data from request
        student_data = request.get_json()
        
        if not student_data:
            return jsonify({'error': 'No student data provided'}), 400
        
        # Create risk profile
        risk_profile = predictor.create_risk_profile(student_data)
        
        return jsonify({
            'status': 'success',
            'prediction_type': 'risk_profile',
            'data': risk_profile,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': [
            '/',
            '/health',
            '/info',
            '/predict/confirm',
            '/predict/join',
            '/predict/dropout',
            '/predict/journey',
            '/predict/batch',
            '/predict/risk-profile'
        ],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("="*50)
    print("UNIVERSITY ADMISSION PREDICTION API")
    print("="*50)
    
    # Initialize predictor
    if initialize_predictor():
        print("Starting Flask API server...")
        print("API Documentation:")
        print("  • GET  /           - API information")
        print("  • GET  /health     - Health check")
        print("  • GET  /info       - Model information")
        print("  • POST /predict/confirm - Predict admission confirmation")
        print("  • POST /predict/join    - Predict joining decision")
        print("  • POST /predict/dropout - Predict dropout risk")
        print("  • POST /predict/journey - Predict complete journey")
        print("  • POST /predict/batch   - Batch predictions")
        print("  • POST /predict/risk-profile - Get risk profile")
        print("\nExample usage:")
        print("  curl -X POST http://localhost:5000/predict/journey \\")
        print("       -H 'Content-Type: application/json' \\")
        print("       -d '{\"Application_ID\":\"TEST001\",\"Gender\":\"Male\",...}'")
        print("\n" + "="*50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize predictor. Exiting.")
        sys.exit(1)

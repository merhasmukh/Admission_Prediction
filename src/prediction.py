import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdmissionPredictor:
    """
    Main prediction class for university admission outcomes
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.model_performance = {}
        self.load_components()
    
    def load_components(self):
        """
        Load all trained models and preprocessor
        """
        try:
            # Load preprocessor
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                from src.data_preprocessing import DataPreprocessor
                self.preprocessor = DataPreprocessor()
                self.preprocessor.load_preprocessor(preprocessor_path)
                print("✓ Preprocessor loaded successfully")
            else:
                print("⚠ Preprocessor not found")
            
            # Load models
            model_files = {
                'admission_confirmation': 'admission_confirmation_model.pkl',
                'joining': 'joining_model.pkl',
                'dropout': 'dropout_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    print(f"✓ {model_name} model loaded successfully")
                else:
                    print(f"⚠ {model_name} model not found")
            
            # Load performance metrics
            performance_path = os.path.join(self.models_dir, 'model_performance.pkl')
            if os.path.exists(performance_path):
                self.model_performance = joblib.load(performance_path)
                print("✓ Model performance metrics loaded")
            
        except Exception as e:
            print(f"Error loading components: {str(e)}")
    
    def preprocess_input(self, student_data):
        """
        Preprocess input data for prediction
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Cannot preprocess data.")
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = student_data.copy()
        
        # Preprocess using fitted preprocessor
        processed_df = self.preprocessor.transform(df)
        
        # Get feature columns
        target_cols = ['Admission_Confirmed', 'Joined', 'Week1_Dropout', 'Week2_Dropout', 
                      'Week3_Dropout', 'Week4_Dropout', 'Final_Dropout_Label']
        exclude_cols = target_cols + ['Application_ID']
        feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
        
        return processed_df[feature_cols]
    
    def predict_admission_confirmation(self, student_data):
        """
        Predict if student will confirm admission
        """
        if 'admission_confirmation' not in self.models:
            return None, "Admission confirmation model not available"
        
        try:
            # Preprocess data
            X = self.preprocess_input(student_data)
            
            # Make prediction
            model = self.models['admission_confirmation']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Will Confirm' if prediction == 1 else 'Will Not Confirm',
                'confidence': float(probability[1]) if probability is not None else None,
                'model_performance': self.model_performance.get('admission_confirmation', {})
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error in admission confirmation prediction: {str(e)}"
    
    def predict_joining(self, student_data):
        """
        Predict if confirmed student will join
        """
        if 'joining' not in self.models:
            return None, "Joining prediction model not available"
        
        try:
            # Preprocess data
            X = self.preprocess_input(student_data)
            
            # Make prediction
            model = self.models['joining']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Will Join' if prediction == 1 else 'Will Not Join',
                'confidence': float(probability[1]) if probability is not None else None,
                'model_performance': self.model_performance.get('joining', {})
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error in joining prediction: {str(e)}"
    
    def predict_dropout(self, student_data):
        """
        Predict if joined student will drop out
        """
        if 'dropout' not in self.models:
            return None, "Dropout prediction model not available"
        
        try:
            # Preprocess data
            X = self.preprocess_input(student_data)
            
            # Make prediction
            model = self.models['dropout']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Will Drop Out' if prediction == 1 else 'Will Continue',
                'confidence': float(probability[1]) if probability is not None else None,
                'model_performance': self.model_performance.get('dropout', {})
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error in dropout prediction: {str(e)}"
    
    def predict_complete_journey(self, student_data):
        """
        Predict complete student journey: confirmation -> joining -> dropout
        """
        results = {
            'student_info': student_data if isinstance(student_data, dict) else student_data.iloc[0].to_dict(),
            'predictions': {},
            'journey_probability': 1.0,
            'risk_assessment': 'Low'
        }
        
        # Step 1: Admission Confirmation
        confirm_result, confirm_error = self.predict_admission_confirmation(student_data)
        if confirm_result:
            results['predictions']['admission_confirmation'] = confirm_result
            
            if confirm_result['prediction'] == 1:  # Will confirm
                # Step 2: Joining Prediction (only if confirmed)
                join_result, join_error = self.predict_joining(student_data)
                if join_result:
                    results['predictions']['joining'] = join_result
                    
                    if join_result['prediction'] == 1:  # Will join
                        # Step 3: Dropout Prediction (only if joined)
                        dropout_result, dropout_error = self.predict_dropout(student_data)
                        if dropout_result:
                            results['predictions']['dropout'] = dropout_result
                            
                            # Calculate overall journey probability
                            confirm_prob = confirm_result.get('confidence', 0.5)
                            join_prob = join_result.get('confidence', 0.5)
                            continue_prob = 1 - dropout_result.get('confidence', 0.5)
                            
                            results['journey_probability'] = confirm_prob * join_prob * continue_prob
                            
                            # Risk assessment
                            dropout_risk = dropout_result.get('confidence', 0.5)
                            if dropout_risk > 0.7:
                                results['risk_assessment'] = 'High'
                            elif dropout_risk > 0.4:
                                results['risk_assessment'] = 'Medium'
                            else:
                                results['risk_assessment'] = 'Low'
                        else:
                            results['predictions']['dropout'] = {'error': dropout_error}
                    else:
                        results['predictions']['joining']['note'] = 'Student unlikely to join'
                else:
                    results['predictions']['joining'] = {'error': join_error}
            else:
                results['predictions']['admission_confirmation']['note'] = 'Student unlikely to confirm admission'
        else:
            results['predictions']['admission_confirmation'] = {'error': confirm_error}
        
        return results
    
    def predict_batch(self, students_data):
        """
        Predict for multiple students
        """
        if isinstance(students_data, str):
            # Load from CSV file
            students_df = pd.read_csv(students_data)
        else:
            students_df = students_data.copy()
        
        batch_results = []
        
        print(f"Processing {len(students_df)} students...")
        
        for idx, row in students_df.iterrows():
            student_data = row.to_dict()
            result = self.predict_complete_journey(student_data)
            result['student_id'] = idx
            batch_results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} students...")
        
        return batch_results
    
    def get_model_info(self):
        """
        Get information about loaded models
        """
        info = {
            'models_loaded': list(self.models.keys()),
            'preprocessor_loaded': self.preprocessor is not None,
            'model_performance': self.model_performance
        }
        
        return info
    
    def create_risk_profile(self, student_data):
        """
        Create a comprehensive risk profile for a student
        """
        journey_result = self.predict_complete_journey(student_data)
        
        # Extract key information
        student_info = journey_result['student_info']
        predictions = journey_result['predictions']
        
        # Create risk profile
        risk_profile = {
            'student_id': student_info.get('Application_ID', 'Unknown'),
            'overall_risk': journey_result['risk_assessment'],
            'journey_success_probability': journey_result['journey_probability'],
            'risk_factors': [],
            'recommendations': []
        }
        
        # Analyze risk factors based on student data
        if student_info.get('Marks_12th_Percent', 0) < 60:
            risk_profile['risk_factors'].append('Low 12th grade marks')
            risk_profile['recommendations'].append('Provide academic support')
        
        if student_info.get('Entrance_Exam_Score', 0) < 40:
            risk_profile['risk_factors'].append('Low entrance exam score')
            risk_profile['recommendations'].append('Offer remedial classes')
        
        if student_info.get('Distance_From_Home_KM', 0) > 200:
            risk_profile['risk_factors'].append('Lives far from campus')
            risk_profile['recommendations'].append('Provide hostel accommodation')
        
        if student_info.get('Family_Income') == 'Low':
            risk_profile['risk_factors'].append('Low family income')
            risk_profile['recommendations'].append('Offer financial assistance')
        
        if student_info.get('Admission_Round', 1) > 1:
            risk_profile['risk_factors'].append('Late admission round')
            risk_profile['recommendations'].append('Provide orientation support')
        
        # Add prediction-based recommendations
        if 'dropout' in predictions and predictions['dropout'].get('confidence', 0) > 0.5:
            risk_profile['recommendations'].append('Monitor closely during first month')
            risk_profile['recommendations'].append('Assign mentor for support')
        
        return risk_profile

def create_sample_student():
    """
    Create a sample student for testing
    """
    return {
        'Application_ID': 'TEST001',
        'Gender': 'Male',
        'Marks_12th_Percent': 78.5,
        'Entrance_Exam_Score': 65.0,
        'Form_Submission_Date': '2024-06-15',
        'Category': 'General',
        'Admission_Round': 1,
        'Distance_From_Home_KM': 150,
        'Family_Income': 'Middle',
        'First_Preference': 1
    }

def main():
    """
    Main function for testing predictions
    """
    # Initialize predictor
    predictor = AdmissionPredictor()
    
    # Get model info
    print("Model Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with sample student
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE STUDENT")
    print("="*50)
    
    sample_student = create_sample_student()
    print("Sample student data:")
    for key, value in sample_student.items():
        print(f"  {key}: {value}")
    
    # Complete journey prediction
    print("\nComplete Journey Prediction:")
    result = predictor.predict_complete_journey(sample_student)
    
    print(f"Overall Risk Assessment: {result['risk_assessment']}")
    print(f"Journey Success Probability: {result['journey_probability']:.3f}")
    
    for stage, prediction in result['predictions'].items():
        if 'error' not in prediction:
            print(f"\n{stage.title()}:")
            print(f"  Prediction: {prediction['prediction_label']}")
            if prediction['confidence']:
                print(f"  Confidence: {prediction['confidence']:.3f}")
    
    # Risk profile
    print("\nRisk Profile:")
    risk_profile = predictor.create_risk_profile(sample_student)
    print(f"Risk Factors: {risk_profile['risk_factors']}")
    print(f"Recommendations: {risk_profile['recommendations']}")

if __name__ == "__main__":
    main()

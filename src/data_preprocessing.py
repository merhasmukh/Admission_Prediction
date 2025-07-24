import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import joblib
import os

class DataPreprocessor:
    """
    Data preprocessing pipeline for university admission prediction.
    Handles encoding, missing values, and feature engineering.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        
    def create_date_features(self, df):
        """Create date-based features from form submission date"""
        df = df.copy()
        
        # Convert to datetime
        df['Form_Submission_Date'] = pd.to_datetime(df['Form_Submission_Date'])
        
        # Extract date features
        df['Submission_Month'] = df['Form_Submission_Date'].dt.month
        df['Submission_Day'] = df['Form_Submission_Date'].dt.day
        df['Submission_Weekday'] = df['Form_Submission_Date'].dt.weekday
        
        # Days since start of admission period (assuming June 1st is start)
        admission_start = datetime(2024, 6, 1)
        df['Days_Since_Admission_Start'] = (df['Form_Submission_Date'] - admission_start).dt.days
        
        # Drop original date column
        df = df.drop('Form_Submission_Date', axis=1)
        
        return df
    
    def create_engineered_features(self, df):
        """Create additional engineered features"""
        df = df.copy()
        
        # Academic performance composite score
        df['Academic_Score'] = (df['Marks_12th_Percent'] * 0.6 + df['Entrance_Exam_Score'] * 0.4)
        
        # Performance categories
        df['High_Performer'] = ((df['Marks_12th_Percent'] >= 85) & (df['Entrance_Exam_Score'] >= 70)).astype(int)
        df['Low_Performer'] = ((df['Marks_12th_Percent'] < 60) | (df['Entrance_Exam_Score'] < 40)).astype(int)
        
        # Distance categories
        df['Local_Student'] = (df['Distance_From_Home_KM'] <= 50).astype(int)
        df['Distant_Student'] = (df['Distance_From_Home_KM'] > 200).astype(int)
        
        # Late application indicator
        df['Late_Application'] = (df['Days_Since_Admission_Start'] > 60).astype(int)
        
        # Round preference interaction
        df['First_Choice_Early_Round'] = (df['First_Preference'] & (df['Admission_Round'] == 1)).astype(int)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using label encoding"""
        df = df.copy()
        
        categorical_columns = ['Gender', 'Category', 'Family_Income']
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df[col].unique())
                        known_values = set(self.label_encoders[col].classes_)
                        
                        if not unique_values.issubset(known_values):
                            # Add unknown categories to encoder
                            new_values = unique_values - known_values
                            extended_classes = list(self.label_encoders[col].classes_) + list(new_values)
                            self.label_encoders[col].classes_ = np.array(extended_classes)
                        
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def handle_missing_values(self, df, fit=True):
        """Handle missing values using median imputation for numerical features"""
        df = df.copy()
        
        # Identify numerical columns (excluding target variables)
        target_cols = ['Admission_Confirmed', 'Joined', 'Week1_Dropout', 'Week2_Dropout', 
                      'Week3_Dropout', 'Week4_Dropout', 'Final_Dropout_Label']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in target_cols and col != 'Application_ID']
        
        if fit:
            df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.imputer.transform(df[numerical_cols])
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        df = df.copy()
        
        # Identify features to scale (excluding binary features and targets)
        target_cols = ['Admission_Confirmed', 'Joined', 'Week1_Dropout', 'Week2_Dropout', 
                      'Week3_Dropout', 'Week4_Dropout', 'Final_Dropout_Label']
        
        binary_cols = ['First_Preference', 'High_Performer', 'Low_Performer', 
                      'Local_Student', 'Distant_Student', 'Late_Application', 
                      'First_Choice_Early_Round']
        
        exclude_cols = target_cols + binary_cols + ['Application_ID']
        
        scale_cols = [col for col in df.columns if col not in exclude_cols]
        
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df
    
    def fit_transform(self, df):
        """Complete preprocessing pipeline - fit and transform"""
        print("Starting data preprocessing...")
        
        # Store original shape
        original_shape = df.shape
        print(f"Original data shape: {original_shape}")
        
        # Step 1: Create date features
        df = self.create_date_features(df)
        print("✓ Date features created")
        
        # Step 2: Create engineered features
        df = self.create_engineered_features(df)
        print("✓ Engineered features created")
        
        # Step 3: Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        print("✓ Categorical features encoded")
        
        # Step 4: Handle missing values
        df = self.handle_missing_values(df, fit=True)
        print("✓ Missing values handled")
        
        # Step 5: Scale features
        df = self.scale_features(df, fit=True)
        print("✓ Features scaled")
        
        # Store feature columns for later use
        target_cols = ['Admission_Confirmed', 'Joined', 'Week1_Dropout', 'Week2_Dropout', 
                      'Week3_Dropout', 'Week4_Dropout', 'Final_Dropout_Label']
        self.feature_columns = [col for col in df.columns if col not in target_cols + ['Application_ID']]
        
        print(f"Final data shape: {df.shape}")
        print(f"Features created: {len(self.feature_columns)}")
        print("Preprocessing completed successfully!")
        
        return df
    
    def transform(self, df):
        """Transform new data using fitted preprocessors"""
        print("Transforming new data...")
        
        # Apply same preprocessing steps (without fitting)
        df = self.create_date_features(df)
        df = self.create_engineered_features(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.handle_missing_values(df, fit=False)
        df = self.scale_features(df, fit=False)
        
        print("Data transformation completed!")
        return df
    
    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load a fitted preprocessor"""
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.imputer = preprocessor_data['imputer']
        self.feature_columns = preprocessor_data['feature_columns']
        
        print(f"Preprocessor loaded from {filepath}")

def preprocess_data(input_file, output_file=None):
    """
    Main function to preprocess the admission data
    """
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit and transform data
    processed_df = preprocessor.fit_transform(df)
    
    # Save processed data
    if output_file:
        processed_df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    return processed_df, preprocessor

if __name__ == "__main__":
    # Process the sample data
    processed_df, preprocessor = preprocess_data(
        'data/sample_data.csv', 
        'data/processed/processed_data.csv'
    )
    
    print("\nPreprocessed data info:")
    print(processed_df.info())
    print("\nFirst few rows:")
    print(processed_df.head())

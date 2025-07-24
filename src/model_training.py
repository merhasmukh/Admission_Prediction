import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AdmissionModelTrainer:
    """
    Train three separate models for university admission prediction:
    1. Admission Confirmation Model
    2. Joining Prediction Model  
    3. Dropout Prediction Model
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        
    def prepare_data_for_model(self, df, target_column, filter_condition=None):
        """
        Prepare data for a specific model by filtering and selecting features
        """
        # Apply filter condition if provided
        if filter_condition is not None:
            df_filtered = df[filter_condition].copy()
        else:
            df_filtered = df.copy()
        
        # Define feature columns (exclude targets and ID)
        target_cols = ['Admission_Confirmed', 'Joined', 'Week1_Dropout', 'Week2_Dropout', 
                      'Week3_Dropout', 'Week4_Dropout', 'Final_Dropout_Label']
        exclude_cols = target_cols + ['Application_ID']
        
        feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]
        
        X = df_filtered[feature_cols]
        y = df_filtered[target_column]
        
        return X, y, feature_cols
    
    def train_admission_confirmation_model(self, df):
        """
        Train model to predict if student will confirm admission
        Uses all application data
        """
        print("\n" + "="*50)
        print("TRAINING ADMISSION CONFIRMATION MODEL")
        print("="*50)
        
        # Prepare data - use all students
        X, y, feature_cols = self.prepare_data_for_model(df, 'Admission_Confirmed')
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Test F1 Score: {f1:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (F1: {best_score:.4f})")
        
        # Store model and performance
        self.models['admission_confirmation'] = best_model
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model_name': best_model_name
        }
        
        self.model_performance['admission_confirmation'] = performance
        
        print(f"\nFinal Performance:")
        for metric, value in performance.items():
            if metric != 'model_name':
                print(f"{metric.capitalize()}: {value:.4f}")
        
        return best_model, performance
    
    def train_joining_model(self, df):
        """
        Train model to predict if confirmed student will join
        Uses only confirmed students
        """
        print("\n" + "="*50)
        print("TRAINING JOINING PREDICTION MODEL")
        print("="*50)
        
        # Prepare data - only confirmed students
        X, y, feature_cols = self.prepare_data_for_model(
            df, 'Joined', filter_condition=df['Admission_Confirmed'] == 1
        )
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        if len(X) < 50:
            print("Warning: Very few confirmed students for training joining model")
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Test F1 Score: {f1:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (F1: {best_score:.4f})")
        
        # Store model and performance
        self.models['joining'] = best_model
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model_name': best_model_name
        }
        
        self.model_performance['joining'] = performance
        
        print(f"\nFinal Performance:")
        for metric, value in performance.items():
            if metric != 'model_name':
                print(f"{metric.capitalize()}: {value:.4f}")
        
        return best_model, performance
    
    def train_dropout_model(self, df):
        """
        Train model to predict if joined student will drop out
        Uses only joined students
        """
        print("\n" + "="*50)
        print("TRAINING DROPOUT PREDICTION MODEL")
        print("="*50)
        
        # Prepare data - only joined students
        X, y, feature_cols = self.prepare_data_for_model(
            df, 'Final_Dropout_Label', filter_condition=df['Joined'] == 1
        )
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        if len(X) < 30:
            print("Warning: Very few joined students for training dropout model")
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to try
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=3),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Test F1 Score: {f1:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (F1: {best_score:.4f})")
        
        # Store model and performance
        self.models['dropout'] = best_model
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model_name': best_model_name
        }
        
        self.model_performance['dropout'] = performance
        
        print(f"\nFinal Performance:")
        for metric, value in performance.items():
            if metric != 'model_name':
                print(f"{metric.capitalize()}: {value:.4f}")
        
        return best_model, performance
    
    def train_all_models(self, df):
        """
        Train all three models sequentially
        """
        print("Starting training of all admission prediction models...")
        print(f"Dataset shape: {df.shape}")
        
        # Train admission confirmation model
        self.train_admission_confirmation_model(df)
        
        # Train joining model
        self.train_joining_model(df)
        
        # Train dropout model
        self.train_dropout_model(df)
        
        print("\n" + "="*50)
        print("ALL MODELS TRAINING COMPLETED")
        print("="*50)
        
        # Summary
        print("\nModel Performance Summary:")
        for model_name, performance in self.model_performance.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Algorithm: {performance['model_name']}")
            print(f"  F1 Score: {performance['f1']:.4f}")
            print(f"  Accuracy: {performance['accuracy']:.4f}")
            print(f"  Precision: {performance['precision']:.4f}")
            print(f"  Recall: {performance['recall']:.4f}")
    
    def save_models(self, models_dir='models'):
        """
        Save all trained models
        """
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model is not None:
                filepath = os.path.join(models_dir, f'{model_name}_model.pkl')
                joblib.dump(model, filepath)
                print(f"Saved {model_name} model to {filepath}")
        
        # Save performance metrics
        performance_file = os.path.join(models_dir, 'model_performance.pkl')
        joblib.dump(self.model_performance, performance_file)
        print(f"Saved performance metrics to {performance_file}")
    
    def load_models(self, models_dir='models'):
        """
        Load all trained models
        """
        model_files = {
            'admission_confirmation': 'admission_confirmation_model.pkl',
            'joining': 'joining_model.pkl',
            'dropout': 'dropout_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name} model from {filepath}")
        
        # Load performance metrics
        performance_file = os.path.join(models_dir, 'model_performance.pkl')
        if os.path.exists(performance_file):
            self.model_performance = joblib.load(performance_file)
            print(f"Loaded performance metrics from {performance_file}")

def train_models(data_file):
    """
    Main function to train all models
    """
    # Load processed data
    print(f"Loading processed data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Initialize trainer
    trainer = AdmissionModelTrainer()
    
    # Train all models
    trainer.train_all_models(df)
    
    # Save models
    trainer.save_models()
    
    return trainer

if __name__ == "__main__":
    # Train models using processed data
    trainer = train_models('data/processed/processed_data.csv')

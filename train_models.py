#!/usr/bin/env python3
"""
Main script to train all University Admission + Dropout Prediction models
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_all_models

def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("UNIVERSITY ADMISSION + DROPOUT PREDICTION MODEL TRAINING")
    print("="*70)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check if sample data exists, if not generate it
    sample_data_path = 'data/sample_data.csv'
    if not os.path.exists(sample_data_path):
        print("\nStep 1: Generating sample data...")
        from data_generator import save_sample_data
        save_sample_data()
    else:
        print(f"\nStep 1: Sample data already exists at {sample_data_path}")
    
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    processed_data_path = 'data/processed/processed_data.csv'
    os.makedirs('data/processed', exist_ok=True)
    
    processed_df, preprocessor = preprocess_data(sample_data_path, processed_data_path)
    print(f"✓ Preprocessed data saved to {processed_data_path}")
    
    # Step 3: Train models
    print("\nStep 3: Training models...")
    trainer = train_models(processed_data_path)
    print("✓ All models trained and saved")
    
    # Step 4: Evaluate models
    print("\nStep 4: Evaluating models...")
    evaluator = evaluate_all_models(processed_data_path)
    print("✓ Model evaluation completed")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("\nFiles created:")
    print(f"  • Sample data: {sample_data_path}")
    print(f"  • Processed data: {processed_data_path}")
    print(f"  • Preprocessor: models/preprocessor.pkl")
    print(f"  • Models: models/")
    print(f"  • Evaluation results: models/evaluation_results.pkl")
    print(f"  • Reports: reports/")
    
    print("\nNext steps:")
    print("  1. Run 'streamlit run dashboard.py' to launch the dashboard")
    print("  2. Run 'python api.py' to start the REST API")
    print("  3. Use src/prediction.py for programmatic predictions")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

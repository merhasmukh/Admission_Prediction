import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve
import joblib
import os

class ModelEvaluator:
    """
    Comprehensive evaluation of admission prediction models
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_single_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model with comprehensive metrics
        """
        print(f"\n{'='*50}")
        print(f"EVALUATING {model_name.upper()} MODEL")
        print(f"{'='*50}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Add AUC if probabilities available
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"True Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        return metrics
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """
        Plot confusion matrix for a model
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        cm = self.evaluation_results[model_name]['metrics']['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f'Confusion Matrix - {model_name.title()} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, model_name, save_path=None):
        """
        Plot ROC curve for a model
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        if results['probabilities'] is None:
            print(f"No probabilities available for {model_name}")
            return
        
        y_true = results['true_labels']
        y_proba = results['probabilities']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = results['metrics']['auc_roc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name.title()} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.title()} Model')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=15, save_path=None):
        """
        Plot feature importance for tree-based models
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name.title()} Model')
        plt.barh(range(top_n), importances[indices][::-1])
        plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        # Print top features
        print(f"\nTop {top_n} Most Important Features for {model_name.title()}:")
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:25s} ({importances[idx]:.4f})")
    
    def compare_models(self, save_path=None):
        """
        Compare performance across all evaluated models
        """
        if not self.evaluation_results:
            print("No models have been evaluated yet")
            return
        
        # Prepare comparison data
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name.title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics.get('auc_roc', 'N/A')
            })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Plot comparison
        if len(comparison_data) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Performance Comparison', fontsize=16)
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i//2, i%2]
                values = [d[metric] for d in comparison_data]
                models = [d['Model'] for d in comparison_data]
                
                bars = ax.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
                ax.set_title(metric)
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Rotate x-axis labels if needed
                if len(models) > 2:
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Comparison plot saved to {save_path}")
            
            plt.show()
        
        return comparison_df
    
    def generate_evaluation_report(self, output_dir='reports'):
        """
        Generate comprehensive evaluation report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating evaluation report in {output_dir}/...")
        
        # Create plots for each model
        for model_name in self.evaluation_results.keys():
            # Confusion matrix
            cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
            self.plot_confusion_matrix(model_name, cm_path)
            
            # ROC curve
            roc_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
            self.plot_roc_curve(model_name, roc_path)
        
        # Model comparison
        comparison_path = os.path.join(output_dir, 'model_comparison.png')
        comparison_df = self.compare_models(comparison_path)
        
        # Save comparison data
        comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        print(f"Evaluation report generated successfully in {output_dir}/")
    
    def save_evaluation_results(self, filepath):
        """
        Save evaluation results to file
        """
        joblib.dump(self.evaluation_results, filepath)
        print(f"Evaluation results saved to {filepath}")
    
    def load_evaluation_results(self, filepath):
        """
        Load evaluation results from file
        """
        self.evaluation_results = joblib.load(filepath)
        print(f"Evaluation results loaded from {filepath}")

def evaluate_all_models(data_file, models_dir='models'):
    """
    Main function to evaluate all trained models
    """
    from src.data_preprocessing import DataPreprocessor
    from src.model_training import AdmissionModelTrainer
    
    print("Loading data and models for evaluation...")
    
    # Load processed data
    df = pd.read_csv(data_file)
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor(os.path.join(models_dir, 'preprocessor.pkl'))
    
    # Load models
    trainer = AdmissionModelTrainer()
    trainer.load_models(models_dir)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    model_configs = [
        ('admission_confirmation', 'Admission_Confirmed', None),
        ('joining', 'Joined', df['Admission_Confirmed'] == 1),
        ('dropout', 'Final_Dropout_Label', df['Joined'] == 1)
    ]
    
    for model_name, target_col, filter_condition in model_configs:
        if model_name in trainer.models and trainer.models[model_name] is not None:
            print(f"\nEvaluating {model_name} model...")
            
            # Prepare test data
            if filter_condition is not None:
                test_df = df[filter_condition].copy()
            else:
                test_df = df.copy()
            
            # Get features and target
            target_cols = ['Admission_Confirmed', 'Joined', 'Week1_Dropout', 'Week2_Dropout', 
                          'Week3_Dropout', 'Week4_Dropout', 'Final_Dropout_Label']
            exclude_cols = target_cols + ['Application_ID']
            feature_cols = [col for col in test_df.columns if col not in exclude_cols]
            
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            # Evaluate model
            model = trainer.models[model_name]
            evaluator.evaluate_single_model(model, X_test, y_test, model_name)
            
            # Plot feature importance
            evaluator.plot_feature_importance(model, feature_cols, model_name)
    
    # Generate comprehensive report
    evaluator.generate_evaluation_report()
    
    # Save results
    evaluator.save_evaluation_results(os.path.join(models_dir, 'evaluation_results.pkl'))
    
    return evaluator

if __name__ == "__main__":
    # Evaluate all models
    evaluator = evaluate_all_models('data/processed/processed_data.csv')

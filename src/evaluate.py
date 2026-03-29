"""
Model Evaluation Module for Network Intrusion Detection System
==============================================================
This module evaluates trained models on test data and provides detailed analysis.

Key features:
- Load trained models from disk
- Load test data
- Calculate evaluation metrics
- Generate confusion matrix and classification reports
- Compare model performance
- Visualize results
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_auc_score,
                            roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evaluator class for comprehensive model performance analysis.
    
    This class loads trained models and test data, then provides:
    - Accuracy, Precision, Recall, F1-Score metrics
    - Confusion Matrix analysis
    - Classification Reports
    - Detailed performance comparison
    """
    
    def __init__(self, data_folder="../data/processed", models_folder="../models"):
        """
        Initialize the evaluator.
        
        Args:
            data_folder (str): Path to processed data directory
            models_folder (str): Path to saved models directory
        """
        self.data_folder = data_folder
        self.models_folder = models_folder
        self.models = {}
        self.results = {}
        self.test_data = None
    
    
    def load_models(self):
        """
        Load all trained models from disk.
        
        Returns:
            dict: Dictionary of {model_name: model_object}
        """
        print("\n" + "="*80)
        print("📥 LOADING TRAINED MODELS")
        print("="*80 + "\n")
        
        if not os.path.exists(self.models_folder):
            print(f"❌ Models folder not found: {self.models_folder}")
            print(f"   Please run train.py first to train models")
            return False
        
        model_files = [f for f in os.listdir(self.models_folder) if f.endswith('.pkl')]
        
        if not model_files:
            print(f"❌ No trained models found in {self.models_folder}")
            print(f"   Please run train.py first to train models")
            return False
        
        for model_file in model_files:
            model_path = os.path.join(self.models_folder, model_file)
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    self.models[model_name] = model
                    print(f"✓ Loaded: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
        
        print(f"\n✓ Total models loaded: {len(self.models)}")
        return len(self.models) > 0
    
    
    def load_test_data(self, filename):
        """
        Load test data from preprocessed CSV files.
        
        Args:
            filename (str): Base filename (e.g., "friday" loads friday_X.csv and friday_y.csv)
        
        Returns:
            tuple: (X_test, y_test) or (None, None) if files not found
        """
        print("\n" + "="*80)
        print("📥 LOADING TEST DATA")
        print("="*80 + "\n")
        
        X_path = os.path.join(self.data_folder, f"{filename}_X.csv")
        y_path = os.path.join(self.data_folder, f"{filename}_y.csv")
        
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            print(f"❌ Test data files not found!")
            print(f"   Expected: {X_path}")
            print(f"   Expected: {y_path}")
            print(f"   Please run data_preprocessing.py first")
            return None, None
        
        X_test = pd.read_csv(X_path).values
        y_test = pd.read_csv(y_path).values.ravel()
        
        print(f"✓ Loaded test data from {filename}")
        print(f"   Features shape: {X_test.shape}")
        print(f"   Labels shape: {y_test.shape}")
        print(f"   Classes: {np.unique(y_test)}")
        
        self.test_data = (X_test, y_test)
        return X_test, y_test
    
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all loaded models on test data.
        
        Args:
            X_test (array): Test features
            y_test (array): True labels
        """
        print("\n" + "="*80)
        print("📊 EVALUATING MODELS")
        print("="*80 + "\n")
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store results
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1-Score:  {f1:.4f}")
            
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
    
    
    def display_comparison_table(self):
        """
        Display model comparison in a table format.
        """
        print("\n" + "="*80)
        print("📈 MODEL PERFORMANCE COMPARISON")
        print("="*80 + "\n")
        
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        comparison_df = pd.DataFrame(self.results).T[['accuracy', 'precision', 'recall', 'f1']]
        comparison_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        comparison_df['Acc %'] = (comparison_df['Accuracy'] * 100).round(2)
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(comparison_df.round(4).to_string())
        
        # Find best model
        best_model = comparison_df.index[0]
        best_acc = comparison_df.loc[best_model, 'Accuracy']
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        return best_model
    
    
    def show_confusion_matrix_analysis(self, y_test, model_name=None):
        """
        Display and analyze confusion matrix for a specific model.
        
        Args:
            y_test (array): True labels
            model_name (str): Name of model to analyze (best if None)
        """
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        # Use best model if not specified
        if model_name is None:
            model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        
        if model_name not in self.results:
            print(f"❌ Model {model_name} not found in results")
            return
        
        print("\n" + "="*80)
        print(f"🔍 CONFUSION MATRIX - {model_name.upper()}")
        print("="*80 + "\n")
        
        cm = self.results[model_name]['confusion_matrix']
        y_pred = self.results[model_name]['predictions']
        
        print("Confusion Matrix:")
        print("(Rows: Actual Class, Columns: Predicted Class)\n")
        print(cm)
        
        # Calculate metrics from confusion matrix
        print(f"\n📊 Analysis:")
        print(f"   True Positives (TP):   {cm[1,1]} - Correctly detected attacks")
        print(f"   True Negatives (TN):   {cm[0,0]} - Correctly identified benign")
        print(f"   False Positives (FP):  {cm[0,1]} - False alarms")
        print(f"   False Negatives (FN):  {cm[1,0]} - Missed attacks")
        
        # Calculate additional metrics
        total = cm.sum()
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"\n🎯 Key Metrics:")
        print(f"   Sensitivity (Detection Rate): {sensitivity*100:.2f}%")
        print(f"   Specificity (False Alarm Rate): {specificity*100:.2f}%")
    
    
    def show_detailed_report(self, y_test, model_name=None):
        """
        Show detailed classification report for a model.
        
        Args:
            y_test (array): True labels
            model_name (str): Model to analyze (best if None)
        """
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        if model_name is None:
            model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        
        print("\n" + "="*80)
        print(f"📋 DETAILED CLASSIFICATION REPORT - {model_name.upper()}")
        print("="*80 + "\n")
        
        y_pred = self.results[model_name]['predictions']
        
        # Determine number of classes dynamically
        num_classes = len(np.unique(y_test))
        possible_names = ['Benign', 'DDoS', 'DoS', 'Probe', 'R2L', 'U2R']
        # Use only the names that match the number of classes found
        target_names = possible_names[:num_classes]
        
        print(classification_report(y_test, y_pred, 
                                   target_names=target_names,
                                   zero_division=0))


def main():
    """
    Main function to execute model evaluation pipeline.
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MODEL EVALUATION PIPELINE" + " "*35 + "║")
    print("╚" + "="*78 + "╝")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    if not evaluator.load_models():
        print("\n❌ No models available for evaluation")
        print("   Please run train.py first to train models")
        return
    
    # Load test data
    X_test, y_test = evaluator.load_test_data("friday")
    if X_test is None:
        print("\n❌ Could not load test data")
        print("   Please run data_preprocessing.py first")
        return
    
    # Evaluate all models
    evaluator.evaluate_all_models(X_test, y_test)
    
    # Display comparison
    best_model = evaluator.display_comparison_table()
    
    # Show detailed analysis
    evaluator.show_confusion_matrix_analysis(y_test, best_model)
    evaluator.show_detailed_report(y_test, best_model)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

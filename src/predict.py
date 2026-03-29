"""
Model Prediction Module for Network Intrusion Detection System
==============================================================
This module makes predictions on new/unseen network traffic data using trained models.

Key features:
- Load trained models from disk
- Make predictions on new data
- Apply preprocessing transformations (scaling, encoding)
- Interpret predictions (convert back to class names)
- Batch prediction support
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionPredictor:
    """
    Predictor class for making inference on new network traffic data.
    
    This class:
    - Loads trained models
    - Applies preprocessing transformations
    - Makes predictions
    - Interprets results
    """
    
    def __init__(self, models_folder="../models", scaler_path=None, encoder_path=None):
        """
        Initialize the predictor with trained models and preprocessing objects.
        
        Args:
            models_folder (str): Path to saved models directory
            scaler_path (str): Path to saved StandardScaler object
            encoder_path (str): Path to saved LabelEncoder object
        """
        self.models_folder = models_folder
        self.models = {}
        self.scaler = None
        self.encoder = None
        self.best_model = None
        self.best_model_name = None
        
        self.load_models()
        if scaler_path:
            self.load_scaler(scaler_path)
        if encoder_path:
            self.load_encoder(encoder_path)
    
    
    def load_models(self):
        """
        Load all trained models from disk.
        
        Returns:
            bool: True if models loaded successfully
        """
        print("\n" + "="*80)
        print("📥 LOADING TRAINED MODELS FOR PREDICTION")
        print("="*80 + "\n")
        
        if not os.path.exists(self.models_folder):
            print(f"❌ Models folder not found: {self.models_folder}")
            return False
        
        model_files = [f for f in os.listdir(self.models_folder) if f.endswith('.pkl')]
        
        if not model_files:
            print(f"❌ No trained models found in {self.models_folder}")
            return False
        
        for model_file in model_files:
            model_path = os.path.join(self.models_folder, model_file)
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    self.models[model_name] = model
                    print(f"✓ Loaded: {model_name}")
                    
                    # Use first loaded model as default
                    if self.best_model is None:
                        self.best_model = model
                        self.best_model_name = model_name
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
        
        print(f"\n✓ Total models loaded: {len(self.models)}")
        return len(self.models) > 0
    
    
    def load_scaler(self, scaler_path):
        """
        Load StandardScaler object for feature normalization.
        
        Args:
            scaler_path (str): Path to saved scaler pickle file
        """
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                print(f"✓ Loaded StandardScaler from {scaler_path}")
        except Exception as e:
            print(f"✗ Failed to load scaler: {str(e)}")
    
    
    def load_encoder(self, encoder_path):
        """
        Load LabelEncoder object for converting predictions back to class names.
        
        Args:
            encoder_path (str): Path to saved encoder pickle file
        """
        try:
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
                print(f"✓ Loaded LabelEncoder from {encoder_path}")
        except Exception as e:
            print(f"✗ Failed to load encoder: {str(e)}")
    
    
    def preprocess_features(self, X):
        """
        Apply feature scaling to new data using saved scaler.
        
        Args:
            X (array or DataFrame): Input features (unscaled)
        
        Returns:
            array: Scaled features
        """
        if self.scaler is None:
            print("⚠️  No scaler loaded - returning features as-is")
            return X
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    
    def predict(self, X, model_name=None, return_probabilities=False):
        """
        Make predictions on new data.
        
        Args:
            X (array or DataFrame): Input features
            model_name (str): Name of model to use (default: best model)
            return_probabilities (bool): Return prediction probabilities if supported
        
        Returns:
            array: Predicted class labels (numeric)
        """
        # Use best model if not specified
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                print(f"❌ Model '{model_name}' not found")
                return None
            model = self.models[model_name]
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Make predictions
        y_pred = model.predict(X_processed)
        
        return y_pred
    
    
    def predict_with_probabilities(self, X, model_name=None):
        """
        Make predictions with probabilities (when supported).
        
        Args:
            X (array or DataFrame): Input features
            model_name (str): Name of model to use
        
        Returns:
            tuple: (predictions, probabilities)
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                print(f"❌ Model '{model_name}' not found")
                return None, None
            model = self.models[model_name]
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Make predictions
        y_pred = model.predict(X_processed)
        
        # Get probabilities if available
        try:
            y_proba = model.predict_proba(X_processed)
            return y_pred, y_proba
        except AttributeError:
            print(f"⚠️  Model '{model_name}' does not support probability predictions")
            return y_pred, None
    
    
    def decode_predictions(self, y_pred):
        """
        Convert numeric predictions back to class names.
        
        Args:
            y_pred (array): Numeric predictions
        
        Returns:
            array: Class names
        """
        if self.encoder is None:
            print("⚠️  No encoder loaded - returning numeric predictions")
            return y_pred
        
        try:
            y_decoded = self.encoder.inverse_transform(y_pred.astype(int))
            return y_decoded
        except Exception as e:
            print(f"⚠️  Could not decode predictions: {str(e)}")
            return y_pred
    
    
    def predict_single_sample(self, X_single, model_name=None):
        """
        Make prediction on a single sample with detailed output.
        
        Args:
            X_single (array or list): Single sample features
            model_name (str): Model to use
        
        Returns:
            dict: Prediction details
        """
        if len(X_single.shape) == 1:
            X_single = X_single.reshape(1, -1)
        
        # Preprocess
        X_processed = self.preprocess_features(X_single)
        
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                return None
            model = self.models[model_name]
        
        # Predict
        y_pred = model.predict(X_processed)[0]
        
        # Get probability if available
        y_proba = None
        try:
            y_proba = model.predict_proba(X_processed)[0]
        except:
            pass
        
        # Decode label
        y_label = self.decode_predictions(np.array([y_pred]))[0]
        
        result = {
            'model': model_name,
            'prediction_numeric': y_pred,
            'prediction_label': y_label,
            'probabilities': y_proba
        }
        
        return result
    
    
    def predict_batch(self, X, model_name=None):
        """
        Make predictions on multiple samples.
        
        Args:
            X (array or DataFrame): Multiple samples
            model_name (str): Model to use
        
        Returns:
            dict: Dictionary with predictions and labels
        """
        # Make predictions
        y_pred_numeric = self.predict(X, model_name)
        
        if y_pred_numeric is None:
            return None
        
        # Decode to labels
        y_pred_labels = self.decode_predictions(y_pred_numeric)
        
        result = {
            'predictions_numeric': y_pred_numeric,
            'predictions_labels': y_pred_labels,
            'count': len(y_pred_numeric),
            'unique_classes': np.unique(y_pred_labels)
        }
        
        return result
    
    
    def display_prediction(self, result):
        """
        Display prediction result in a readable format.
        
        Args:
            result (dict): Prediction result from predict_single_sample
        """
        if result is None:
            print("❌ No prediction result")
            return
        
        print("\n" + "="*80)
        print("🔮 PREDICTION RESULT")
        print("="*80 + "\n")
        
        print(f"Model Used: {result['model']}")
        print(f"Predicted Class: {result['prediction_label']}")
        print(f"Prediction Code: {int(result['prediction_numeric'])}")
        
        if result['probabilities'] is not None:
            print(f"\nClass Probabilities:")
            class_names = ['Benign', 'DDoS', 'DoS', 'Probe', 'R2L', 'U2R']
            for i, prob in enumerate(result['probabilities']):
                if i < len(class_names):
                    print(f"  {class_names[i]}: {prob*100:.2f}%")


def demo_prediction():
    """
    Demo function showing how to use the predictor with sample data.
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "MODEL PREDICTION DEMO WITH SAMPLE DATA" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n" + "="*80)
    print("DEMO: Making Predictions on Sample Network Traffic")
    print("="*80)
    
    # Create sample network traffic data
    print("\n📊 Creating sample network traffic data...")
    
    sample_features = np.array([
        [-0.5, -0.3, 0.2, -0.1, 0.4],    # Sample 1: Benign-like
        [1.2, 1.5, 1.1, 0.8, 1.3],       # Sample 2: Attack-like
        [-0.2, 0.1, -0.3, 0.2, -0.1],    # Sample 3: Benign-like
    ])
    
    df_sample = pd.DataFrame(sample_features, 
                            columns=[f'Feature_{i+1}' for i in range(5)])
    
    print("\nSample Network Data (Scaled Features):")
    print(df_sample.round(2).to_string())
    
    print("\n💡 Features represent network properties:")
    print("   - Feature_1-5: Network metrics (bytes, packets, duration, etc.)")
    print("\n📌 How predictions work:")
    print("   1. Features are already scaled (standardized)")
    print("   2. Trained model analyzes the patterns")
    print("   3. Model predicts: Normal traffic or Attack")
    print("   4. Returns confidence/probability")
    
    print("\n" + "="*80)
    print("✨ PREDICTION WORKFLOW")
    print("="*80)
    
    print("\n1️⃣ LOAD MODELS")
    print("   predictor = IntrusionDetectionPredictor()")
    
    print("\n2️⃣ LOAD PREPROCESSING OBJECTS")
    print("   predictor.load_scaler('models/friday_scaler.pkl')")
    print("   predictor.load_encoder('models/friday_label_encoder.pkl')")
    
    print("\n3️⃣ MAKE PREDICTIONS")
    print("   predictions = predictor.predict(new_data)")
    
    print("\n4️⃣ CONVERT TO CLASS NAMES")
    print("   labels = predictor.decode_predictions(predictions)")
    
    print("\n" + "="*80)
    print("✨ DEMO COMPLETE ✨")
    print("="*80 + "\n")


def main():
    """
    Main function to demonstrate and execute predictions.
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*18 + "INTRUSION DETECTION - PREDICTION MODULE" + " "*23 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # First, show the demo
    demo_prediction()
    
    # Try to initialize predictor with models AND preprocessing objects
    print("\n" + "="*80)
    print("🔍 CHECKING FOR TRAINED MODELS AND PREPROCESSING OBJECTS")
    print("="*80 + "\n")
    
    # Initialize predictor with paths to scaler and encoder
    # Note: Scaler and encoder are saved in ../data/processed/ by data_preprocessing.py
    predictor = IntrusionDetectionPredictor(
        models_folder='../models',
        scaler_path='../data/processed/friday_scaler.pkl',
        encoder_path='../data/processed/friday_label_encoder.pkl'
    )
    
    if not predictor.models:
        print("\n⚠️  No trained models found!")
        print("\n📋 To make predictions, follow these steps:")
        print("   1. python3 data_preprocessing.py")
        print("   2. python3 train.py")
        print("   3. python3 evaluate.py")
        print("   4. python3 predict.py")
        print("\n💡 Then predictions will work with real data!")
        return
    
    # Check if preprocessing objects loaded
    if not predictor.scaler:
        print("\n⚠️  Warning: Scaler not loaded!")
        print("   Models exist but preprocessing objects missing.")
        print("   Predictions may not work correctly without scaler.")
        print("\n📋 Make sure preprocessing files exist:")
        print("   - ../models/friday_scaler.pkl")
        print("   - ../models/friday_label_encoder.pkl")
        print("\n💡 These are created by data_preprocessing.py")
        return
    
    # Models and preprocessing objects found!
    print("✓ Models loaded successfully")
    print("✓ Scaler loaded successfully")
    if predictor.encoder:
        print("✓ Encoder loaded successfully")
    
    # Make predictions with real test data
    print("\n" + "="*80)
    print("📊 MAKING PREDICTIONS WITH REAL TEST DATA")
    print("="*80 + "\n")
    
    try:
        # Load real test data from preprocessed files
        import pandas as pd
        
        test_X_path = '../data/processed/friday_X.csv'
        test_y_path = '../data/processed/friday_y.csv'
        
        # Try to load real test data
        try:
            X_test = pd.read_csv(test_X_path).values
            y_test = pd.read_csv(test_y_path).values.ravel()
            
            print(f"✓ Loaded test data from {test_X_path}")
            print(f"  Shape: {X_test.shape}")
            print(f"  Features: {X_test.shape[1]}")
            print(f"  Samples: {X_test.shape[0]}\n")
            
            # Make predictions on first 3 samples
            print(f"Making predictions on first 3 samples...\n")
            
            for idx in range(min(3, len(X_test))):
                sample = X_test[idx]
                actual_label = y_test[idx]
                
                result = predictor.predict_single_sample(sample)
                
                if result:
                    print(f"Sample {idx + 1}:")
                    print(f"  Actual: {actual_label}")
                    print(f"  Predicted: {result['prediction_label']}")
                    print(f"  Code: {int(result['prediction_numeric'])}")
                    if result['probabilities'] is not None:
                        max_prob = result['probabilities'].max()
                        print(f"  Confidence: {max_prob*100:.2f}%")
                    print()
        
        except FileNotFoundError:
            print(f"⚠️  Test data not found at {test_X_path}")
            print("   Creating simple synthetic sample for demonstration...\n")
            
            # Create sample with correct number of features
            num_features = X_test.shape[1] if 'X_test' in locals() else 87
            
            sample_data = np.random.randn(2, num_features)
            
            print(f"Created synthetic data with {num_features} features\n")
            print("Making predictions...\n")
            
            for idx, sample in enumerate(sample_data, 1):
                result = predictor.predict_single_sample(sample)
                
                if result:
                    print(f"Sample {idx}:")
                    print(f"  Predicted: {result['prediction_label']}")
                    print(f"  Code: {int(result['prediction_numeric'])}")
                    if result['probabilities'] is not None:
                        max_prob = result['probabilities'].max()
                        print(f"  Confidence: {max_prob*100:.2f}%")
                    print()
    
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        print("\n💡 Make sure:")
        print("   1. Test data exists in ../data/processed/")
        print("   2. Models exist in ../models/")
        print("   3. Preprocessing objects exist in ../models/")
        return
    
    print("="*80)
    print("✅ PREDICTION DEMO COMPLETE")
    print("="*80)
    print("\n📌 To use predictor in your code:")
    print("""
   from predict import IntrusionDetectionPredictor
   
   predictor = IntrusionDetectionPredictor(
       models_folder='../models',
       scaler_path='../models/friday_scaler.pkl',
       encoder_path='../models/friday_label_encoder.pkl'
   )
   
   result = predictor.predict_single_sample(new_data)
   predictor.display_prediction(result)
    """)


if __name__ == "__main__":
    main()

"""
Model Training Module for Network Intrusion Detection System
=============================================================
This module handles training multiple machine learning models on preprocessed data.

It includes:
- Loading preprocessed data from CSV files
- Splitting data into training and testing sets
- Training multiple classification models
- Evaluating model performance with various metrics
- Saving trained models for later predictions
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


class IntrusionDetectionTrainer:
    """
    Trainer class for building and evaluating intrusion detection models.
    
    This class handles the complete training workflow:
    1. Load preprocessed data
    2. Split into training and testing sets
    3. Train multiple models
    4. Evaluate and compare performance
    5. Save best models
    """
    
    def __init__(self, data_folder="../data/processed", models_folder="../models"):
        """
        Initialize the trainer with data and model folders.
        
        Args:
            data_folder (str): Path to processed data directory
            models_folder (str): Path to save trained models
        """
        self.data_folder = data_folder
        self.models_folder = models_folder
        self.models = {}
        self.results = {}
        
        # Create models folder if it doesn't exist
        Path(self.models_folder).mkdir(parents=True, exist_ok=True)
    
    
    def load_data(self, filename):
        """
        Load preprocessed features and labels from CSV files.
        
        Args:
            filename (str): Base filename (without suffix)
                          e.g., "monday" loads "monday_X.csv" and "monday_y.csv"
        
        Returns:
            tuple: (X, y) - Features and labels as numpy arrays
        """
        X_path = os.path.join(self.data_folder, f"{filename}_X.csv")
        y_path = os.path.join(self.data_folder, f"{filename}_y.csv")
        
        # Check if files exist
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            print(f"⚠️  Processed data files not found!")
            print(f"   Expected: {X_path}")
            print(f"   Expected: {y_path}")
            print(f"\n💡 Please run data preprocessing first:")
            print(f"   python3 data_preprocessing.py")
            print(f"\n   Or use demo_train.py to see training with sample data:")
            print(f"   python3 demo_train.py")
            raise FileNotFoundError(f"Processed data files not found. Run data_preprocessing.py first.")
        
        # Load feature data
        X = pd.read_csv(X_path).values
        
        # Load label data and flatten from 2D to 1D array
        y = pd.read_csv(y_path).values.ravel()
        
        print(f"📂 Loaded data from {filename}")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        return X, y
    
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Training set: Used to teach the model patterns (80%)
        Testing set: Used to evaluate model performance on unseen data (20%)
        
        Args:
            X (array): Features
            y (array): Labels
            test_size (float): Proportion of data for testing (default 20%)
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintains class distribution in both sets
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {X_train.shape[0]} ({100*(1-test_size):.0f}%)")
        print(f"   Testing samples: {X_test.shape[0]} ({100*test_size:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    
    def train_models(self, X_train, y_train):
        """
        Train multiple machine learning models.
        
        Models trained:
        1. Logistic Regression - Fast, linear decision boundaries
        2. Random Forest - Ensemble of decision trees, handles non-linearity
        3. Support Vector Machine (SVM) - Good for complex patterns
        4. Naive Bayes - Fast, probabilistic classifier
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
        """
        print(f"\n{'='*80}")
        print("🤖 TRAINING MODELS")
        print(f"{'='*80}")
        
        # Model 1: Logistic Regression
        print(f"\n1️⃣ Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        print(f"   ✓ Training complete")
        
        # Model 2: Random Forest
        print(f"\n2️⃣ Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print(f"   ✓ Training complete (100 trees)")
        
        # Model 3: Support Vector Machine
        print(f"\n3️⃣ Training Support Vector Machine (SVM)...")
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train, y_train)
        self.models['SVM'] = svm
        print(f"   ✓ Training complete (RBF kernel)")
        
        # Model 4: Naive Bayes
        print(f"\n4️⃣ Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        self.models['Naive Bayes'] = nb
        print(f"   ✓ Training complete")
    
    
    def save_models(self):
        """
        Save all trained models to disk using pickle.
        
        These models can be loaded later by evaluate.py and predict.py
        for evaluation and making predictions on new data.
        """
        print(f"\n{'='*80}")
        print("💾 SAVING TRAINED MODELS")
        print(f"{'='*80}\n")
        
        for model_name, model in self.models.items():
            filepath = os.path.join(self.models_folder, f"{model_name.lower().replace(' ', '_')}.pkl")
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✓ Saved: {model_name}")
            print(f"  Location: {filepath}")
    


def main():
    """
    Main function to execute the TRAINING ONLY pipeline.
    
    NOTE: Model evaluation is handled separately in evaluate.py
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "MODEL TRAINING PIPELINE (Training Only)" + " "*25 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Initialize trainer
    trainer = IntrusionDetectionTrainer()
    
    try:
        # Load data
        print("="*80)
        print("📥 LOADING DATA")
        print("="*80)
        X, y = trainer.load_data("friday")  # Using friday.csv as example
        
        # Split data - we need test data for evaluation later
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Train models
        trainer.train_models(X_train, y_train)
        
        # Save models
        trainer.save_models()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n📌 Next Steps:")
        print(f"   1. ✓ Models trained and saved in: ../models/")
        print(f"   2. → Evaluate models: python3 evaluate.py")
        print(f"   3. → Make predictions: python3 predict.py")
        print()
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"\n{'='*80}")
        print("📋 QUICK START GUIDE")
        print(f"{'='*80}")
        print(f"\n1️⃣ First, preprocess all raw data files:")
        print(f"   python3 data_preprocessing.py")
        print(f"\n2️⃣ Then, train the models:")
        print(f"   python3 train.py")
        print(f"\n3️⃣ Then, evaluate the models:")
        print(f"   python3 evaluate.py")
        print(f"\n4️⃣ Or, see a demo with sample data:")
        print(f"   python3 demo_train.py")
        print()
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()

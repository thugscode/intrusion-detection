# Network Intrusion Detection System (IDS)

A machine learning-based Network Intrusion Detection System that classifies network traffic as benign or malicious attacks using multiple algorithms.

## Project Overview

This project implements a complete ML pipeline for intrusion detection:
- **Data Preprocessing**: Clean and normalize network traffic data
- **Model Training**: Train 4 different ML algorithms
- **Evaluation**: Comprehensive model performance analysis
- **Prediction**: Make predictions on new network traffic

## Dataset

The system classifies network traffic into 6 categories:
- **Benign**: Normal traffic
- **DoS**: Denial of Service attacks
- **DDoS**: Distributed Denial of Service attacks
- **Probe**: Reconnaissance attacks
- **R2L**: Remote to Local attacks
- **U2R**: User to Root attacks

Each sample contains 87 network features extracted from traffic flows.

## Models Trained

1. **Logistic Regression** - Fast, interpretable linear model
2. **Random Forest** - Ensemble with 100 decision trees
3. **Support Vector Machine (SVM)** - Non-linear kernel (RBF)
4. **Naive Bayes** - Probabilistic classifier

## Project Structure

```
intrusion-detection/
├── data/
│   ├── raw/              # Original CSV files
│   └── processed/        # Cleaned and normalized data
├── models/               # Trained ML models (pickle files)
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── data_preprocessing.py    # Data cleaning and normalization
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation and metrics
│   ├── predict.py               # Prediction on new data
│   ├── demo_preprocessing.py    # Demo: Show preprocessing steps
│   └── demo_train.py            # Demo: Show training process
├── requirements.txt      # Python dependencies
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intrusion-detection.git
cd intrusion-detection
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Data
Clean and normalize raw CSV files:
```bash
cd src
python3 data_preprocessing.py
```

Output files saved to `data/processed/`:
- `{filename}_X.csv` - Features
- `{filename}_y.csv` - Labels
- `{filename}_scaler.pkl` - StandardScaler object
- `{filename}_label_encoder.pkl` - LabelEncoder object

### 2. Train Models
Train all 4 models on preprocessed data:
```bash
python3 train.py
```

Models saved to `models/`:
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`
- `naive_bayes_model.pkl`

### 3. Evaluate Models
Evaluate trained models and compare performance:
```bash
python3 evaluate.py
```

Displays:
- Accuracy, Precision, Recall, F1-Score for each model
- Confusion matrices
- Detailed classification reports

### 4. Make Predictions
Predict on new network traffic data:
```bash
python3 predict.py
```

## Demo Scripts

### See Data Preprocessing Steps (6 rows)
```bash
python3 demo_preprocessing.py
```

Shows complete transformation:
1. Original raw data
2. After removing duplicates/NaN
3. After removing irrelevant columns
4. After label encoding
5. After feature scaling

### See Training Process (50 rows)
```bash
python3 demo_train.py
```

Shows all 4 models training and evaluation metrics.

## Quick Start Example

```python
from predict import IntrusionDetectionPredictor
import pandas as pd

# Initialize predictor
predictor = IntrusionDetectionPredictor(
    models_folder='../models',
    scaler_path='../data/processed/friday_scaler.pkl',
    encoder_path='../data/processed/friday_label_encoder.pkl'
)

# Load test data (87 features)
X_test = pd.read_csv('../data/processed/friday_X.csv').values

# Make prediction
result = predictor.predict_single_sample(X_test[0])
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## File Descriptions

### data_preprocessing.py
- Loads raw CSV files from `data/raw/`
- Removes duplicates, NaN, infinite values
- Encodes text labels to numeric
- Scales features using StandardScaler
- Saves processed data and preprocessing objects

**Key Functions:**
- `process_all_files()` - Batch process all CSV files
- `preprocess_pipeline()` - Single file preprocessing

### train.py
- Loads preprocessed data
- Splits into 80% train, 20% test (stratified)
- Trains 4 different ML models
- Saves models to pickle files

**Key Class:**
- `IntrusionDetectionTrainer` - Handles training pipeline

### evaluate.py
- Loads trained models and test data
- Calculates accuracy, precision, recall, F1-score
- Displays confusion matrix analysis
- Generates detailed classification reports

**Key Class:**
- `ModelEvaluator` - Handles evaluation pipeline

### predict.py
- Loads trained models and preprocessing objects
- Makes predictions on new data
- Supports single sample and batch predictions
- Returns predictions with confidence scores

**Key Class:**
- `IntrusionDetectionPredictor` - Handles inference pipeline

## Performance Metrics

The system evaluates models using:
- **Accuracy** - Overall correctness
- **Precision** - True positives among predicted positives
- **Recall** - True positives among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Breakdown of TP, TN, FP, FN

## Data Flow

```
Raw CSV Files (87 features)
        ↓
data_preprocessing.py
        ↓
Processed Data + Scaler + Encoder
        ↓
train.py
        ↓
Trained Models (4 algorithms)
        ↓
evaluate.py → Model Metrics
predict.py  → Predictions
```

## Dependencies

- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **scikit-learn** - ML algorithms, preprocessing, metrics

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created as part of a Network Intrusion Detection research project.

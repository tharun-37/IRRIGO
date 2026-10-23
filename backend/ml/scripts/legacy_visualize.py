import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Ignore seaborn future warnings for cleaner terminal output
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. Load the Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_outputs', 'rf_model.joblib')
bundle = joblib.load(MODEL_PATH)

classifier = bundle['classifier']
regressor = bundle['regressor']
label_encoder = bundle['label_encoder']

FEATURES = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'PH', 'Light_Intensity']

# ==========================================
# GRAPH 1: Feature Importance 
# ==========================================
plt.figure(figsize=(10, 6))
importances = classifier.feature_importances_
# Updated to avoid the hue warning
sns.barplot(x=importances, y=FEATURES, hue=FEATURES, palette='viridis', legend=False)
plt.title('Sensor Importance for Disease Classification', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Sensors')
plt.tight_layout()
plt.savefig('Slide_Feature_Importance.png')
print("Feature Importance graph saved!")

# ==========================================
# GRAPH 2 & 3: Using your specific dataset
# ==========================================
try:
    # Read the dataset you provided
    df = pd.read_csv('training_dataset_with_yield.csv') 
    
    X = df[FEATURES]
    # Updated column names based on your CSV
    y_class = df['Target_Disease'] 
    y_yield = df['Yield_Rate']         
    
    # Split the data just like it was during training
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_yield, y_test_yield = train_test_split(X, y_yield, test_size=0.2, random_state=42)

    # Scale the test data
    X_test_scaled = bundle['scaler'].transform(X_test)
    
# --- Graph 2: Confusion Matrix ---
    y_pred_class = classifier.predict(X_test_scaled)
    
    # NEW LINE: Convert the string labels from the CSV into numbers so they match the model's predictions
    y_test_class_encoded = label_encoder.transform(y_test_class)
    
    # Use the encoded true labels for the matrix
    cm = confusion_matrix(y_test_class_encoded, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Disease Classification Accuracy (Confusion Matrix)')
    plt.xlabel('Predicted Disease')
    plt.ylabel('Actual Disease')
    plt.tight_layout()
    plt.savefig('Slide_Confusion_Matrix.png')
    print("Confusion Matrix saved!")

    # --- Graph 3: Actual vs Predicted Yield ---
    y_pred_yield = regressor.predict(X_test_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_yield, y_pred_yield, alpha=0.6, color='green')
    
    # Draw the perfect prediction line
    min_val = min(y_test_yield.min(), y_pred_yield.min())
    max_val = max(y_test_yield.max(), y_pred_yield.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Yield Prediction: Actual vs Predicted')
    plt.xlabel('Actual Yield Rate')
    plt.ylabel('Predicted Yield Rate')
    plt.tight_layout()
    plt.savefig('Slide_Yield_Regression.png')
    print("Yield Regression scatter plot saved!")

except FileNotFoundError:
    print("Error: Make sure 'training_dataset_with_yield.csv' is in the same folder as this script!")
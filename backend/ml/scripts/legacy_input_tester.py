import os
import json
import joblib
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_outputs', 'rf_model.joblib')
THRESHOLDS_PATH = os.path.join(BASE_DIR, 'selected_thresholds.json')

FEATURES = [
    'Temperature', 'Humidity', 'Moisture', 'Nitrogen',
    'Phosphorus', 'Potassium', 'PH', 'Light_Intensity'
]

INPUT_RANGES = {
    'Temperature': {'ideal': (20.0, 30.0), 'label': '°C'},
    'Humidity': {'ideal': (55.0, 85.0), 'label': '%'},
    'Moisture': {'ideal': (40.0, 70.0), 'label': '%'},
    'Nitrogen': {'ideal': (30.0, 70.0), 'label': 'mg/kg'},
    'Phosphorus': {'ideal': (20.0, 50.0), 'label': 'mg/kg'},
    'Potassium': {'ideal': (20.0, 60.0), 'label': 'mg/kg'},
    'PH': {'ideal': (6.0, 7.5), 'label': 'pH'},
    'Light_Intensity': {'ideal': (400.0, 800.0), 'label': 'lux'},
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'Model file not found at {MODEL_PATH}. Run improved_model.py first to train and save it.'
        )

    bundle = joblib.load(MODEL_PATH)
    return bundle['classifier'], bundle['regressor'], bundle['scaler'], bundle['label_encoder']


def load_thresholds():
    if not os.path.exists(THRESHOLDS_PATH):
        return None
    with open(THRESHOLDS_PATH, 'r') as handle:
        return json.load(handle)


def build_feature_row(values):
    row = pd.Series(values, dtype=float)
    missing = [feature for feature in FEATURES if feature not in row.index]
    if missing:
        raise ValueError(f'Missing required feature(s): {missing}')
    ordered = row[FEATURES]
    ordered = ordered.fillna(ordered.median())
    return ordered


def prompt_for_values():
    values = {}
    print('Enter numeric values for each feature:')
    for feature in FEATURES:
        config = INPUT_RANGES.get(feature, {})
        low, high = config.get('ideal', (None, None))
        label = config.get('label', '')
        prompt = f"{feature}"
        if low is not None and high is not None:
            prompt += f" ({low}-{high} {label})"
        prompt += ': '
        while True:
            raw_value = input(prompt).strip()
            try:
                values[feature] = float(raw_value)
                break
            except ValueError:
                print('Please enter a valid number.')
    return values


def summarize_inputs(values):
    summary = {}
    for feature, config in INPUT_RANGES.items():
        value = float(values[feature])
        low, high = config['ideal']
        if low <= value <= high:
            status = 'ideal'
        elif value < low:
            status = 'low'
        else:
            status = 'high'
        summary[feature] = {
            'value': value,
            'ideal_range': [low, high],
            'unit': config['label'],
            'status': status,
        }
    return summary


def predict_disease_and_yield(values):
    classifier, regressor, scaler, label_encoder = load_model()
    row = build_feature_row(values)
    frame = pd.DataFrame([row])
    scaled = scaler.transform(frame)

    predicted_index = classifier.predict(scaled)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    yield_prediction = float(regressor.predict(scaled)[0])

    return {
        'prediction': predicted_label,
        'yield_prediction': round(yield_prediction, 2),
        'probabilities': {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, classifier.predict_proba(scaled)[0])
        },
    }


def predict_disease(values):
    return predict_disease_and_yield(values)


if __name__ == '__main__':
    values = prompt_for_values()
    result = predict_disease(values)
    print(json.dumps(result, indent=2))

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Default fallback values for testing / demo mode
DEFAULT_INPUTS = {
    'Temperature': 25.0,        # Celsius (deg C)
    'Humidity': 75.0,           # Percentage (%)
    'Moisture': 55.0,           # Soil Moisture (%)
    'Nitrogen': 45.0,           # Soil Nitrogen (mg/kg)
    'Phosphorus': 30.0,         # Soil Phosphorus (mg/kg)
    'Potassium': 35.0,          # Soil Potassium (mg/kg)
    'PH': 6.5,                  # Soil pH
    'Light_Intensity': 600.0,   # Lux
    'Field_Area_m2': 100.0,     # Field Area in Square Meters (m²)
}

FEATURE_ORDER = [
    'Temperature', 'Humidity', 'Moisture', 'Nitrogen',
    'Phosphorus', 'Potassium', 'PH', 'Light_Intensity', 'Yield_Rate'
]

BASE_SENSOR_FEATURES = [
    'Temperature', 'Humidity', 'Moisture', 'Nitrogen',
    'Phosphorus', 'Potassium', 'PH', 'Light_Intensity'
]

FEATURE_BOUNDS = {
    'Temperature': (10.0, 50.0, "deg C"),
    'Humidity': (10.0, 100.0, "%"),
    'Moisture': (5.0, 95.0, "%"),
    'Nitrogen': (0.0, 150.0, "mg/kg"),
    'Phosphorus': (0.0, 120.0, "mg/kg"),
    'Potassium': (0.0, 120.0, "mg/kg"),
    'PH': (3.0, 10.0, ""),
    'Light_Intensity': (50.0, 1500.0, "Lux"),
}


def calculate_yield_rate(inputs):
    """
    Calculates crop Yield Rate score (10 - 100) based on optimal sensor ranges.
    Score formula evaluates soil nutrient balance, moisture, pH, and climate conditions.
    """
    def normalize(val, lo, hi):
        return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    n_norm = normalize(inputs['Nitrogen'], 10.0, 100.0)
    p_norm = normalize(inputs['Phosphorus'], 10.0, 80.0)
    k_norm = normalize(inputs['Potassium'], 10.0, 80.0)
    moist_norm = normalize(inputs['Moisture'], 20.0, 70.0)
    ph_norm = normalize(inputs['PH'], 5.0, 8.0)
    light_norm = normalize(inputs['Light_Intensity'], 200.0, 1000.0)
    hum_norm = normalize(inputs['Humidity'], 30.0, 90.0)

    score = (
        0.20 * n_norm +
        0.15 * p_norm +
        0.15 * k_norm +
        0.20 * moist_norm +
        0.15 * ph_norm +
        0.10 * light_norm +
        0.05 * hum_norm
    )

    yield_rate = 20.0 + 60.0 * score
    return float(np.clip(round(yield_rate, 2), 10.0, 100.0))


def calculate_water_requirement_formula(inputs):
    """
    Calculates required irrigation water supply in Liters/m² (mm equivalent).
    Evaluates soil moisture deficit from target field capacity (65%) and evapotranspiration (ET0).
    """
    moist_deficit = max(0.0, 65.0 - inputs['Moisture'])
    temp_factor = max(0.0, inputs['Temperature'] - 10.0)
    hum_factor = 1.0 + (100.0 - np.clip(inputs['Humidity'], 10.0, 100.0)) / 100.0
    light_factor = np.clip(inputs['Light_Intensity'], 100.0, 1200.0) / 500.0
    
    et0 = 0.08 * temp_factor * hum_factor * light_factor
    water_req = 0.45 * moist_deficit + et0
    return float(np.clip(round(water_req, 2), 0.0, 50.0))


def load_model(model_path):
    """Loads the trained Random Forest classifier, water regressor, scaler, and label encoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run 'improved_model.py' first to train and export the model."
        )
    artifact = joblib.load(model_path)
    model = artifact.get('model')
    water_reg = artifact.get('water_regressor')
    scaler = artifact.get('scaler')
    label_encoder = artifact.get('label_encoder')
    return model, water_reg, scaler, label_encoder


def get_user_inputs(interactive=True):
    """Prompts the user for sensor values or returns default sample inputs."""
    if not interactive:
        print("\n[INFO] Running in Automated Demo Mode with default inputs.\n")
        return DEFAULT_INPUTS.copy()

    print("Please enter the requested sensor metrics:\n")
    user_inputs = {}

    for feature, (min_v, max_v, unit) in FEATURE_BOUNDS.items():
        default_val = DEFAULT_INPUTS[feature]
        unit_str = f" in {unit}" if unit else ""
        prompt_text = f"{feature}{unit_str}: "

        while True:
            raw_val = input(prompt_text).strip()
            if raw_val == "":
                user_inputs[feature] = default_val
                break
            try:
                val = float(raw_val)
                user_inputs[feature] = val
                break
            except ValueError:
                print("    [!] Invalid input. Please enter a numerical value.")

    # Prompt for Field Area
    default_area = DEFAULT_INPUTS['Field_Area_m2']
    area_prompt = "Field Area in Square Meters (m^2): "
    while True:
        raw_val = input(area_prompt).strip()
        if raw_val == "":
            user_inputs['Field_Area_m2'] = default_area
            break
        try:
            val = float(raw_val)
            if val <= 0:
                print("    [!] Field area must be greater than 0.")
                continue
            user_inputs['Field_Area_m2'] = val
            break
        except ValueError:
            print("    [!] Invalid input. Please enter a numerical value.")

    return user_inputs


def predict(inputs, model, water_regressor, scaler, label_encoder):
    """Computes yield rate, water requirement, and predicts disease classification."""
    yield_rate = calculate_yield_rate(inputs)
    inputs_with_yield = inputs.copy()
    inputs_with_yield['Yield_Rate'] = yield_rate

    base_df = pd.DataFrame([inputs])[BASE_SENSOR_FEATURES]
    if water_regressor is not None:
        predicted_water_m2 = float(water_regressor.predict(base_df)[0])
        predicted_water_m2 = float(np.clip(round(predicted_water_m2, 2), 0.0, 50.0))
    else:
        predicted_water_m2 = calculate_water_requirement_formula(inputs)

    input_df = pd.DataFrame([inputs_with_yield])[FEATURE_ORDER]
    input_scaled = scaler.transform(input_df)

    pred_class_idx = model.predict(input_scaled)[0]
    predicted_disease = label_encoder.inverse_transform([pred_class_idx])[0]
    probabilities = model.predict_proba(input_scaled)[0]

    prob_dict = {
        label_encoder.inverse_transform([idx])[0]: round(prob * 100, 2)
        for idx, prob in enumerate(probabilities)
    }

    return yield_rate, predicted_water_m2, predicted_disease, prob_dict, inputs_with_yield


def display_results(yield_rate, water_m2, predicted_disease, probabilities, inputs, show_inputs=False):
    """Prints clean minimal output without heavy headings."""
    field_area = inputs.get('Field_Area_m2', 100.0)
    total_liters = water_m2 * field_area

    # Only show inputs list in demo mode
    if show_inputs:
        print("inputs :")
        for feat in BASE_SENSOR_FEATURES:
            val = inputs[feat]
            unit = FEATURE_BOUNDS.get(feat, (None, None, ""))[2]
            unit_str = f" {unit}" if unit else ""
            print(f"  - {feat}: {val}{unit_str}")
        print(f"  - Field_Area: {field_area} m^2\n")
    else:
        print()

    # Final water needed
    print(f"water needed : {total_liters:,.2f} Liters total ({water_m2:.2f} Liters / m^2)")

    # Yield rate assessment
    if yield_rate >= 75:
        quality = "Excellent"
    elif yield_rate >= 50:
        quality = "Moderate"
    else:
        quality = "Poor"
    print(f"yield rate : {yield_rate:.2f} / 100.0 ({quality})")

    # Disease / health issue diagnosis
    top_prob = probabilities.get(predicted_disease, 0.0)
    if predicted_disease.upper() == 'HEALTHY':
        health_str = f"HEALTHY (Confidence: {top_prob:.2f}%)"
    else:
        health_str = f"{predicted_disease.upper()} (Confidence: {top_prob:.2f}%)"
    print(f"possible health issues : {health_str}\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model_outputs', 'rf_model.joblib')

    is_interactive = True
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--demo', '-d', '--non-interactive']:
        is_interactive = False
    elif not sys.stdin.isatty():
        is_interactive = False

    try:
        model, water_regressor, scaler, label_encoder = load_model(model_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    user_inputs = get_user_inputs(interactive=is_interactive)
    yield_rate, water_m2, predicted_disease, probabilities, full_inputs = predict(
        user_inputs, model, water_regressor, scaler, label_encoder
    )
    display_results(
        yield_rate, water_m2, predicted_disease, probabilities, full_inputs,
        show_inputs=not is_interactive
    )


if __name__ == '__main__':
    main()

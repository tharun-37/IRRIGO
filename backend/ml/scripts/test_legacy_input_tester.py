import unittest
from unittest.mock import patch

from sensor_data_mode.model_input_tester import build_feature_row, predict_disease, predict_disease_and_yield, prompt_for_values, summarize_inputs


class ModelInputTesterTests(unittest.TestCase):
    def test_build_feature_row_uses_expected_columns(self):
        values = {
            'Temperature': 24.5,
            'Humidity': 70.0,
            'Moisture': 45.0,
            'Nitrogen': 40.0,
            'Phosphorus': 30.0,
            'Potassium': 35.0,
            'PH': 6.5,
            'Light_Intensity': 500.0,
        }
        row = build_feature_row(values)
        self.assertEqual(list(row.index), [
            'Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'PH', 'Light_Intensity'
        ])

    def test_predict_disease_and_yield_returns_known_label(self):
        result = predict_disease_and_yield({
            'Temperature': 24.5,
            'Humidity': 70.0,
            'Moisture': 45.0,
            'Nitrogen': 40.0,
            'Phosphorus': 30.0,
            'Potassium': 35.0,
            'PH': 6.5,
            'Light_Intensity': 500.0,
        })
        self.assertIn(result['prediction'], {'Healthy', 'Early_Blight', 'Root_Rot', 'Powdery_Mildew', 'Rust', 'Bacterial_Leaf_Spot'})
        self.assertGreaterEqual(result['yield_prediction'], 10.0)
        self.assertLessEqual(result['yield_prediction'], 100.0)

    def test_prompt_for_values_reads_numeric_inputs(self):
        with patch('builtins.input', side_effect=['24.5', '70.0', '45.0', '40.0', '30.0', '35.0', '6.5', '500.0']):
            values = prompt_for_values()
        self.assertEqual(values['Temperature'], 24.5)
        self.assertEqual(values['Humidity'], 70.0)
        self.assertIn('Temperature', values)

    def test_summary_reports_ideal_status(self):
        values = {
            'Temperature': 24.5,
            'Humidity': 70.0,
            'Moisture': 45.0,
            'Nitrogen': 40.0,
            'Phosphorus': 30.0,
            'Potassium': 35.0,
            'PH': 6.5,
            'Light_Intensity': 500.0,
        }
        summary = summarize_inputs(values)
        self.assertIn('Temperature', summary)
        self.assertEqual(summary['PH']['status'], 'ideal')


if __name__ == '__main__':
    unittest.main()

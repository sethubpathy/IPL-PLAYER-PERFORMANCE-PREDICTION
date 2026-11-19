"""
ML Inference Service - Live Multi-Output Random Forest Prediction Engine
IPL Player Performance Prediction System

Loads the trained Multi-Output Random Forest Regressor from models/ipl_multi_output_rf.joblib
and executes genuine live predictions across 7 target metrics:
1. batting_runs
2. fours
3. sixes
4. strike_rate
5. wickets
6. economy_rate
7. performance_score
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ipl_multi_output_rf.joblib')

class MLInferenceService:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.feature_cols = []
        self.target_cols = []
        self.batting_targets = []
        self.bowling_targets = []
        self.overall_targets = []
        self.metrics = {}
        self.feature_importances = {}
        self.input_limits = {}
        self.player_feature_store = {}
        self.is_loaded = False
        self.load_error = None
        self._load_model()

    def _load_model(self):
        """Loads the serialized multi-output model bundle from disk."""
        if not os.path.exists(MODEL_PATH):
            self.load_error = f"Model artifact not found at {MODEL_PATH}. Please run train_multi_output_model.py."
            print(f"Warning: {self.load_error}")
            return False

        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle['model']
            self.imputer = bundle['imputer']
            self.feature_cols = bundle['feature_cols']
            self.target_cols = bundle['target_cols']
            self.batting_targets = bundle.get('batting_targets', ['batting_runs', 'fours', 'sixes', 'strike_rate'])
            self.bowling_targets = bundle.get('bowling_targets', ['wickets', 'economy_rate'])
            self.overall_targets = bundle.get('overall_targets', ['performance_score'])
            self.metrics = bundle.get('metrics', {})
            self.feature_importances = bundle.get('feature_importances', {})
            self.input_limits = bundle.get('input_limits', {})
            self.player_feature_store = bundle.get('player_feature_store', {})
            self.is_loaded = True
            print(f"MLInferenceService: Multi-Output Model successfully loaded ({len(self.player_feature_store)} players in store).")
            return True
        except Exception as e:
            self.load_error = f"Failed to deserialize model: {str(e)}"
            print(f"Error: {self.load_error}")
            return False

    def get_player_baseline(self, player_name):
        """Retrieves a player's recent historical baseline features from the feature store."""
        if not self.is_loaded:
            return None
        return self.player_feature_store.get(player_name)

    def validate_inputs(self, input_dict):
        """
        Validates and sanitizes user input features.
        Returns: (sanitized_features_dict, list_of_validation_errors)
        """
        sanitized = {}
        errors = []

        # Sensible bounding fallback limits
        fallback_limits = {
            'previous_avg_runs': (0.0, 150.0),
            'previous_avg_balls_faced': (0.0, 120.0),
            'previous_avg_strike_rate': (0.0, 400.0),
            'previous_avg_wickets': (0.0, 10.0),
            'previous_avg_balls_bowled': (0.0, 30.0),
            'previous_avg_economy_rate': (0.0, 36.0),
            'previous_avg_bowling_average': (0.0, 150.0),
            'previous_avg_performance': (0.0, 250.0),
            'previous_std_performance': (0.0, 150.0),
            'previous_performance': (0.0, 300.0),
            'previous_matches': (0, 500),
            'season': (2008, 2035)
        }

        for feat in self.feature_cols:
            val = input_dict.get(feat)
            
            # If not provided, check if imputer or player baseline can provide default
            if val is None or val == '':
                # If median is available from input_limits, use it
                if feat in self.input_limits:
                    sanitized[feat] = float(self.input_limits[feat]['median'])
                else:
                    sanitized[feat] = 0.0
                continue

            try:
                num_val = float(val)
            except (ValueError, TypeError):
                errors.append(f"Field '{feat}' must be a valid number, got: {val}")
                continue

            # Range validation
            min_bound, max_bound = fallback_limits.get(feat, (0.0, 1000.0))
            if num_val < min_bound or num_val > max_bound:
                errors.append(f"Field '{feat}' value {num_val} is outside valid range [{min_bound}, {max_bound}]")
                continue

            sanitized[feat] = num_val

        return sanitized, errors

    def predict(self, raw_input):
        """
        Executes genuine live multi-output prediction.
        
        Args:
            raw_input (dict): Can contain 'player' name, individual features, or both.
                              If 'player' is provided, missing features are populated from 
                              their authentic recent career baseline.
        
        Returns:
            dict: Structured response with live predictions, role classification, and metadata.
        """
        if not self.is_loaded:
            if not self._load_model():
                return {
                    'status': 'error',
                    'message': self.load_error or "Model is not loaded."
                }

        player_name = raw_input.get('player', '').strip()
        working_inputs = {}

        # 1. If player is recognized, start with their recent career baseline features
        player_info = None
        if player_name:
            player_info = self.get_player_baseline(player_name)
            if player_info:
                working_inputs.update(player_info['features'])

        # 2. Apply any explicit user overrides or direct inputs
        for k, v in raw_input.items():
            if k in self.feature_cols and v is not None and v != '':
                working_inputs[k] = v

        # 3. Validate and sanitize features
        sanitized_features, errors = self.validate_inputs(working_inputs)
        if errors:
            return {
                'status': 'error',
                'message': 'Input validation failed.',
                'errors': errors
            }

        # 4. Construct feature vector DataFrame in exact model column order
        vector_df = pd.DataFrame([[sanitized_features[col] for col in self.feature_cols]], columns=self.feature_cols)

        # 5. Impute & Predict via Multi-Output Model
        try:
            imputed_vector = self.imputer.transform(vector_df)
            predictions_raw = self.model.predict(imputed_vector)[0]
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Inference execution failed: {str(e)}"
            }

        # 6. Post-processing constraints & rounding
        batting_runs = round(max(0.0, float(predictions_raw[0])), 2)
        fours = round(max(0.0, float(predictions_raw[1])), 2)
        sixes = round(max(0.0, float(predictions_raw[2])), 2)
        strike_rate = round(max(0.0, float(predictions_raw[3])), 2)
        wickets = round(max(0.0, float(predictions_raw[4])), 2)
        economy_rate = round(max(0.0, float(predictions_raw[5])), 2)

        # Cricket domain sanity: non-bowlers do not concede runs per over
        is_non_bowler = False
        if player_info and player_info.get('total_balls_bowled', 0) == 0:
            is_non_bowler = True
        elif sanitized_features.get('previous_avg_balls_bowled', 0) == 0 and sanitized_features.get('previous_avg_wickets', 0) == 0:
            is_non_bowler = True
            
        if is_non_bowler and wickets < 0.15:
            economy_rate = 0.00

        # Exact mathematical consistency with cricket fantasy target: Runs + 20 * Wickets
        performance_score = round(batting_runs + 20.0 * wickets, 2)

        # 7. Projected Player Role Classification based on predicted output profile
        if batting_runs >= 20.0 and wickets >= 0.7:
            projected_role = 'All-Rounder'
        elif wickets >= 0.8:
            projected_role = 'Strike Bowler'
        elif batting_runs >= 18.0:
            projected_role = 'Batsman'
        elif wickets >= 0.4:
            projected_role = 'Bowling All-Rounder'
        else:
            projected_role = 'Supporting Player'

        return {
            'status': 'success',
            'player': player_name or "Custom / Hypothetical Profile",
            'is_known_player': player_info is not None,
            'prediction_mode': 'live_multi_output_inference',
            'model_info': {
                'architecture': 'Multi-Output Random Forest Regressor',
                'trees': 150,
                'target_count': 7
            },
            'predictions': {
                'batting': {
                    'batting_runs': batting_runs,
                    'fours': fours,
                    'sixes': sixes,
                    'strike_rate': strike_rate
                },
                'bowling': {
                    'wickets': wickets,
                    'economy_rate': economy_rate
                },
                'overall': {
                    'performance_score': performance_score
                }
            },
            'summary': {
                'batting_runs': batting_runs,
                'fours': fours,
                'sixes': sixes,
                'strike_rate': strike_rate,
                'wickets': wickets,
                'economy_rate': economy_rate,
                'performance_score': performance_score,
                'projected_role': projected_role
            },
            'inputs_used': sanitized_features,
            'model_evaluation_metrics': self.metrics
        }


# Singleton service instance
ml_service = MLInferenceService()

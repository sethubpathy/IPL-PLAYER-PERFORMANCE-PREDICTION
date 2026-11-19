"""
Comprehensive Verification Test Suite for Future IPL Season Forecasting (2024, 2025, 2026, 2027)
Tests all requirements:
1. Historical dataset usage (2008-2023) and multi-output Random Forest.
2. 7 distinct target predictions: runs, fours, sixes, strike rate, wickets, economy, performance score.
3. Separate, non-identical forecasts for 2024, 2025, 2026, and 2027 with documented autoregressive trajectory & widening uncertainty.
4. Season selector (2024, 2025, 2026, 2027) on /predictions.
5. Leaderboards (Orange Cap, Purple Cap, MVP) and player-wise forecasts for all horizons.
6. JSON API (/api/forecast and /api/predict).
7. Preservation of 2021-2023 test set validation results under /model-analysis.
8. Zero fabrication of 2024-2027 actual data.
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from app import app
from data_service import data_service
from ml_inference_service import ml_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_1_historical_training_and_model_artifact():
    """Verify that multi-output model is trained on 2008-2023 data and saved."""
    model_path = os.path.join(BASE_DIR, 'models', 'ipl_multi_output_rf.joblib')
    assert os.path.exists(model_path), "Model artifact ipl_multi_output_rf.joblib not found"
    
    metadata_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
    assert os.path.exists(metadata_path), "model_metadata.json not found"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    assert "21,716 records" in meta['training_data'], "Expected 21,716 historical records in metadata"
    assert "2008–2023" in meta['training_data'], "Expected 2008-2023 seasons in metadata"
    
    all_targets = meta['targets']['batting'] + meta['targets']['bowling'] + meta['targets']['overall']
    assert len(all_targets) == 7, "Expected 7 target variables"
    expected_targets = ['batting_runs', 'fours', 'sixes', 'strike_rate', 'wickets', 'economy_rate', 'performance_score']
    for t in expected_targets:
        assert t in all_targets, f"Target {t} missing from model metadata"
    
    # Verify 2024-2027 in forecast horizons
    assert '2024' in meta['forecast_horizons']
    assert '2025' in meta['forecast_horizons']
    assert '2026' in meta['forecast_horizons']
    assert '2027' in meta['forecast_horizons']
    print("Test 1 Passed: Historical model artifact verified with 21,716 records and 7 targets (2024-2027 horizons).")


def test_2_forecast_files_and_targets():
    """Verify that IPL_Future_Season_Forecasts.csv covers 2024-2027 with 7 outputs."""
    csv_path = os.path.join(BASE_DIR, 'IPL_Future_Season_Forecasts.csv')
    assert os.path.exists(csv_path), "IPL_Future_Season_Forecasts.csv not found"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 1168, f"Expected 1168 records (292 players x 4 seasons), got {len(df)}"
    seasons = sorted(df['season'].unique().tolist())
    assert seasons == [2024, 2025, 2026, 2027], f"Expected seasons [2024, 2025, 2026, 2027], got {seasons}"
    
    required_cols = [
        'player', 'season', 'horizon', 'projected_match_runs', 'projected_season_runs',
        'projected_fours', 'projected_season_fours', 'projected_sixes', 'projected_season_sixes',
        'projected_strike_rate', 'projected_match_wickets', 'projected_season_wickets',
        'projected_economy_rate', 'projected_performance_score', 'projected_role',
        'confidence_interval_runs', 'confidence_interval_wickets', 'uncertainty_factor',
        'rank_runs', 'rank_wickets', 'rank_mvp'
    ]
    for col in required_cols:
        assert col in df.columns, f"Column {col} missing from forecast CSV"
    
    # Confirm no NaN values in critical target predictions
    for col in ['projected_match_runs', 'projected_match_wickets', 'projected_economy_rate', 'projected_performance_score']:
        assert df[col].isnull().sum() == 0, f"Found nulls in {col}"
    
    print("Test 2 Passed: Forecast file structure verified for 292 players across 2024, 2025, 2026, and 2027.")


def test_3_distinct_multi_season_forecasts():
    """Verify that predictions differ across 2024, 2025, 2026, and 2027 with widening uncertainty."""
    df = pd.read_csv(os.path.join(BASE_DIR, 'IPL_Future_Season_Forecasts.csv'))
    
    for player in ['V Kohli', 'RG Sharma', 'AD Russell', 'JJ Bumrah', 'Shubman Gill']:
        p_df = df[df['player'] == player].sort_values(by='season')
        assert len(p_df) == 4, f"Expected 4 forecasts for {player}, got {len(p_df)}"
        
        runs_2024 = p_df[p_df['season'] == 2024]['projected_season_runs'].values[0]
        runs_2025 = p_df[p_df['season'] == 2025]['projected_season_runs'].values[0]
        runs_2026 = p_df[p_df['season'] == 2026]['projected_season_runs'].values[0]
        runs_2027 = p_df[p_df['season'] == 2027]['projected_season_runs'].values[0]
        
        unc_2024 = p_df[p_df['season'] == 2024]['uncertainty_factor'].values[0]
        unc_2025 = p_df[p_df['season'] == 2025]['uncertainty_factor'].values[0]
        unc_2026 = p_df[p_df['season'] == 2026]['uncertainty_factor'].values[0]
        unc_2027 = p_df[p_df['season'] == 2027]['uncertainty_factor'].values[0]
        
        # Uncertainty factors must strictly increase over time
        assert unc_2024 < unc_2025 < unc_2026 < unc_2027, f"Uncertainty factors must widen: {unc_2024} < {unc_2025} < {unc_2026} < {unc_2027}"
        assert unc_2024 == 1.0
        assert unc_2025 == 1.25
        assert unc_2026 == 1.55
        assert unc_2027 == 1.85
        
        # Forecasts must not be identical across all four seasons
        assert not (runs_2024 == runs_2025 == runs_2026 == runs_2027), f"Forecasts for {player} must differ across seasons"
        print(f"  {player} Projected Runs: 2024={runs_2024}, 2025={runs_2025}, 2026={runs_2026}, 2027={runs_2027} (Uncertainty: {unc_2024}x -> {unc_2025}x -> {unc_2026}x -> {unc_2027}x)")
    
    print("Test 3 Passed: Separate, distinct forecasts confirmed across all 4 seasons with widening uncertainty.")


def test_4_no_future_actual_data_leakage():
    """Verify that NO actual match records or fabricated actuals exist for 2024-2027."""
    perf_path = os.path.join(BASE_DIR, 'IPL_Player_Match_Performance.csv')
    df_perf = pd.read_csv(perf_path)
    max_season = int(df_perf['season'].max())
    assert max_season == 2023, f"Historical dataset contains data beyond 2023! Found season {max_season}"
    
    # Check that future forecasts file has NO 'actual_performance' or 'actual_runs' column
    df_future = pd.read_csv(os.path.join(BASE_DIR, 'IPL_Future_Season_Forecasts.csv'))
    for col in df_future.columns:
        assert 'actual' not in col.lower(), f"Forbidden actual column '{col}' found in future forecasts file!"
    
    print("Test 4 Passed: Zero actual data leakage and zero fabrication confirmed (historical max season = 2023).")


def test_5_rf_predictions_route_and_season_selector():
    """Verify /predictions route renders season selector for 2024, 2025, 2026, 2027 and 7 outputs."""
    client = app.test_client()
    
    # 1. Default request
    resp = client.get('/predictions')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert 'Future Season Machine Learning Forecasting Hub' in html
    assert 'Season 2024 (Immediate)' in html
    assert 'Season 2025 (Medium)' in html
    assert 'Season 2026 (Long)' in html
    assert 'Season 2027 (Extended)' in html
    assert 'Projected Batting Runs' in html
    assert 'Projected Boundaries' in html
    assert 'Projected Strike Rate' in html
    assert 'Projected Wickets' in html
    assert 'Projected Bowling Economy' in html
    assert 'Projected MVP Score' in html
    assert 'Multi-Year Forecast Trajectory' in html
    assert 'Orange Cap Projections' in html
    assert 'Purple Cap Projections' in html
    assert 'Tournament MVP Projections' in html
    
    # 2. Season 2027 specific request
    resp_2027 = client.get('/predictions?player=V+Kohli&season=2027')
    assert resp_2027.status_code == 200
    html_2027 = resp_2027.data.decode('utf-8')
    assert 'Season 2027' in html_2027
    df_fc = pd.read_csv(os.path.join(BASE_DIR, 'IPL_Future_Season_Forecasts.csv'))
    kohli_runs_27 = str(df_fc[(df_fc['player'] == 'V Kohli') & (df_fc['season'] == 2027)]['projected_season_runs'].iloc[0])
    assert kohli_runs_27 in html_2027  # Kohli's 2027 projected runs
    assert '1.85' in html_2027   # Kohli's 2027 uncertainty multiplier
    
    print("Test 5 Passed: /predictions page renders future forecasting hub, season selectors (including 2027), and 7 outputs.")


def test_6_api_forecast_endpoints():
    """Verify /api/forecast returns structured JSON forecasts for players and seasons including 2027."""
    client = app.test_client()
    
    # 1. Single player query for 2027
    r1 = client.get('/api/forecast?player=V+Kohli&season=2027')
    assert r1.status_code == 200
    d1 = r1.get_json()
    assert d1['status'] == 'success'
    assert d1['player'] == 'V Kohli'
    assert d1['selected_season'] == '2027'
    assert 'primary_forecast' in d1
    pf = d1['primary_forecast']
    assert pf['season'] == 2027
    assert pf['projected_season_runs'] == 401.0
    assert pf['uncertainty_factor'] == 1.85
    assert len(pf['confidence_interval_runs']) == 2
    assert len(d1['multi_year_progression']) == 4  # 2024, 2025, 2026, 2027
    
    # 2. 2027 Overview with rankings
    r2 = client.get('/api/forecast?season=2027')
    assert r2.status_code == 200
    d2 = r2.get_json()
    assert d2['status'] == 'success'
    assert d2['season'] == '2027'
    assert 'rankings' in d2
    assert len(d2['rankings']['top_batsmen']) == 5
    assert len(d2['rankings']['top_bowlers']) == 5
    assert len(d2['rankings']['top_mvps']) == 5
    
    # 3. Check that 2024, 2025, 2026 remain fully accessible
    for yr in ['2024', '2025', '2026']:
        ry = client.get(f'/api/forecast?player=V+Kohli&season={yr}')
        assert ry.status_code == 200
        dy = ry.get_json()
        assert dy['primary_forecast']['season'] == int(yr)
    
    print("Test 6 Passed: /api/forecast returns valid, input-dependent JSON across all seasons (2024, 2025, 2026, 2027).")


def test_7_model_analytics_preserves_2021_2023_test_metrics():
    """Verify that /model-analysis preserves and displays the historical 2021-2023 held-out test evaluation."""
    client = app.test_client()
    resp = client.get('/model-analysis')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    
    # Verify 4,085 test samples and 2021-2023 held-out test benchmarks are displayed
    assert '4,085' in html
    assert '2021–2023' in html
    assert 'Multi-Output Evaluation Benchmarks' in html
    assert 'batting_runs' in html
    assert 'wickets' in html
    assert 'economy_rate' in html
    assert 'performance_score' in html
    assert 'View 2024–2027 Future Forecasts' in html
    
    print("Test 7 Passed: /model-analysis preserves and showcases 2021-2023 test set validation results.")


def test_8_live_prediction_studio_continues_functioning():
    """Verify that /api/predict and /predict continue functioning properly for custom inputs."""
    client = app.test_client()
    
    # Test /api/predict with player name
    r1 = client.get('/api/predict?player=V+Kohli')
    assert r1.status_code == 200
    d1 = r1.get_json()
    assert d1['status'] == 'success'
    assert 'predictions' in d1
    assert 'batting_runs' in d1['summary']
    assert 'performance_score' in d1['summary']
    assert 'batting_runs' in d1['predictions']['batting']
    assert 'wickets' in d1['predictions']['bowling']
    assert 'performance_score' in d1['predictions']['overall']
    
    # Test /api/predict with custom features
    custom_payload = {
        'previous_avg_runs': 55.0,
        'previous_avg_strike_rate': 155.0,
        'previous_avg_wickets': 1.5,
        'previous_avg_economy_rate': 7.5,
        'previous_matches': 60
    }
    r2 = client.post('/api/predict', json=custom_payload)
    assert r2.status_code == 200
    d2 = r2.get_json()
    assert d2['status'] == 'success'
    assert d2['summary']['batting_runs'] > 10
    assert d2['summary']['performance_score'] > 20
    
    print("Test 8 Passed: Live prediction studio & /api/predict remain fully operational.")


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING COMPREHENSIVE FUTURE FORECASTING TEST SUITE (2024-2027)")
    print("=" * 70)
    test_1_historical_training_and_model_artifact()
    test_2_forecast_files_and_targets()
    test_3_distinct_multi_season_forecasts()
    test_4_no_future_actual_data_leakage()
    test_5_rf_predictions_route_and_season_selector()
    test_6_api_forecast_endpoints()
    test_7_model_analytics_preserves_2021_2023_test_metrics()
    test_8_live_prediction_studio_continues_functioning()
    print("=" * 70)
    print("ALL 8 VERIFICATION TESTS PASSED SUCCESSFULLY FOR 2024-2027!")
    print("=" * 70)

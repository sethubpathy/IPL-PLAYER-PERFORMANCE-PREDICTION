"""
Step 3 Verification & Integration Test Suite
Validates Live Prediction Studio, all 7 outputs, UI template, player baselines,
input validation, and all existing routes.
"""

import json
from app import app, ml_service

def run_tests():
    client = app.test_client()
    print("==================================================")
    print("RUNNING STEP 3 LIVE PREDICTION STUDIO TESTS")
    print("==================================================")

    # 1. Test GET /predict (HTML UI rendering)
    print("\n--- 1. Testing GET /predict (Live Studio HTML Page) ---")
    resp = client.get('/predict', headers={'Accept': 'text/html'})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    html = resp.get_data(as_text=True)
    assert "Live Prediction Studio" in html, "Page header missing"
    assert "predictionRadarChart" in html, "Radar chart canvas missing"
    assert "predictionForm" in html, "Prediction form missing"
    assert "Verified Precomputed Model Evaluation Mode" not in html, "Option C banner found in studio!"
    assert "Option C" not in html, "Option C text found in studio!"
    print("[PASS] 200 OK - Live Prediction Studio HTML rendered with all components.")

    # 2. Test /api/player-baseline/<player_name>
    print("\n--- 2. Testing Player Baseline Endpoints ---")
    resp_kohli = client.get('/api/player-baseline/V%20Kohli')
    assert resp_kohli.status_code == 200
    kohli_base = resp_kohli.get_json()
    assert kohli_base['status'] == 'success'
    assert 'previous_avg_runs' in kohli_base['baseline']['features']
    print(f"[PASS] 200 OK - Kohli Baseline loaded: {kohli_base['baseline']['features']['previous_avg_runs']} avg runs, {kohli_base['baseline']['role']} role")

    resp_bumrah = client.get('/api/player-baseline/JJ%20Bumrah')
    assert resp_bumrah.status_code == 200
    bumrah_base = resp_bumrah.get_json()
    assert bumrah_base['status'] == 'success'
    print(f"[PASS] 200 OK - Bumrah Baseline loaded: {bumrah_base['baseline']['features']['previous_avg_wickets']} avg wickets, {bumrah_base['baseline']['role']} role")

    resp_nonexistent = client.get('/api/player-baseline/UnknownPlayerXYZ')
    assert resp_nonexistent.status_code == 404
    print("[PASS] 404 OK - Nonexistent player correctly rejected.")

    # 3. Test Multi-Output Live Predictions across Different Archetypes
    print("\n--- 3. Testing Real ML Predictions (7 Targets) Across Roles ---")
    test_cases = [
        ('Batsman (Virat Kohli)', {'player': 'V Kohli', 'previous_avg_runs': 38.0, 'previous_avg_strike_rate': 135.0, 'previous_avg_wickets': 0.0, 'previous_avg_economy_rate': 0.0, 'previous_matches': 230}),
        ('Strike Bowler (Jasprit Bumrah)', {'player': 'JJ Bumrah', 'previous_avg_runs': 3.0, 'previous_avg_strike_rate': 60.0, 'previous_avg_wickets': 1.3, 'previous_avg_economy_rate': 7.2, 'previous_matches': 120}),
        ('All-Rounder (Ravindra Jadeja)', {'player': 'RA Jadeja', 'previous_avg_runs': 24.0, 'previous_avg_strike_rate': 140.0, 'previous_avg_wickets': 0.9, 'previous_avg_economy_rate': 7.6, 'previous_matches': 210}),
        ('Finisher (Andre Russell)', {'player': 'AD Russell', 'previous_avg_runs': 30.0, 'previous_avg_strike_rate': 180.0, 'previous_avg_wickets': 0.7, 'previous_avg_economy_rate': 9.2, 'previous_matches': 110}),
        ('Custom Profile (Rookie Prodigy)', {'player': 'Rookie Prodigy', 'previous_avg_runs': 45.0, 'previous_avg_strike_rate': 155.0, 'previous_avg_wickets': 0.5, 'previous_avg_economy_rate': 8.0, 'previous_matches': 25})
    ]

    for label, payload in test_cases:
        resp = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        assert resp.status_code == 200, f"Failed for {label}"
        data = resp.get_json()
        assert data['status'] == 'success'
        summary = data['summary']
        
        # Verify all 7 targets + projected role are present and numeric
        assert 'batting_runs' in summary and summary['batting_runs'] >= 0
        assert 'fours' in summary and summary['fours'] >= 0
        assert 'sixes' in summary and summary['sixes'] >= 0
        assert 'strike_rate' in summary and summary['strike_rate'] >= 0
        assert 'wickets' in summary and summary['wickets'] >= 0
        assert 'economy_rate' in summary and summary['economy_rate'] >= 0
        assert 'performance_score' in summary and summary['performance_score'] >= 0
        assert 'projected_role' in summary

        print(f"[PASS] {label}:")
        print(f"       Role: {summary['projected_role']} | Score: {summary['performance_score']} pts")
        print(f"       Runs: {summary['batting_runs']}, 4s: {summary['fours']}, 6s: {summary['sixes']}, SR: {summary['strike_rate']}")
        print(f"       Wickets: {summary['wickets']}, Economy: {summary['economy_rate']}")

    # 4. Test Validation Rejections
    print("\n--- 4. Testing Input Validation & Rejection ---")
    invalid_cases = [
        ("Negative Runs", {'previous_avg_runs': -10.0}),
        ("Excessive Strike Rate", {'previous_avg_strike_rate': 550.0}),
        ("Negative Wickets", {'previous_avg_wickets': -2.0}),
        ("Non-numeric matches", {'previous_matches': 'two_hundred'})
    ]

    for label, payload in invalid_cases:
        resp = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        assert resp.status_code == 400, f"Expected 400 for {label}, got {resp.status_code}"
        err_json = resp.get_json()
        assert err_json['status'] == 'error'
        print(f"[PASS] 400 Bad Request correctly returned for {label}: {err_json['errors']}")

    # 5. Test Preservation of All Existing Routes
    print("\n--- 5. Testing Existing Route Preservation ---")
    routes = [
        ('/', 'Dashboard Overview'),
        ('/predictions', 'RF Predictions Explorer'),
        ('/players', 'Player Catalog'),
        ('/player/V%20Kohli', 'Player Detail - Kohli'),
        ('/compare?p1=V%20Kohli&p2=MS%20Dhoni', 'Compare Kohli vs Dhoni'),
        ('/model-analysis', 'Model Analytics'),
        ('/api/predictions/player/V%20Kohli', 'API Player Predictions'),
        ('/api/charts/feature-importance', 'API Feature Importance'),
        ('/api/charts/error-distribution', 'API Error Distribution'),
        ('/api/players/search?q=kohli', 'API Player Search')
    ]
    for route, desc in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Failed on {route}"
        print(f"[PASS] 200 OK - {desc} ({route})")

    # 6. Check for banned phrases in all templates
    print("\n--- 6. Checking Templates for Banner Cleanliness ---")
    for r, _ in routes[:6]:
        resp = client.get(r)
        txt = resp.get_data(as_text=True)
        assert "Verified Precomputed Model Evaluation Mode" not in txt, f"Banner text found in {r}"
        assert "transparency-banner" not in txt, f"Banner class found in {r}"

    print("[PASS] Verified: Zero occurrences of transparency banners across all pages.")

    print("\n==================================================")
    print("ALL STEP 3 TESTS PASSED PERFECTLY!")
    print("==================================================")

if __name__ == '__main__':
    run_tests()

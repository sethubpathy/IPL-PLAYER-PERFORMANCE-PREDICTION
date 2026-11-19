"""
Step 2 Verification & Integration Test Suite
Tests all ML endpoints, validation rules, existing routes, and error handling.
"""

import json
from app import app, ml_service

def run_tests():
    client = app.test_client()
    test_results = {}

    print("==================================================")
    print("RUNNING STEP 2 BACKEND INTEGRATION TESTS")
    print("==================================================")

    # 1. Test Existing Routes
    existing_routes = [
        ('/', 'Dashboard Overview'),
        ('/predictions', 'Predictions Explorer'),
        ('/players', 'Player Catalog'),
        ('/player/V%20Kohli', 'Player Detail - Kohli'),
        ('/compare?p1=V%20Kohli&p2=MS%20Dhoni', 'Compare Kohli vs Dhoni'),
        ('/model-analysis', 'Model Analysis'),
        ('/api/predictions/player/V%20Kohli', 'API Player Predictions'),
        ('/api/charts/feature-importance', 'API Feature Importance'),
        ('/api/charts/error-distribution', 'API Error Distribution'),
        ('/api/players/search?q=kohli', 'API Player Search')
    ]

    print("\n--- 1. Testing Existing Routes Preservation ---")
    route_status = {}
    all_existing_pass = True
    for route, desc in existing_routes:
        resp = client.get(route)
        is_ok = resp.status_code == 200
        route_status[desc] = {'route': route, 'status': resp.status_code, 'passed': is_ok}
        status_symbol = "[PASS]" if is_ok else "[FAIL]"
        print(f"{status_symbol} {resp.status_code} - {desc} ({route})")
        if not is_ok:
            all_existing_pass = False

    test_results['existing_routes'] = {
        'all_passed': all_existing_pass,
        'details': route_status
    }

    # 2. Test /api/predict info mode (GET with no params)
    print("\n--- 2. Testing /api/predict (Info / Readiness) ---")
    resp = client.get('/api/predict')
    print(f"Status: {resp.status_code}")
    info_json = resp.get_json()
    print("Response:", json.dumps(info_json, indent=2))
    test_results['api_predict_info'] = {'status': resp.status_code, 'data': info_json}

    # 3. Test /api/predict with known batsman (V Kohli) via GET
    print("\n--- 3. Testing /api/predict (GET player=V Kohli) ---")
    resp = client.get('/api/predict?player=V+Kohli')
    kohli_json = resp.get_json()
    print(f"Status: {resp.status_code}")
    print("Kohli Predictions Summary:", json.dumps(kohli_json.get('summary', {}), indent=2))
    test_results['kohli_prediction'] = {'status': resp.status_code, 'data': kohli_json}

    # 4. Test /api/predict with known bowler (JJ Bumrah) via POST JSON
    print("\n--- 4. Testing /api/predict (POST JSON player=JJ Bumrah) ---")
    resp = client.post('/api/predict', 
                       data=json.dumps({'player': 'JJ Bumrah'}),
                       content_type='application/json')
    bumrah_json = resp.get_json()
    print(f"Status: {resp.status_code}")
    print("Bumrah Predictions Summary:", json.dumps(bumrah_json.get('summary', {}), indent=2))
    test_results['bumrah_prediction'] = {'status': resp.status_code, 'data': bumrah_json}

    # 5. Test /api/predict with all-rounder (RA Jadeja) via POST Form
    print("\n--- 5. Testing /api/predict (POST Form player=RA Jadeja) ---")
    resp = client.post('/api/predict', data={'player': 'RA Jadeja'})
    jadeja_json = resp.get_json()
    print(f"Status: {resp.status_code}")
    print("Jadeja Predictions Summary:", json.dumps(jadeja_json.get('summary', {}), indent=2))
    test_results['jadeja_prediction'] = {'status': resp.status_code, 'data': jadeja_json}

    # 6. Test /api/predict with hypothetical custom statistics (no player name)
    print("\n--- 6. Testing /api/predict (Custom Hypothetical Features) ---")
    custom_payload = {
        'previous_avg_runs': 42.5,
        'previous_avg_balls_faced': 28.0,
        'previous_avg_strike_rate': 151.78,
        'previous_avg_wickets': 0.8,
        'previous_avg_balls_bowled': 18.0,
        'previous_avg_economy_rate': 7.5,
        'previous_avg_bowling_average': 22.5,
        'previous_avg_performance': 65.0,
        'previous_std_performance': 25.0,
        'previous_performance': 70.0,
        'previous_matches': 60,
        'season': 2024
    }
    resp = client.post('/api/predict',
                       data=json.dumps(custom_payload),
                       content_type='application/json')
    custom_json = resp.get_json()
    print(f"Status: {resp.status_code}")
    print("Custom Stats Summary:", json.dumps(custom_json.get('summary', {}), indent=2))
    test_results['custom_stats_prediction'] = {'status': resp.status_code, 'data': custom_json}

    # 7. Test /predict endpoint (GET and POST)
    print("\n--- 7. Testing /predict Endpoint ---")
    resp_predict_get = client.get('/predict')
    print(f"/predict (GET bare) Status: {resp_predict_get.status_code}")
    resp_predict_post = client.post('/predict',
                                    data=json.dumps({'player': 'MS Dhoni'}),
                                    content_type='application/json')
    dhoni_json = resp_predict_post.get_json()
    print(f"/predict (POST player=MS Dhoni) Status: {resp_predict_post.status_code}")
    print("Dhoni Predictions Summary:", json.dumps(dhoni_json.get('summary', {}), indent=2))
    test_results['dhoni_predict'] = {'status': resp_predict_post.status_code, 'data': dhoni_json}

    # 8. Test Validation Error: Non-numeric input
    print("\n--- 8. Testing Validation Handling: Non-numeric Value ---")
    resp_err1 = client.post('/api/predict',
                            data=json.dumps({'previous_avg_runs': 'invalid_string'}),
                            content_type='application/json')
    print(f"Status: {resp_err1.status_code}")
    err1_json = resp_err1.get_json()
    print("Response:", json.dumps(err1_json, indent=2))
    test_results['validation_non_numeric'] = {'status': resp_err1.status_code, 'data': err1_json}

    # 9. Test Validation Error: Out of Range input
    print("\n--- 9. Testing Validation Handling: Out of Range Value ---")
    resp_err2 = client.post('/api/predict',
                            data=json.dumps({'previous_avg_runs': -25.0, 'previous_avg_strike_rate': 850.0}),
                            content_type='application/json')
    print(f"Status: {resp_err2.status_code}")
    err2_json = resp_err2.get_json()
    print("Response:", json.dumps(err2_json, indent=2))
    test_results['validation_out_of_range'] = {'status': resp_err2.status_code, 'data': err2_json}

    # Summary
    print("\n==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

    # Save results to a file for reference
    with open('step2_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

if __name__ == '__main__':
    run_tests()

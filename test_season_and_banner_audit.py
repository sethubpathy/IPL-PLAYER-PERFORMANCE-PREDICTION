"""
Verification Test Suite:
1. Verify Option C banner is completely eliminated from all pages.
2. Verify season dropdowns across Dashboard, Player Explorer, Player Career Profile, Head-to-Head, and RF Predictions.
3. Verify test set seasons (2021-2023) vs historical training seasons (2008-2020) distinction.
"""

from app import app, data_service

def run_season_and_banner_tests():
    client = app.test_client()
    print("==================================================================")
    print("RUNNING SEASON DROPDOWNS & BANNER ELIMINATION AUDIT")
    print("==================================================================")

    banned_phrases = [
        "Verified Precomputed Model Evaluation Mode",
        "transparency-banner",
        "missing from the repository",
        "Option C"
    ]

    all_routes_to_check = [
        ('/', 'Dashboard Overview'),
        ('/predictions', 'RF Predictions Explorer'),
        ('/players', 'Player Explorer'),
        ('/player/V%20Kohli', 'Player Detail - Kohli'),
        ('/compare?p1=V%20Kohli&p2=RG%20Sharma', 'Head-to-Head Compare'),
        ('/model-analysis', 'Model Analytics'),
        ('/predict', 'Live Prediction Studio')
    ]

    print("\n--- 1. Testing Banner Elimination Across All Pages ---")
    for route, desc in all_routes_to_check:
        resp = client.get(route, headers={'Accept': 'text/html'})
        assert resp.status_code == 200, f"Route {route} failed with {resp.status_code}"
        html = resp.get_data(as_text=True)

        for phrase in banned_phrases:
            assert phrase not in html, f"Banned phrase '{phrase}' found on {desc} ({route})"
        print(f"[PASS] 200 OK - {desc}: Clean, zero banners or Option C disclaimers found.")

    print("\n--- 2. Testing Dashboard Season Dropdown (2008–2023) ---")
    resp_dash = client.get('/')
    html_dash = resp_dash.get_data(as_text=True)
    assert 'Tournament Season Filter:' in html_dash, "Season filter toolbar missing on Dashboard"
    # Verify all 16 seasons are present in dropdown
    for yr in range(2008, 2024):
        assert f'value="{yr}"' in html_dash, f"Season {yr} missing from Dashboard dropdown"
    print("[PASS] Dashboard dropdown contains all 16 seasons (2008 to 2023).")

    # Test Dashboard with specific season: 2016 (Training Era)
    resp_dash_2016 = client.get('/?season=2016')
    assert resp_dash_2016.status_code == 200
    html_2016 = resp_dash_2016.get_data(as_text=True)
    assert "Historical Training Era (2016)" in html_2016
    print("[PASS] Dashboard successfully filters to Season 2016 (Historical Training Era).")

    # Test Dashboard with specific season: 2023 (Test Era)
    resp_dash_2023 = client.get('/?season=2023')
    assert resp_dash_2023.status_code == 200
    html_2023 = resp_dash_2023.get_data(as_text=True)
    assert "Held-Out Test Era (2023)" in html_2023
    print("[PASS] Dashboard successfully filters to Season 2023 (Held-Out Test Era).")

    print("\n--- 3. Testing Player Explorer Season Dropdown (2008–2023) ---")
    resp_players = client.get('/players')
    html_players = resp_players.get_data(as_text=True)
    assert 'name="season"' in html_players, "Season dropdown missing on Player Explorer"
    for yr in range(2008, 2024):
        assert f'value="{yr}"' in html_players, f"Season {yr} missing from Player Explorer dropdown"
    print("[PASS] Player Explorer dropdown contains all 16 seasons (2008 to 2023).")

    # Test Player Explorer filtered to season 2018
    resp_players_2018 = client.get('/players?season=2018')
    assert resp_players_2018.status_code == 200
    html_p2018 = resp_players_2018.get_data(as_text=True)
    assert "Season 2018" in html_p2018
    print("[PASS] Player Explorer filters successfully for Season 2018.")

    print("\n--- 4. Testing Player Career Profile Season Dropdown ---")
    resp_kohli = client.get('/player/V%20Kohli')
    html_kohli = resp_kohli.get_data(as_text=True)
    for yr in [2008, 2011, 2016, 2021, 2022, 2023]:
        assert f'value="{yr}"' in html_kohli, f"Season {yr} missing from Kohli profile dropdown"
    assert "Held-Out Model Test Era (2021–2023)" in html_kohli
    assert "Historical Training Baseline Era (2008–2020)" in html_kohli
    print("[PASS] Player Profile contains all active career seasons with distinct era optgroups.")

    # Test Kohli filtered to 2016
    resp_kohli_2016 = client.get('/player/V%20Kohli?season=2016')
    assert resp_kohli_2016.status_code == 200
    html_k2016 = resp_kohli_2016.get_data(as_text=True)
    assert "Displaying statistics for <strong>IPL Season 2016</strong>" in html_k2016
    assert "973" in html_k2016  # Kohli scored 973 runs in 2016!
    print("[PASS] Player Profile displays authentic 2016 stats (973 runs in Season 2016).")

    print("\n--- 5. Testing Head-to-Head Season Dropdown ---")
    resp_compare = client.get('/compare?p1=V%20Kohli&p2=RG%20Sharma')
    html_compare = resp_compare.get_data(as_text=True)
    assert 'name="season"' in html_compare, "Season dropdown missing on Compare page"
    for yr in range(2008, 2024):
        assert f'value="{yr}"' in html_compare, f"Season {yr} missing from Compare dropdown"
    print("[PASS] Compare page dropdown contains all 16 seasons (2008 to 2023).")

    # Test Compare filtered to season 2023
    resp_comp_2023 = client.get('/compare?p1=V%20Kohli&p2=RG%20Sharma&season=2023')
    assert resp_comp_2023.status_code == 200
    html_c2023 = resp_comp_2023.get_data(as_text=True)
    assert "Held-Out Test Era (Season 2023)" in html_c2023
    print("[PASS] Head-to-Head Compare successfully compares Season 2023.")

    print("\n--- 6. Testing RF Predictions Season Dropdown (2024–2027 Future Horizons) ---")
    resp_pred = client.get('/predictions?player=V%20Kohli')
    html_pred = resp_pred.get_data(as_text=True)
    
    # Must contain 2024, 2025, 2026, 2027
    assert 'value="2024"' in html_pred, "Season 2024 missing from RF predictions"
    assert 'value="2025"' in html_pred, "Season 2025 missing from RF predictions"
    assert 'value="2026"' in html_pred, "Season 2026 missing from RF predictions"
    assert 'value="2027"' in html_pred, "Season 2027 missing from RF predictions"

    # Must NOT contain historical years in future forecasting dropdown
    for yr in range(2008, 2024):
        assert f'<option value="{yr}"' not in html_pred, f"Historical season {yr} erroneously found in RF future forecasts dropdown"

    assert "Future Season Machine Learning Forecasting Hub" in html_pred, "Forecasting hub header missing"
    print("[PASS] RF Predictions dropdown correctly configured for future forecast seasons (2024, 2025, 2026, 2027).")

    print("\n==================================================================")
    print("ALL SEASON DROPDOWN AND BANNER ELIMINATION TESTS PASSED 100%!")
    print("==================================================================")

if __name__ == '__main__':
    run_season_and_banner_tests()

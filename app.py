"""
IPL Player Performance Prediction - Flask Web Application
Theme: Professional Dark IPL Stadium Aesthetic
Model: Multi-Output Random Forest Regressor & Historical Performance Analytics
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from data_service import data_service
from ml_inference_service import ml_service

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ipl-prediction-ml-secure-key-2026'

# Ensure data is loaded on first request / startup
data_service.load_data()


@app.context_processor
def inject_global_data():
    """Inject common template helpers and metadata."""
    return {
        'model_name': 'Multi-Output Random Forest Regressor',
        'model_status': 'Live Multi-Output Inference Active' if ml_service.is_loaded else 'Precomputed Test Output (Seasons 2021–2023)',
        'has_binary_model': ml_service.is_loaded,
        'pred_players_list': data_service.players_with_predictions,
        'all_players_list': data_service.all_players[:100]  # sample for autocomplete
    }


@app.route('/')
def index():
    """Dashboard / Overview page with full 2008-2023 season filtering."""
    selected_season = request.args.get('season', 'All').strip()
    stats = data_service.get_dashboard_stats(season=selected_season)
    features = data_service.get_feature_importances()
    comparisons = data_service.get_model_comparison()
    return render_template(
        'index.html',
        stats=stats,
        features=features,
        comparisons=comparisons,
        all_seasons=data_service.all_seasons,
        selected_season=selected_season
    )


@app.route('/predictions')
def predictions():
    """
    Future Season Forecasting Hub (2024, 2025, 2026) using Multi-Output Random Forest.
    Trained on complete historical match records (2008–2023).
    Displays 7 predicted performance outputs, player rankings, multi-year trajectories, and uncertainty intervals.
    """
    selected_player = request.args.get('player', 'V Kohli').strip()
    selected_season = request.args.get('season', '2024').strip()
    role_filter = request.args.get('role', 'all').strip()

    # Valid forecast seasons (2024, 2025, 2026, 2027)
    forecast_seasons = ['2024', '2025', '2026', '2027', 'All']
    if selected_season not in forecast_seasons:
        selected_season = '2024'

    # Fallback to top player if selected_player not in forecast list
    forecast_players = data_service.players_with_forecasts
    if selected_player not in forecast_players and forecast_players:
        match = [p for p in forecast_players if p.lower() == selected_player.lower()]
        if match:
            selected_player = match[0]
        else:
            selected_player = 'V Kohli' if 'V Kohli' in forecast_players else forecast_players[0]

    # Player Multi-Output Forecast Data
    player_forecast = data_service.get_player_future_forecast(selected_player, season=selected_season)

    # Season Rankings (Orange Cap, Purple Cap, MVP)
    rankings_season = 2024 if selected_season == 'All' else int(selected_season)
    rankings = data_service.get_forecast_rankings(season=rankings_season, top_n=5)

    # Full Catalog for Table
    catalog = data_service.get_all_forecasts_catalog(season=selected_season, role_filter=role_filter)

    return render_template(
        'predictions.html',
        selected_player=selected_player,
        selected_season=selected_season,
        forecast_seasons=['2024', '2025', '2026', '2027', 'All'],
        player_forecast=player_forecast,
        rankings=rankings,
        rankings_season=rankings_season,
        catalog=catalog,
        role_filter=role_filter,
        players_list=forecast_players,
        test_seasons=data_service.test_seasons
    )


@app.route('/players')
def players():
    """All Players Catalog & Career Search supporting all 2008-2023 seasons."""
    search_query = request.args.get('q', '').strip()
    filter_type = request.args.get('filter', 'all')  # all, predicted, batsman, bowler
    selected_season = request.args.get('season', 'All').strip()

    catalog = data_service.get_all_players_catalog(season=selected_season)

    if search_query:
        catalog = [p for p in catalog if search_query.lower() in p['player'].lower()]

    if filter_type == 'predicted':
        catalog = [p for p in catalog if p['has_predictions']]
    elif filter_type == 'batsman':
        threshold = 150 if selected_season != 'All' else 500
        catalog = [p for p in catalog if p['runs'] >= threshold]
    elif filter_type == 'bowler':
        threshold = 8 if selected_season != 'All' else 20
        catalog = [p for p in catalog if p['wickets'] >= threshold]

    return render_template(
        'players.html',
        catalog=catalog,
        search_query=search_query,
        filter_type=filter_type,
        selected_season=selected_season,
        all_seasons=data_service.all_seasons,
        total_count=len(catalog)
    )


@app.route('/player/<player_name>')
def player_detail(player_name):
    """Detailed Career Profile of an Individual Player with 2008-2023 Season Filtering."""
    selected_season = request.args.get('season', 'All').strip()
    profile = data_service.get_player_profile(player_name, season=selected_season)
    if not profile:
        return render_template('404.html', message=f"Player '{player_name}' not found."), 404

    return render_template(
        'player_detail.html',
        profile=profile,
        career=profile,
        pred_data=profile['rf']['data']
    )


@app.route('/compare')
def compare():
    """Head-to-Head Comparison between Two Players across full history or specific seasons."""
    p1 = request.args.get('p1', 'V Kohli')
    p2 = request.args.get('p2', 'RG Sharma')
    selected_season = request.args.get('season', 'All').strip()

    c1 = data_service.get_player_career(p1, season=selected_season)
    c2 = data_service.get_player_career(p2, season=selected_season)

    pred_season = selected_season if selected_season in ['2021', '2022', '2023'] else None
    pred1 = data_service.get_player_predictions(p1, season=pred_season) if (c1 and c1.get('has_predictions')) else None
    pred2 = data_service.get_player_predictions(p2, season=pred_season) if (c2 and c2.get('has_predictions')) else None

    season_era = None
    if selected_season != 'All':
        try:
            s_int = int(selected_season)
            season_era = {
                'season': s_int,
                'is_test_era': s_int in data_service.test_seasons,
                'label': f"Held-Out Test Era (Season {s_int})" if s_int in data_service.test_seasons else f"Historical Training Era (Season {s_int})"
            }
        except (ValueError, TypeError):
            pass

    return render_template(
        'compare.html',
        p1=p1,
        p2=p2,
        c1=c1,
        c2=c2,
        pred1=pred1,
        pred2=pred2,
        all_players=data_service.all_players,
        all_seasons=data_service.all_seasons,
        selected_season=selected_season,
        season_era=season_era
    )


@app.route('/model-analysis')
def model_analysis():
    """Model Architecture, Feature Importance & Residual Analytics."""
    features = data_service.get_feature_importances()
    metrics = data_service.get_dashboard_stats()['metrics']
    comparisons = data_service.get_model_comparison()
    err_dist = data_service.get_error_distribution()

    return render_template(
        'model_analysis.html',
        features=features,
        metrics=metrics,
        comparisons=comparisons,
        err_dist=err_dist,
        multi_metrics=ml_service.metrics,
        evaluation_split={
            'train_period': '2008–2020 (17,631 match records)',
            'test_period': '2021–2023 (4,085 held-out match records)',
            'total_samples': 21716,
            'forecast_horizon': '2024, 2025, 2026, 2027 (Pure Out-of-Sample Forecasting)'
        }
    )


# ==========================================
# JSON API ENDPOINTS
# ==========================================

@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    """
    JSON API for Future Season IPL Forecasting (2024, 2025, 2026, 2027).
    Multi-Output Random Forest projections for 7 target dimensions.
    Query Params:
        season: '2024', '2025', '2026', '2027', or 'All' (default: '2024')
        player: Optional player name (e.g. 'V Kohli')
        role: Optional role filter ('Batsman', 'Strike Bowler', etc.)
        limit: Optional maximum rows in catalog
    """
    season = request.args.get('season', '2024').strip()
    player = request.args.get('player', '').strip()
    role = request.args.get('role', 'all').strip()
    limit = request.args.get('limit', type=int)

    if season not in ['2024', '2025', '2026', '2027', 'All']:
        season = '2024'

    if player:
        forecast_data = data_service.get_player_future_forecast(player, season=season)
        if not forecast_data:
            return jsonify({
                'status': 'error',
                'message': f"No future forecast available for player '{player}'."
            }), 404

        return jsonify({
            'status': 'success',
            'type': 'player_future_forecast',
            'model': 'Multi-Output Random Forest Regressor (2008–2023 Historical Training)',
            'forecast_notice': 'Projected performance metrics for future IPL seasons (2024–2027). No 2024–2027 actual results exist or are fabricated.',
            'target_metrics': [
                'batting_runs', 'fours', 'sixes', 'strike_rate',
                'wickets', 'economy_rate', 'performance_score'
            ],
            'player': forecast_data['player'],
            'selected_season': forecast_data['selected_season'],
            'primary_forecast': forecast_data['primary'],
            'multi_year_progression': forecast_data['multi_year'],
            'chart_data': forecast_data['chart_data']
        }), 200

    # If no specific player requested, return season rankings + catalog
    rankings_season = 2024 if season == 'All' else int(season)
    rankings = data_service.get_forecast_rankings(season=rankings_season, top_n=5)
    catalog = data_service.get_all_forecasts_catalog(season=season, role_filter=role, limit=limit)

    return jsonify({
        'status': 'success',
        'type': 'future_season_forecast_overview',
        'season': season,
        'model': 'Multi-Output Random Forest Regressor (2008–2023 Historical Training)',
        'forecast_notice': 'Projected performance metrics for future IPL seasons (2024–2027). No 2024–2027 actual results exist or are fabricated.',
        'target_metrics': [
            'batting_runs', 'fours', 'sixes', 'strike_rate',
            'wickets', 'economy_rate', 'performance_score'
        ],
        'total_players_projected': len(catalog),
        'rankings': rankings,
        'catalog': catalog
    }), 200

@app.route('/api/predictions/player/<player_name>')
def api_player_predictions(player_name):
    """Return JSON predictions for a given player."""
    season = request.args.get('season')
    data = data_service.get_player_predictions(player_name, season=season)
    if not data:
        return jsonify({'error': 'Player not found or has no predictions'}), 404
    return jsonify(data)


@app.route('/api/predict', methods=['GET', 'POST'])
def api_predict():
    """
    Live Multi-Output Machine Learning Prediction API Endpoint.
    Accepts JSON, Form Data, or Query Parameters.
    Returns 7 predicted targets: batting runs, fours, sixes, strike rate, wickets, economy, performance score.
    """
    if not ml_service.is_loaded:
        if not ml_service._load_model():
            return jsonify({
                'status': 'error',
                'message': ml_service.load_error or 'Multi-output machine learning model is not available.'
            }), 503

    if request.method == 'POST':
        payload = request.get_json(silent=True) or request.form.to_dict()
    else:
        payload = request.args.to_dict()

    if not payload:
        return jsonify({
            'status': 'ready',
            'endpoint': '/api/predict',
            'model': 'Multi-Output Random Forest Regressor',
            'target_metrics': ml_service.target_cols,
            'features_required': ml_service.feature_cols,
            'usage': {
                'by_player': '/api/predict?player=V+Kohli',
                'custom_features': '/api/predict?previous_avg_runs=35&previous_avg_strike_rate=135&previous_matches=50'
            }
        }), 200

    result = ml_service.predict(payload)
    if result.get('status') == 'error':
        return jsonify(result), 400

    return jsonify(result), 200


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Live Prediction Studio UI & Inference Endpoint.
    - Standard browser GET: Renders interactive Live Prediction Studio.
    - POST / JSON API: Executes live ML inference and returns JSON.
    """
    if not ml_service.is_loaded:
        if not ml_service._load_model():
            if request.headers.get('Accept') == 'application/json' or request.is_json:
                return jsonify({
                    'status': 'error',
                    'message': ml_service.load_error or 'Multi-output model artifact not found.'
                }), 503

    if request.method == 'POST':
        payload = request.get_json(silent=True) or request.form.to_dict()
        if not payload:
            return jsonify({'status': 'error', 'message': 'No input data provided.'}), 400
        result = ml_service.predict(payload)
        status_code = 400 if result.get('status') == 'error' else 200
        return jsonify(result), status_code

    # For GET requests: check if client explicitly requested JSON API format
    wants_json = (
        request.headers.get('Accept') == 'application/json' or
        request.args.get('format') == 'json' or
        (request.args and not request.accept_mimetypes.accept_html)
    )

    if wants_json:
        payload = request.args.to_dict()
        if not payload:
            return jsonify({
                'status': 'ready',
                'message': 'Live Multi-Output Machine Learning Inference Endpoint Active.',
                'model': 'Multi-Output Random Forest Regressor (150 trees, max_depth=12)',
                'targets': {
                    'batting': ml_service.batting_targets,
                    'bowling': ml_service.bowling_targets,
                    'overall': ml_service.overall_targets
                },
                'evaluation_metrics': ml_service.metrics,
                'instruction': 'Submit player name or feature values via GET or POST to generate predictions.'
            }), 200

        result = ml_service.predict(payload)
        status_code = 400 if result.get('status') == 'error' else 200
        return jsonify(result), status_code

    # Browser GET request: Render the Live Prediction Studio HTML page
    selected_player = request.args.get('player', '')
    popular_players = [
        'V Kohli', 'RG Sharma', 'MS Dhoni', 'JJ Bumrah', 'RA Jadeja',
        'Rashid Khan', 'Shubman Gill', 'KL Rahul', 'Mohammed Shami', 'AD Russell'
    ]
    return render_template(
        'prediction_studio.html',
        all_players=data_service.all_players,
        popular_players=popular_players,
        selected_player=selected_player,
        feature_limits=ml_service.input_limits,
        evaluation_metrics=ml_service.metrics
    )


@app.route('/api/player-baseline/<path:player_name>')
def api_player_baseline(player_name):
    """Return historical baseline features for a player from the feature store."""
    baseline = ml_service.get_player_baseline(player_name)
    if not baseline:
        return jsonify({'status': 'error', 'message': f"Player '{player_name}' not found in feature store."}), 404
    return jsonify({
        'status': 'success',
        'player': player_name,
        'baseline': baseline
    })


@app.route('/api/charts/feature-importance')
def api_feature_importance():
    """Return feature importance data for Chart.js."""
    feats = data_service.get_feature_importances()
    return jsonify({
        'labels': [f['name'] for f in feats],
        'features': [f['feature'] for f in feats],
        'values': [f['importance'] for f in feats]
    })


@app.route('/api/charts/error-distribution')
def api_error_distribution():
    """Return absolute error histogram for Chart.js."""
    dist = data_service.get_error_distribution()
    return jsonify(dist)


@app.route('/api/players/search')
def api_search_players():
    """Fast search endpoint for player autocomplete."""
    q = request.args.get('q', '').strip().lower()
    only_predicted = request.args.get('predicted', 'false').lower() == 'true'

    source = data_service.players_with_predictions if only_predicted else data_service.all_players
    if not q:
        matches = source[:20]
    else:
        matches = [p for p in source if q in p.lower()][:20]

    return jsonify({'results': matches})


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', message="The requested page could not be found."), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    print(f"IPL Player Performance Web App running on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug_mode)

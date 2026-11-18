"""
IPL Player Performance Prediction - Flask Web Application
Theme: Professional Dark IPL Stadium Aesthetic
Model Mode: Option C - Precomputed Random Forest Results & Evaluation
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from data_service import data_service

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ipl-prediction-ml-secure-key-2026'

# Ensure data is loaded on first request / startup
data_service.load_data()


@app.context_processor
def inject_global_data():
    """Inject common template helpers and metadata."""
    return {
        'model_name': 'Random Forest Regressor',
        'model_status': 'Precomputed Test Output (Seasons 2021–2023)',
        'has_binary_model': False,
        'pred_players_list': data_service.players_with_predictions,
        'all_players_list': data_service.all_players[:100]  # sample for autocomplete
    }


@app.route('/')
def index():
    """Dashboard / Overview page."""
    stats = data_service.get_dashboard_stats()
    features = data_service.get_feature_importances()
    comparisons = data_service.get_model_comparison()
    return render_template(
        'index.html',
        stats=stats,
        features=features,
        comparisons=comparisons
    )


@app.route('/predictions')
def predictions():
    """Random Forest Precomputed Predictions Explorer."""
    selected_player = request.args.get('player', 'V Kohli')
    selected_season = request.args.get('season', 'All')

    # If selected player is not in predicted list, fallback to first available
    if selected_player not in data_service.players_with_predictions:
        selected_player = data_service.players_with_predictions[0] if data_service.players_with_predictions else ''

    player_data = data_service.get_player_predictions(selected_player, season=selected_season)
    
    # Available seasons for this player
    available_seasons = ['All']
    if player_data and player_data.get('summary', {}).get('seasons'):
        available_seasons.extend([str(s) for s in player_data['summary']['seasons']])

    return render_template(
        'predictions.html',
        selected_player=selected_player,
        selected_season=selected_season,
        available_seasons=available_seasons,
        player_data=player_data,
        players_list=data_service.players_with_predictions
    )


@app.route('/players')
def players():
    """All Players Catalog & Career Search."""
    search_query = request.args.get('q', '').strip()
    filter_type = request.args.get('filter', 'all')  # all, predicted, batsman, bowler

    catalog = data_service.get_all_players_catalog()

    if search_query:
        catalog = [p for p in catalog if search_query.lower() in p['player'].lower()]

    if filter_type == 'predicted':
        catalog = [p for p in catalog if p['has_predictions']]
    elif filter_type == 'batsman':
        catalog = [p for p in catalog if p['runs'] >= 500]
    elif filter_type == 'bowler':
        catalog = [p for p in catalog if p['wickets'] >= 20]

    return render_template(
        'players.html',
        catalog=catalog,
        search_query=search_query,
        filter_type=filter_type,
        total_count=len(catalog)
    )


@app.route('/player/<player_name>')
def player_detail(player_name):
    """Detailed Profile of an Individual Player."""
    career = data_service.get_player_career(player_name)
    if not career:
        return render_template('404.html', message=f"Player '{player_name}' not found."), 404

    # If player has RF predictions, load them too
    pred_data = data_service.get_player_predictions(player_name) if career['has_predictions'] else None

    return render_template(
        'player_detail.html',
        career=career,
        pred_data=pred_data
    )


@app.route('/compare')
def compare():
    """Head-to-Head Comparison between Two Players."""
    p1 = request.args.get('p1', 'V Kohli')
    p2 = request.args.get('p2', 'RG Sharma')

    c1 = data_service.get_player_career(p1)
    c2 = data_service.get_player_career(p2)

    pred1 = data_service.get_player_predictions(p1) if (c1 and c1['has_predictions']) else None
    pred2 = data_service.get_player_predictions(p2) if (c2 and c2['has_predictions']) else None

    return render_template(
        'compare.html',
        p1=p1,
        p2=p2,
        c1=c1,
        c2=c2,
        pred1=pred1,
        pred2=pred2,
        all_players=data_service.all_players
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
        err_dist=err_dist
    )


# ==========================================
# JSON API ENDPOINTS
# ==========================================

@app.route('/api/predictions/player/<player_name>')
def api_player_predictions(player_name):
    """Return JSON predictions for a given player."""
    season = request.args.get('season')
    data = data_service.get_player_predictions(player_name, season=season)
    if not data:
        return jsonify({'error': 'Player not found or has no predictions'}), 404
    return jsonify(data)


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

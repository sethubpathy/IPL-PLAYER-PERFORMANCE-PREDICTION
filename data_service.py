"""
IPL Player Performance Prediction - Data Service
Handles loading, caching, and querying of all 6 project CSV files:
1. IPL_Random_Forest_Predictions.csv
2. IPL_Player_Match_Performance.csv
3. IPL_Final_Project_Cleaned_Data.csv
4. IPL_Feature_Importance.csv
5. IPL_Model_Evaluation_Metrics.csv
6. IPL_Model_Comparison.csv
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DataService:
    def __init__(self):
        self.predictions_df = None
        self.performance_df = None
        self.feature_importance_df = None
        self.metrics_df = None
        self.comparison_df = None
        self.venue_lookup = {}
        self.players_with_predictions = []
        self.all_players = []
        self.career_cache = {}
        self.loaded = False

    def load_data(self):
        if self.loaded:
            return

        # 1. Feature Importance
        feat_path = os.path.join(BASE_DIR, 'IPL_Feature_Importance.csv')
        if os.path.exists(feat_path):
            self.feature_importance_df = pd.read_csv(feat_path)
            self.feature_importance_df = self.feature_importance_df.sort_values(by='Importance', ascending=False)

        # 2. Evaluation Metrics
        metrics_path = os.path.join(BASE_DIR, 'IPL_Model_Evaluation_Metrics.csv')
        if os.path.exists(metrics_path):
            self.metrics_df = pd.read_csv(metrics_path)

        # 3. Model Comparison
        comp_path = os.path.join(BASE_DIR, 'IPL_Model_Comparison.csv')
        if os.path.exists(comp_path):
            self.comparison_df = pd.read_csv(comp_path)

        # 4. Random Forest Predictions
        pred_path = os.path.join(BASE_DIR, 'IPL_Random_Forest_Predictions.csv')
        if os.path.exists(pred_path):
            self.predictions_df = pd.read_csv(pred_path)
            # Ensure proper rounding and numeric types
            self.predictions_df['Predicted_Performance'] = self.predictions_df['Predicted_Performance'].round(2)
            self.predictions_df['Actual_Performance'] = self.predictions_df['Actual_Performance'].round(2)
            self.predictions_df['Prediction_Error'] = self.predictions_df['Prediction_Error'].round(2)
            self.predictions_df['Absolute_Error'] = self.predictions_df['Absolute_Error'].round(2)
            self.players_with_predictions = sorted(self.predictions_df['player'].dropna().unique().tolist())

        # 5. Player Match Performance (Full 2008-2023)
        perf_path = os.path.join(BASE_DIR, 'IPL_Player_Match_Performance.csv')
        if os.path.exists(perf_path):
            self.performance_df = pd.read_csv(perf_path)
            self.all_players = sorted(self.performance_df['player'].dropna().unique().tolist())

        # 6. Venue lookup from cleaned ball data (match_id -> venue, teams)
        clean_path = os.path.join(BASE_DIR, 'IPL_Final_Project_Cleaned_Data.csv')
        if os.path.exists(clean_path):
            try:
                # Load only match-level metadata to conserve memory
                df_matches = pd.read_csv(
                    clean_path,
                    usecols=['match_id', 'venue', 'team_matchup', 'season', 'start_date']
                ).drop_duplicates(subset=['match_id'])
                for _, row in df_matches.iterrows():
                    self.venue_lookup[row['match_id']] = {
                        'venue': row.get('venue', 'Unknown Venue'),
                        'team_matchup': row.get('team_matchup', 'Matchup Unknown'),
                        'season': row.get('season', ''),
                        'date': row.get('start_date', '')
                    }
            except Exception as e:
                print(f"Notice: Venue lookup load warning: {e}")

        self.loaded = True

    def get_dashboard_stats(self):
        self.load_data()
        total_predictions = len(self.predictions_df) if self.predictions_df is not None else 0
        total_career_matches = len(self.performance_df) if self.performance_df is not None else 0
        total_players = len(self.all_players)
        pred_players_count = len(self.players_with_predictions)

        # Metrics
        metrics = {}
        if self.metrics_df is not None:
            for _, r in self.metrics_df.iterrows():
                metrics[r['Metric']] = float(r['Value'])

        # Top 5 most accurate predictions
        best_preds = []
        if self.predictions_df is not None:
            top_acc = self.predictions_df.sort_values(by='Absolute_Error', ascending=True).head(5)
            best_preds = top_acc[['player', 'season', 'start_date', 'Actual_Performance', 'Predicted_Performance', 'Absolute_Error']].to_dict(orient='records')

        # Top predicted high performers
        top_performers = []
        if self.predictions_df is not None:
            top_p = self.predictions_df.sort_values(by='Predicted_Performance', ascending=False).drop_duplicates(subset=['player']).head(6)
            top_performers = top_p[['player', 'season', 'start_date', 'Actual_Performance', 'Predicted_Performance', 'previous_avg_performance']].to_dict(orient='records')

        return {
            'total_predictions': total_predictions,
            'total_career_records': total_career_matches,
            'total_players': total_players,
            'pred_players_count': pred_players_count,
            'metrics': metrics,
            'best_predictions': best_preds,
            'top_performers': top_performers,
            'seasons_covered': '2008 – 2023',
            'pred_seasons': '2021 – 2023'
        }

    def get_feature_importances(self):
        self.load_data()
        if self.feature_importance_df is None:
            return []
        
        # Add readable labels
        labels = {
            'previous_avg_performance': 'Previous Avg Performance Score',
            'previous_avg_balls_faced': 'Previous Avg Balls Faced',
            'previous_avg_balls_bowled': 'Previous Avg Balls Bowled',
            'previous_avg_strike_rate': 'Previous Avg Batting Strike Rate',
            'previous_matches': 'Total Previous Matches Played',
            'previous_std_performance': 'Consistency (Std Dev of Performance)',
            'previous_avg_economy_rate': 'Previous Avg Bowling Economy Rate',
            'previous_avg_bowling_average': 'Previous Avg Bowling Average',
            'season': 'IPL Season Year',
            'previous_avg_runs': 'Previous Avg Batting Runs',
            'previous_performance': 'Most Recent Match Performance',
            'previous_avg_wickets': 'Previous Avg Wickets Taken'
        }

        res = []
        for _, row in self.feature_importance_df.iterrows():
            feat = row['Feature']
            imp = float(row['Importance'])
            res.append({
                'feature': feat,
                'name': labels.get(feat, feat.replace('_', ' ').title()),
                'importance': round(imp * 100, 2),
                'raw_importance': imp
            })
        return res

    def get_model_comparison(self):
        self.load_data()
        if self.comparison_df is None:
            return []
        return self.comparison_df.to_dict(orient='records')

    def get_player_predictions(self, player_name, season=None):
        self.load_data()
        if self.predictions_df is None:
            return None

        df = self.predictions_df[self.predictions_df['player'].str.lower() == player_name.lower()]
        if df.empty:
            return None

        if season and str(season) != 'All':
            df = df[df['season'].astype(str) == str(season)]

        df = df.sort_values(by=['season', 'start_date'], ascending=True)

        records = []
        for _, row in df.iterrows():
            mid = row['match_id']
            meta = self.venue_lookup.get(mid, {})
            records.append({
                'match_id': mid,
                'season': row['season'],
                'date': row['start_date'],
                'venue': meta.get('venue', 'IPL Venue'),
                'team_matchup': meta.get('team_matchup', 'IPL Match'),
                'batting_runs': int(row.get('batting_runs', 0)),
                'balls_faced': int(row.get('balls_faced', 0)),
                'strike_rate': round(float(row.get('strike_rate', 0)), 1),
                'wickets': int(row.get('wickets', 0)),
                'runs_conceded': int(row.get('runs_conceded', 0)),
                'economy_rate': round(float(row.get('economy_rate', 0)), 2),
                'actual_performance': round(float(row['Actual_Performance']), 2),
                'predicted_performance': round(float(row['Predicted_Performance']), 2),
                'prediction_error': round(float(row['Prediction_Error']), 2),
                'absolute_error': round(float(row['Absolute_Error']), 2),
                # 12 Input Features
                'features': {
                    'previous_avg_performance': round(float(row.get('previous_avg_performance', 0)), 2),
                    'previous_avg_balls_faced': round(float(row.get('previous_avg_balls_faced', 0)), 2),
                    'previous_avg_balls_bowled': round(float(row.get('previous_avg_balls_bowled', 0)), 2),
                    'previous_avg_strike_rate': round(float(row.get('previous_avg_strike_rate', 0)), 2),
                    'previous_matches': int(row.get('previous_matches', 0)),
                    'previous_std_performance': round(float(row.get('previous_std_performance', 0)), 2),
                    'previous_avg_economy_rate': round(float(row.get('previous_avg_economy_rate', 0)), 2),
                    'previous_avg_bowling_average': round(float(row.get('previous_avg_bowling_average', 0)), 2),
                    'season': int(row.get('season', 0)),
                    'previous_avg_runs': round(float(row.get('previous_avg_runs', 0)), 2),
                    'previous_performance': round(float(row.get('previous_performance', 0)), 2),
                    'previous_avg_wickets': round(float(row.get('previous_avg_wickets', 0)), 2),
                }
            })

        # Summary statistics for this player
        actuals = [r['actual_performance'] for r in records]
        preds = [r['predicted_performance'] for r in records]
        abs_errors = [r['absolute_error'] for r in records]

        summary = {
            'player': df.iloc[0]['player'],
            'total_predicted_matches': len(records),
            'avg_actual': round(float(np.mean(actuals)), 2) if actuals else 0,
            'avg_predicted': round(float(np.mean(preds)), 2) if preds else 0,
            'mae': round(float(np.mean(abs_errors)), 2) if abs_errors else 0,
            'rmse': round(float(np.sqrt(np.mean(np.array(abs_errors)**2))), 2) if abs_errors else 0,
            'max_predicted': round(float(np.max(preds)), 2) if preds else 0,
            'max_actual': round(float(np.max(actuals)), 2) if actuals else 0,
            'seasons': sorted(df['season'].unique().tolist())
        }

        return {
            'summary': summary,
            'matches': records
        }

    def get_player_career(self, player_name):
        self.load_data()
        if self.performance_df is None:
            return None

        if player_name.lower() in self.career_cache:
            return self.career_cache[player_name.lower()]

        df = self.performance_df[self.performance_df['player'].str.lower() == player_name.lower()]
        if df.empty:
            return None

        real_name = df.iloc[0]['player']
        total_matches = len(df)
        total_runs = int(df['batting_runs'].sum())
        total_balls_faced = int(df['balls_faced'].sum())
        total_fours = int(df['fours'].sum())
        total_sixes = int(df['sixes'].sum())
        batting_sr = round((total_runs / total_balls_faced * 100), 2) if total_balls_faced > 0 else 0

        total_wickets = int(df['wickets'].sum())
        total_runs_conceded = int(df['runs_conceded'].sum())
        total_balls_bowled = int(df['balls_bowled'].sum())
        bowling_econ = round((total_runs_conceded / (total_balls_bowled / 6)), 2) if total_balls_bowled > 0 else 0
        bowling_avg = round((total_runs_conceded / total_wickets), 2) if total_wickets > 0 else 0

        scores = df['performance_score'].dropna().tolist()
        avg_score = round(float(np.mean(scores)), 2) if scores else 0
        max_score = round(float(np.max(scores)), 2) if scores else 0

        # Season-wise breakdown
        season_stats = []
        for s, sdf in df.groupby('season'):
            s_scores = sdf['performance_score'].dropna().tolist()
            season_stats.append({
                'season': int(s),
                'matches': len(sdf),
                'runs': int(sdf['batting_runs'].sum()),
                'wickets': int(sdf['wickets'].sum()),
                'avg_performance': round(float(np.mean(s_scores)), 2) if s_scores else 0
            })
        season_stats = sorted(season_stats, key=lambda x: x['season'])

        # Check if player has Random Forest predictions
        has_predictions = real_name in self.players_with_predictions

        res = {
            'player': real_name,
            'total_matches': total_matches,
            'total_runs': total_runs,
            'total_wickets': total_wickets,
            'total_balls_faced': total_balls_faced,
            'total_balls_bowled': total_balls_bowled,
            'total_fours': total_fours,
            'total_sixes': total_sixes,
            'batting_sr': batting_sr,
            'bowling_econ': bowling_econ,
            'bowling_avg': bowling_avg,
            'avg_performance_score': avg_score,
            'max_performance_score': max_score,
            'seasons': season_stats,
            'has_predictions': has_predictions,
            'first_season': int(df['season'].min()),
            'last_season': int(df['season'].max())
        }
        self.career_cache[player_name.lower()] = res
        return res

    def get_all_players_catalog(self, limit=None):
        self.load_data()
        catalog = []
        for p in self.all_players:
            c = self.get_player_career(p)
            if c:
                catalog.append({
                    'player': c['player'],
                    'matches': c['total_matches'],
                    'runs': c['total_runs'],
                    'wickets': c['total_wickets'],
                    'avg_score': c['avg_performance_score'],
                    'has_predictions': c['has_predictions'],
                    'span': f"{c['first_season']} - {c['last_season']}"
                })
        # Sort by total matches descending
        catalog = sorted(catalog, key=lambda x: x['matches'], reverse=True)
        if limit:
            return catalog[:limit]
        return catalog

    def get_predictions_catalog(self):
        self.load_data()
        if self.predictions_df is None:
            return []
        
        grouped = self.predictions_df.groupby('player').agg(
            matches=('match_id', 'count'),
            avg_predicted=('Predicted_Performance', 'mean'),
            avg_actual=('Actual_Performance', 'mean'),
            mae=('Absolute_Error', 'mean')
        ).reset_index()

        grouped['avg_predicted'] = grouped['avg_predicted'].round(2)
        grouped['avg_actual'] = grouped['avg_actual'].round(2)
        grouped['mae'] = grouped['mae'].round(2)
        
        # Sort by matches count descending
        grouped = grouped.sort_values(by='matches', ascending=False)
        return grouped.to_dict(orient='records')

    def get_error_distribution(self):
        self.load_data()
        if self.predictions_df is None:
            return {}
        
        errors = self.predictions_df['Absolute_Error'].dropna().tolist()
        # Create bins: 0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50+
        bins = [0, 5, 10, 20, 30, 40, 50, 1000]
        labels = ['0 - 5', '5 - 10', '10 - 20', '20 - 30', '30 - 40', '40 - 50', '50+']
        hist, _ = np.histogram(errors, bins=bins)
        return {
            'labels': labels,
            'counts': [int(c) for c in hist]
        }


# Singleton instance
data_service = DataService()

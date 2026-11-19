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
import json
import ast
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
        self.all_seasons = []
        self.test_seasons = [2021, 2022, 2023]
        self.train_seasons = []
        self.forecast_df = None
        self.forecast_json = None
        self.forecast_seasons = [2024, 2025, 2026, 2027]
        self.players_with_forecasts = []
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
            self.all_seasons = sorted(self.performance_df['season'].dropna().unique().astype(int).tolist())
            self.test_seasons = [s for s in self.all_seasons if s >= 2021]
            self.train_seasons = [s for s in self.all_seasons if s < 2021]

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

        # 7. Future Season Multi-Output Forecasts (2024, 2025, 2026)
        forecast_csv = os.path.join(BASE_DIR, 'IPL_Future_Season_Forecasts.csv')
        forecast_json_path = os.path.join(BASE_DIR, 'models', 'future_season_forecasts.json')
        if os.path.exists(forecast_json_path):
            try:
                with open(forecast_json_path, 'r', encoding='utf-8') as f:
                    self.forecast_json = json.load(f)
            except Exception as e:
                print(f"Notice: Failed to load future forecasts json: {e}")

        if os.path.exists(forecast_csv):
            try:
                self.forecast_df = pd.read_csv(forecast_csv)
                self.players_with_forecasts = sorted(self.forecast_df['player'].dropna().unique().tolist())
                self.forecast_seasons = sorted(self.forecast_df['season'].dropna().unique().astype(int).tolist())
            except Exception as e:
                print(f"Notice: Failed to load future forecasts csv: {e}")

        self.loaded = True

    def get_dashboard_stats(self, season=None):
        self.load_data()
        all_seasons = self.all_seasons or [y for y in range(2008, 2024)]

        is_season_filtered = False
        selected_season = 'All'
        era_info = {
            'is_filtered': False,
            'is_test_era': False,
            'label': 'Full Historical Dataset (2008–2023)'
        }

        if season and str(season).strip() != 'All':
            try:
                s_int = int(season)
                if s_int in all_seasons:
                    is_season_filtered = True
                    selected_season = str(s_int)
                    is_test = s_int in self.test_seasons
                    era_info = {
                        'is_filtered': True,
                        'is_test_era': is_test,
                        'season': s_int,
                        'label': f"Held-Out Model Test Era (Season {s_int})" if is_test else f"Historical Training Baseline Era (Season {s_int})"
                    }
            except (ValueError, TypeError):
                is_season_filtered = False

        if not is_season_filtered:
            total_predictions = len(self.predictions_df) if self.predictions_df is not None else 0
            total_career_matches = len(self.performance_df) if self.performance_df is not None else 0
            total_players = len(self.all_players)
            pred_players_count = len(self.players_with_predictions)

            best_preds = []
            if self.predictions_df is not None:
                top_acc = self.predictions_df.sort_values(by='Absolute_Error', ascending=True).head(5)
                best_preds = top_acc[['player', 'season', 'start_date', 'Actual_Performance', 'Predicted_Performance', 'Absolute_Error']].to_dict(orient='records')

            top_performers = []
            if self.predictions_df is not None:
                top_p = self.predictions_df.sort_values(by='Predicted_Performance', ascending=False).drop_duplicates(subset=['player']).head(6)
                top_performers = top_p[['player', 'season', 'start_date', 'Actual_Performance', 'Predicted_Performance', 'previous_avg_performance']].to_dict(orient='records')
        else:
            s_int = int(selected_season)
            df_s = self.performance_df[self.performance_df['season'] == s_int] if self.performance_df is not None else pd.DataFrame()
            total_career_matches = len(df_s)
            total_players = df_s['player'].nunique() if not df_s.empty else 0

            if s_int in self.test_seasons and self.predictions_df is not None:
                df_preds_s = self.predictions_df[self.predictions_df['season'] == s_int]
                total_predictions = len(df_preds_s)
                pred_players_count = df_preds_s['player'].nunique()
                top_acc = df_preds_s.sort_values(by='Absolute_Error', ascending=True).head(5)
                best_preds = top_acc[['player', 'season', 'start_date', 'Actual_Performance', 'Predicted_Performance', 'Absolute_Error']].to_dict(orient='records')
                top_p = df_preds_s.sort_values(by='Predicted_Performance', ascending=False).drop_duplicates(subset=['player']).head(6)
                top_performers = top_p[['player', 'season', 'start_date', 'Actual_Performance', 'Predicted_Performance', 'previous_avg_performance']].to_dict(orient='records')
            else:
                total_predictions = 0
                pred_players_count = 0
                best_preds = []
                top_performers = []
                if not df_s.empty:
                    top_perf = df_s.sort_values(by='performance_score', ascending=False).drop_duplicates(subset=['player']).head(6)
                    for _, r in top_perf.iterrows():
                        top_performers.append({
                            'player': r['player'],
                            'season': int(r['season']),
                            'start_date': str(r.get('start_date', '')),
                            'Actual_Performance': round(float(r['performance_score']), 2),
                            'Predicted_Performance': 'Training Era',
                            'previous_avg_performance': round(float(r.get('previous_avg_performance', 0)), 2)
                        })

        metrics = {}
        if self.metrics_df is not None:
            for _, r in self.metrics_df.iterrows():
                metrics[r['Metric']] = float(r['Value'])

        return {
            'total_predictions': total_predictions,
            'total_career_records': total_career_matches,
            'total_players': total_players,
            'pred_players_count': pred_players_count,
            'metrics': metrics,
            'best_predictions': best_preds,
            'top_performers': top_performers,
            'seasons_covered': '2008 – 2023',
            'pred_seasons': '2021 – 2023',
            'all_seasons': all_seasons,
            'selected_season': selected_season,
            'is_season_filtered': is_season_filtered,
            'era_info': era_info
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
            if df.empty:
                return None

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

    def get_player_career(self, player_name, season=None):
        self.load_data()
        if self.performance_df is None:
            return None

        cache_key = (player_name.lower(), str(season) if season else 'All')
        if cache_key in self.career_cache:
            return self.career_cache[cache_key]

        df_all = self.performance_df[self.performance_df['player'].str.lower() == player_name.lower()]
        if df_all.empty:
            return None

        real_name = df_all.iloc[0]['player']
        available_seasons = sorted(df_all['season'].dropna().unique().astype(int).tolist())

        if season and str(season).strip() != 'All':
            try:
                s_int = int(season)
                df = df_all[df_all['season'] == s_int]
                is_season_filtered = True
                selected_season_str = str(s_int)
            except (ValueError, TypeError):
                df = df_all
                is_season_filtered = False
                selected_season_str = 'All'
        else:
            df = df_all
            is_season_filtered = False
            selected_season_str = 'All'

        total_matches = len(df)
        total_runs = int(df['batting_runs'].sum()) if total_matches > 0 else 0
        total_balls_faced = int(df['balls_faced'].sum()) if total_matches > 0 else 0
        total_fours = int(df['fours'].sum()) if total_matches > 0 else 0
        total_sixes = int(df['sixes'].sum()) if total_matches > 0 else 0
        batting_sr = round((total_runs / total_balls_faced * 100), 2) if total_balls_faced > 0 else 0

        total_wickets = int(df['wickets'].sum()) if total_matches > 0 else 0
        total_runs_conceded = int(df['runs_conceded'].sum()) if total_matches > 0 else 0
        total_balls_bowled = int(df['balls_bowled'].sum()) if total_matches > 0 else 0
        bowling_econ = round((total_runs_conceded / (total_balls_bowled / 6)), 2) if total_balls_bowled > 0 else 0
        bowling_avg = round((total_runs_conceded / total_wickets), 2) if total_wickets > 0 else 0

        scores = df['performance_score'].dropna().tolist() if total_matches > 0 else []
        avg_score = round(float(np.mean(scores)), 2) if scores else 0
        max_score = round(float(np.max(scores)), 2) if scores else 0

        # Season-wise breakdown across entire career
        season_stats = []
        for s, sdf in df_all.groupby('season'):
            s_scores = sdf['performance_score'].dropna().tolist()
            season_stats.append({
                'season': int(s),
                'matches': len(sdf),
                'runs': int(sdf['batting_runs'].sum()),
                'wickets': int(sdf['wickets'].sum()),
                'avg_performance': round(float(np.mean(s_scores)), 2) if s_scores else 0
            })
        season_stats = sorted(season_stats, key=lambda x: x['season'])

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
            'available_seasons': available_seasons,
            'is_season_filtered': is_season_filtered,
            'selected_season': selected_season_str,
            'has_predictions': has_predictions,
            'first_season': int(df_all['season'].min()),
            'last_season': int(df_all['season'].max())
        }
        self.career_cache[cache_key] = res
        return res

    def get_player_profile(self, player_name, season=None):
        """
        Retrieves comprehensive player career profile from the complete 2008-2023 historical dataset.
        Supports season filtering across all 2008-2023 seasons without restricting to RF test years.
        Strictly restricts Random Forest predictions to their actual held-out test seasons (2021-2023).
        """
        self.load_data()
        if self.performance_df is None:
            return None

        df_player = self.performance_df[self.performance_df['player'].str.lower() == player_name.lower()]
        if df_player.empty:
            return None

        real_name = df_player.iloc[0]['player']
        available_seasons = sorted(df_player['season'].unique().tolist())

        # Complete season-by-season career breakdown (2008-2023)
        season_breakdown = []
        for s, sdf in df_player.groupby('season'):
            s_scores = sdf['performance_score'].dropna().tolist()
            runs = int(sdf['batting_runs'].sum())
            bf = int(sdf['balls_faced'].sum())
            wkts = int(sdf['wickets'].sum())
            rc = int(sdf['runs_conceded'].sum())
            bb = int(sdf['balls_bowled'].sum())
            season_breakdown.append({
                'season': int(s),
                'matches': len(sdf),
                'runs': runs,
                'balls_faced': bf,
                'strike_rate': round(runs / bf * 100, 2) if bf > 0 else 0,
                'fours': int(sdf['fours'].sum()),
                'sixes': int(sdf['sixes'].sum()),
                'wickets': wkts,
                'runs_conceded': rc,
                'balls_bowled': bb,
                'economy': round(rc / (bb / 6), 2) if bb > 0 else 0,
                'bowling_avg': round(rc / wkts, 2) if wkts > 0 else 0,
                'avg_performance': round(float(np.mean(s_scores)), 2) if s_scores else 0,
                'max_performance': round(float(np.max(s_scores)), 2) if s_scores else 0
            })
        season_breakdown = sorted(season_breakdown, key=lambda x: x['season'])

        # Career Overall Totals across all played seasons (2008-2023)
        career_matches = len(df_player)
        career_runs = int(df_player['batting_runs'].sum())
        career_bf = int(df_player['balls_faced'].sum())
        career_fours = int(df_player['fours'].sum())
        career_sixes = int(df_player['sixes'].sum())
        career_sr = round((career_runs / career_bf * 100), 2) if career_bf > 0 else 0

        career_wkts = int(df_player['wickets'].sum())
        career_rc = int(df_player['runs_conceded'].sum())
        career_bb = int(df_player['balls_bowled'].sum())
        career_econ = round((career_rc / (career_bb / 6)), 2) if career_bb > 0 else 0
        career_bavg = round((career_rc / career_wkts), 2) if career_wkts > 0 else 0

        career_scores = df_player['performance_score'].dropna().tolist()
        career_avg_score = round(float(np.mean(career_scores)), 2) if career_scores else 0
        career_max_score = round(float(np.max(career_scores)), 2) if career_scores else 0

        # Handle Season Filtering
        is_filtered = False
        selected_season_str = 'All'
        df_target = df_player

        if season and str(season).strip().lower() not in ('all', '', 'none'):
            try:
                s_int = int(season)
                df_filtered = df_player[df_player['season'] == s_int]
                if not df_filtered.empty:
                    df_target = df_filtered
                    is_filtered = True
                    selected_season_str = str(s_int)
            except ValueError:
                pass

        # Target stats (filtered season or full career)
        target_matches = len(df_target)
        target_runs = int(df_target['batting_runs'].sum())
        target_bf = int(df_target['balls_faced'].sum())
        target_fours = int(df_target['fours'].sum())
        target_sixes = int(df_target['sixes'].sum())
        target_sr = round((target_runs / target_bf * 100), 2) if target_bf > 0 else 0

        target_wkts = int(df_target['wickets'].sum())
        target_rc = int(df_target['runs_conceded'].sum())
        target_bb = int(df_target['balls_bowled'].sum())
        target_econ = round((target_rc / (target_bb / 6)), 2) if target_bb > 0 else 0
        target_bavg = round((target_rc / target_wkts), 2) if target_wkts > 0 else 0

        target_scores = df_target['performance_score'].dropna().tolist()
        target_avg_score = round(float(np.mean(target_scores)), 2) if target_scores else 0
        target_max_score = round(float(np.max(target_scores)), 2) if target_scores else 0

        # Detailed match performance logs for target scope (sorted chronologically descending)
        df_sorted_matches = df_target.sort_values(by=['season', 'start_date'], ascending=False)
        match_logs = []
        for _, row in df_sorted_matches.iterrows():
            mid = row['match_id']
            meta = self.venue_lookup.get(mid, {})
            runs = int(row.get('batting_runs', 0))
            bf = int(row.get('balls_faced', 0))
            sr = round(float(row.get('strike_rate', 0)), 1)
            wkts = int(row.get('wickets', 0))
            rc = int(row.get('runs_conceded', 0))
            bb = int(row.get('balls_bowled', 0))
            econ = round(float(row.get('economy_rate', 0)), 2)
            score = round(float(row.get('performance_score', 0)), 2)

            match_logs.append({
                'match_id': mid,
                'season': int(row['season']),
                'date': row['start_date'],
                'venue': meta.get('venue', 'IPL Venue'),
                'team_matchup': meta.get('team_matchup', 'IPL Match'),
                'batting_runs': runs,
                'balls_faced': bf,
                'fours': int(row.get('fours', 0)),
                'sixes': int(row.get('sixes', 0)),
                'strike_rate': sr,
                'wickets': wkts,
                'runs_conceded': rc,
                'balls_bowled': bb,
                'economy_rate': econ,
                'performance_score': score
            })

        # Random Forest Model Test Results Integration (Strictly 2021-2023)
        has_predictions = real_name in self.players_with_predictions
        rf_data = None
        rf_status = 'NOT_APPLICABLE'
        rf_note = ''

        if has_predictions:
            if not is_filtered:
                # All seasons: show overall test set performance (2021-2023)
                rf_data = self.get_player_predictions(real_name, season=None)
                rf_status = 'AVAILABLE'
                rf_note = "Showing Random Forest test-set evaluation predictions across held-out seasons (2021–2023)."
            else:
                s_int = int(selected_season_str)
                if s_int in [2021, 2022, 2023]:
                    rf_data = self.get_player_predictions(real_name, season=s_int)
                    if rf_data and rf_data['matches']:
                        rf_status = 'AVAILABLE'
                        rf_note = f"Showing Random Forest test-set predictions for Season {s_int}."
                    else:
                        rf_status = 'NO_MATCHES_IN_SEASON'
                        rf_note = f"No test prediction records found for Season {s_int}."
                else:
                    rf_status = 'PRE_TEST_ERA'
                    rf_note = f"Season {s_int} was part of historical baseline training (2008–2020). Random Forest model test predictions are strictly restricted to the 2021–2023 evaluation set to maintain 100% empirical validity."
        else:
            rf_status = 'NOT_APPLICABLE'
            rf_note = f"{real_name} does not feature in the 2021–2023 Random Forest test evaluation split."

        active_stats = {
            'matches': target_matches,
            'runs': target_runs,
            'balls_faced': target_bf,
            'fours': target_fours,
            'sixes': target_sixes,
            'batting_sr': target_sr,
            'wickets': target_wkts,
            'runs_conceded': target_rc,
            'balls_bowled': target_bb,
            'bowling_econ': target_econ,
            'bowling_avg': target_bavg,
            'avg_performance_score': target_avg_score,
            'max_performance_score': target_max_score,
            'total_impact_score': int(target_runs + 20 * target_wkts)
        }

        return {
            'player': real_name,
            'selected_season': selected_season_str,
            'is_season_filtered': is_filtered,
            'available_seasons': available_seasons,
            'first_season': int(df_player['season'].min()),
            'last_season': int(df_player['season'].max()),
            'has_predictions': has_predictions,

            # Active stats in 'stats' dict for template convenience
            'stats': active_stats,

            # Active stats for selected view (filtered season or full career)
            'total_matches': target_matches,
            'total_runs': target_runs,
            'total_balls_faced': target_bf,
            'total_fours': target_fours,
            'total_sixes': target_sixes,
            'batting_sr': target_sr,
            'total_wickets': target_wkts,
            'total_runs_conceded': target_rc,
            'total_balls_bowled': target_bb,
            'bowling_econ': target_econ,
            'bowling_avg': target_bavg,
            'avg_performance_score': target_avg_score,
            'max_performance_score': target_max_score,
            'total_impact_score': int(target_runs + 20 * target_wkts),

            # Persistent career totals across all seasons
            'career_totals': {
                'matches': career_matches,
                'runs': career_runs,
                'wickets': career_wkts,
                'avg_performance_score': career_avg_score,
                'span': f"{int(df_player['season'].min())} – {int(df_player['season'].max())}"
            },

            # Complete historical career season breakdown for chart & table
            'seasons': season_breakdown,
            'match_logs': match_logs,

            # Random Forest metadata & results
            'rf': {
                'data': rf_data,
                'status': rf_status,
                'note': rf_note
            },
            # Future season forecast (2024–2026)
            'future_forecast': self.get_player_future_forecast(real_name)
        }

    def get_all_players_catalog(self, limit=None, season=None):
        self.load_data()
        if self.performance_df is None:
            return []

        if season and str(season).strip() != 'All':
            try:
                s_int = int(season)
                df_s = self.performance_df[self.performance_df['season'] == s_int]
                if df_s.empty:
                    return []
                grouped = df_s.groupby('player').agg(
                    matches=('match_id', 'count'),
                    runs=('batting_runs', 'sum'),
                    wickets=('wickets', 'sum'),
                    avg_score=('performance_score', 'mean')
                ).reset_index()

                catalog = []
                for _, r in grouped.iterrows():
                    catalog.append({
                        'player': r['player'],
                        'matches': int(r['matches']),
                        'runs': int(r['runs']),
                        'wickets': int(r['wickets']),
                        'avg_score': round(float(r['avg_score']), 2) if pd.notnull(r['avg_score']) else 0.0,
                        'has_predictions': r['player'] in self.players_with_predictions and s_int in self.test_seasons,
                        'span': f"Season {s_int}"
                    })
                catalog = sorted(catalog, key=lambda x: (x['matches'], x['runs']), reverse=True)
                if limit:
                    return catalog[:limit]
                return catalog
            except (ValueError, TypeError):
                pass

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


    def get_player_future_forecast(self, player_name, season='2024'):
        """
        Retrieve future season multi-output forecast for a player.
        Supports seasons: 2024, 2025, 2026, or 'All'.
        Returns 7 predicted outputs, confidence intervals, multi-year progression.
        """
        self.load_data()
        if self.forecast_df is None or self.forecast_df.empty:
            return None

        p_name = player_name.strip().lower()
        matched_df = self.forecast_df[self.forecast_df['player'].str.lower() == p_name]
        if matched_df.empty:
            matched_df = self.forecast_df[self.forecast_df['player'].str.lower().str.contains(p_name, regex=False)]
            if matched_df.empty:
                return None

        canonical_player = matched_df.iloc[0]['player']
        df_player = self.forecast_df[self.forecast_df['player'] == canonical_player].sort_values(by='season')

        multi_year = []
        for _, row in df_player.iterrows():
            ci_runs = row['confidence_interval_runs']
            ci_wkts = row['confidence_interval_wickets']
            if isinstance(ci_runs, str):
                try:
                    ci_runs = ast.literal_eval(ci_runs)
                except Exception:
                    ci_runs = [0, 0]
            if isinstance(ci_wkts, str):
                try:
                    ci_wkts = ast.literal_eval(ci_wkts)
                except Exception:
                    ci_wkts = [0, 0]

            multi_year.append({
                'player': row['player'],
                'season': int(row['season']),
                'horizon': row.get('horizon', ''),
                'projected_match_runs': round(float(row['projected_match_runs']), 2),
                'projected_season_runs': round(float(row['projected_season_runs']), 1),
                'projected_fours': round(float(row['projected_fours']), 2),
                'projected_season_fours': round(float(row['projected_season_fours']), 1),
                'projected_sixes': round(float(row['projected_sixes']), 2),
                'projected_season_sixes': round(float(row['projected_season_sixes']), 1),
                'projected_strike_rate': round(float(row['projected_strike_rate']), 2),
                'projected_match_wickets': round(float(row['projected_match_wickets']), 2),
                'projected_season_wickets': round(float(row['projected_season_wickets']), 1),
                'projected_economy_rate': round(float(row['projected_economy_rate']), 2),
                'projected_performance_score': round(float(row['projected_performance_score']), 2),
                'projected_role': row['projected_role'],
                'confidence_interval_runs': ci_runs,
                'confidence_interval_wickets': ci_wkts,
                'uncertainty_factor': round(float(row.get('uncertainty_factor', 1.0)), 2),
                'rank_runs': int(row.get('rank_runs', 0)),
                'rank_wickets': int(row.get('rank_wickets', 0)),
                'rank_mvp': int(row.get('rank_mvp', 0))
            })

        target_season = 2024
        if season and str(season).strip() != 'All':
            try:
                target_season = int(season)
            except (ValueError, TypeError):
                target_season = 2024

        primary_record = None
        for rec in multi_year:
            if rec['season'] == target_season:
                primary_record = rec
                break
        if not primary_record and multi_year:
            primary_record = multi_year[0]

        chart_data = {
            'seasons': [str(m['season']) for m in multi_year],
            'runs': [m['projected_season_runs'] for m in multi_year],
            'wickets': [m['projected_season_wickets'] for m in multi_year],
            'mvp_score': [m['projected_performance_score'] for m in multi_year],
            'strike_rate': [m['projected_strike_rate'] for m in multi_year],
            'economy': [m['projected_economy_rate'] for m in multi_year]
        }

        return {
            'player': canonical_player,
            'selected_season': str(target_season) if (season and str(season).strip() != 'All') else 'All',
            'primary': primary_record,
            'multi_year': multi_year,
            'chart_data': chart_data,
            'available_seasons': [str(m['season']) for m in multi_year],
            'total_forecast_players': len(self.players_with_forecasts)
        }

    def get_forecast_rankings(self, season=2024, top_n=5):
        """
        Get top forecasted batsmen, bowlers, and MVPs for a target future season.
        """
        self.load_data()
        if self.forecast_df is None or self.forecast_df.empty:
            return {'top_batsmen': [], 'top_bowlers': [], 'top_mvps': []}

        try:
            s_int = int(season)
        except (ValueError, TypeError):
            s_int = 2024

        df_s = self.forecast_df[self.forecast_df['season'] == s_int]
        if df_s.empty:
            df_s = self.forecast_df[self.forecast_df['season'] == 2024]

        top_batsmen = df_s.sort_values(by='projected_season_runs', ascending=False).head(top_n)
        top_bowlers = df_s.sort_values(by='projected_season_wickets', ascending=False).head(top_n)
        top_mvps = df_s.sort_values(by='projected_performance_score', ascending=False).head(top_n)

        def _clean_rows(df_subset):
            recs = []
            for _, r in df_subset.iterrows():
                ci_runs = r['confidence_interval_runs']
                ci_wkts = r['confidence_interval_wickets']
                if isinstance(ci_runs, str):
                    try:
                        ci_runs = ast.literal_eval(ci_runs)
                    except Exception:
                        ci_runs = [0, 0]
                if isinstance(ci_wkts, str):
                    try:
                        ci_wkts = ast.literal_eval(ci_wkts)
                    except Exception:
                        ci_wkts = [0, 0]
                recs.append({
                    'player': r['player'],
                    'season': int(r['season']),
                    'projected_role': r['projected_role'],
                    'projected_season_runs': round(float(r['projected_season_runs']), 1),
                    'projected_match_runs': round(float(r['projected_match_runs']), 2),
                    'projected_strike_rate': round(float(r['projected_strike_rate']), 2),
                    'projected_season_wickets': round(float(r['projected_season_wickets']), 1),
                    'projected_match_wickets': round(float(r['projected_match_wickets']), 2),
                    'projected_economy_rate': round(float(r['projected_economy_rate']), 2),
                    'projected_performance_score': round(float(r['projected_performance_score']), 2),
                    'confidence_interval_runs': ci_runs,
                    'confidence_interval_wickets': ci_wkts,
                    'rank_runs': int(r.get('rank_runs', 0)),
                    'rank_wickets': int(r.get('rank_wickets', 0)),
                    'rank_mvp': int(r.get('rank_mvp', 0))
                })
            return recs

        return {
            'season': s_int,
            'top_batsmen': _clean_rows(top_batsmen),
            'top_bowlers': _clean_rows(top_bowlers),
            'top_mvps': _clean_rows(top_mvps)
        }

    def get_all_forecasts_catalog(self, season='2024', search_query=None, role_filter=None, limit=None):
        """
        Get all player forecasts for a given season or across all future seasons.
        """
        self.load_data()
        if self.forecast_df is None or self.forecast_df.empty:
            return []

        df = self.forecast_df.copy()
        if season and str(season).strip() != 'All':
            try:
                s_int = int(season)
                df = df[df['season'] == s_int]
            except (ValueError, TypeError):
                df = df[df['season'] == 2024]

        if search_query:
            df = df[df['player'].str.lower().str.contains(search_query.strip().lower(), regex=False)]

        if role_filter and role_filter.lower() != 'all':
            df = df[df['projected_role'].str.lower() == role_filter.strip().lower()]

        df = df.sort_values(by=['rank_mvp', 'projected_performance_score'], ascending=[True, False])

        if limit:
            df = df.head(limit)

        catalog = []
        for _, r in df.iterrows():
            ci_runs = r['confidence_interval_runs']
            ci_wkts = r['confidence_interval_wickets']
            if isinstance(ci_runs, str):
                try:
                    ci_runs = ast.literal_eval(ci_runs)
                except Exception:
                    ci_runs = [0, 0]
            if isinstance(ci_wkts, str):
                try:
                    ci_wkts = ast.literal_eval(ci_wkts)
                except Exception:
                    ci_wkts = [0, 0]
            catalog.append({
                'player': r['player'],
                'season': int(r['season']),
                'horizon': r.get('horizon', ''),
                'projected_role': r['projected_role'],
                'projected_season_runs': round(float(r['projected_season_runs']), 1),
                'projected_match_runs': round(float(r['projected_match_runs']), 2),
                'projected_fours': round(float(r['projected_fours']), 2),
                'projected_sixes': round(float(r['projected_sixes']), 2),
                'projected_strike_rate': round(float(r['projected_strike_rate']), 2),
                'projected_season_wickets': round(float(r['projected_season_wickets']), 1),
                'projected_match_wickets': round(float(r['projected_match_wickets']), 2),
                'projected_economy_rate': round(float(r['projected_economy_rate']), 2),
                'projected_performance_score': round(float(r['projected_performance_score']), 2),
                'confidence_interval_runs': ci_runs,
                'confidence_interval_wickets': ci_wkts,
                'uncertainty_factor': round(float(r.get('uncertainty_factor', 1.0)), 2),
                'rank_runs': int(r.get('rank_runs', 0)),
                'rank_wickets': int(r.get('rank_wickets', 0)),
                'rank_mvp': int(r.get('rank_mvp', 0))
            })
        return catalog


# Singleton instance
data_service = DataService()

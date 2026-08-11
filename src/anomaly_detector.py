import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

class FatigueAnomalyDetector:
    """
    Handles Phase 5 & 6 of the pipeline: Time-Series Volatility & Anomaly Detection.
    Calculates rolling standard deviations of pitch residuals within a game.
    Uses an Isolation Forest to flag severe mechanical degradation and outputs a 0-100 index.
    """
    
    def __init__(self, rolling_window=10, contamination=0.05):
        # How many pitches back we look to measure mechanical consistency
        self.rolling_window = rolling_window
        
        # The expected proportion of "fatigued/broken" pitch sequences in the dataset
        # 0.05 means we expect only the top 5% most volatile sequences to be flagged as severe
        self.contamination = contamination

    def calculate_rolling_volatility(self, df):
        """
        Sorts pitches chronologically and calculates the rolling variance of the 
        XGBoost residuals strictly within the confines of a single game.
        """
        print(f"Calculating {self.rolling_window}-pitch rolling volatility per outing...")
        
        # sort chronologically to simulate how the game actually unfolded
        df = df.sort_values(by=['pitcher', 'game_pk', 'at_bat_number', 'pitch_number'])
        
        target_res_cols = [c for c in df.columns if c.startswith('residual_')]
        self.rolling_cols = []
        
        for col in target_res_cols:
            roll_col_name = col.replace('residual_', 'rolling_sd_')
            self.rolling_cols.append(roll_col_name)
            
            # Calculate rolling standard deviation, partitioned by game
            df[roll_col_name] = df.groupby(['pitcher', 'game_pk'])[col].transform(
                lambda x: x.rolling(window=self.rolling_window, min_periods=self.rolling_window).std()
            )
            
        return df

    def detect_anomalies(self, df):
        """
        Trains an Isolation Forest on the rolling volatility metrics to find 
        structural breaks in mechanical consistency.
        """
        print("Scoring pitch sequences with Isolation Forest...")
        
        # Drop early-game pitches that haven't hit the minimum rolling window yet
        eval_df = df.dropna(subset=self.rolling_cols).copy()
        
        if eval_df.empty:
            raise ValueError("No data left after dropping rolling window NaNs. Dataset may be too small.")
            
        # Initialize the Unsupervised Isolation Forest
        iso = IsolationForest(
            n_estimators=150, 
            contamination=self.contamination, 
            random_state=42,
            n_jobs=-1
        )
        
        # Because we've localized the data TWICE already (Z-scores + Individual XGBoost models),
        # we can safely train a global Isolation Forest. The inputs are pure, context-free variance.
        iso.fit(eval_df[self.rolling_cols])
        
        # raw_scores are negative (closer to 0 is normal, highly negative is anomalous)
        raw_scores = iso.score_samples(eval_df[self.rolling_cols])
        
        # Invert the scores so that "Higher = More Anomalous"
        inverted_scores = -raw_scores.reshape(-1, 1)
        
        # Scale to a clean 0-100 "Degradation Index" for the front office dashboard
        print("Scaling anomaly scores into 0-100 Degradation Index...")
        scaler = MinMaxScaler(feature_range=(0, 100))
        eval_df['degradation_index'] = scaler.fit_transform(inverted_scores)
        
        # Create a hard boolean flag for the severe alerts
        eval_df['severe_fatigue_alert'] = eval_df['degradation_index'] >= 95.0
        
        return eval_df

    def execute_pipeline(self, df):
        """Runs the final anomaly detection pipeline."""
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided to FatigueAnomalyDetector.")
            
        rolling_df = self.calculate_rolling_volatility(df)
        scored_df = self.detect_anomalies(rolling_df)
        
        print("Pipeline Complete! Degradation Indices successfully generated.")
        return scored_df

if __name__ == "__main__":
    residuals_path = "data/processed/model_residuals.parquet"
    
    if not os.path.exists(residuals_path):
        print(f"Processed data not found at {residuals_path}. Please run contextual_model.py first.")
    else:
        # Load the residuals
        df = pd.read_parquet(residuals_path)
        
        detector = FatigueAnomalyDetector(rolling_window=10, contamination=0.05)
        final_df = detector.execute_pipeline(df)
        
        # Save the final dataset
        save_path = "data/processed/final_fatigue_scores.parquet"
        final_df.to_parquet(save_path, engine='fastparquet')
        
        print(f"Final scored dataset saved to {save_path}")
        
        # Preview the highest degradation index pitches
        print("\nTop 10 Most Degraded Pitch Sequences:")
        alerts = final_df.sort_values(by='degradation_index', ascending=False)
        print(alerts[['player_name', 'game_date', 'pitch_number', 'release_speed', 'degradation_index']].head(10))
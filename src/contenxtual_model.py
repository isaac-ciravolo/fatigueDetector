import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import os

class ContextualFilter:
    """
    Handles Phase 3 & 4 of the pipeline: Contextual Regression & Residuals.
    Trains an individual XGBoost model for each pitcher to predict what their
    mechanics *should* look like based on the game state.
    Calculates the residuals (Actual - Expected) to feed into the anomaly detector.
    """
    def __init__(self, min_pitches=250):
        # The minimum number of pitches a pitcher must have thrown in the dataset to get a model
        # Relievers easily clear 250 pitches over 2-3 years.
        self.min_pitches = min_pitches
        
        # Environmental and game-state features to train on
        self.feature_cols = ['is_lhb', 'is_stretch', 'score_diff_abs']
        
        # The standardized target metrics to predict
        self.target_cols = [
            'release_pos_x_zscore', 
            'release_pos_z_zscore', 
            'release_extension_zscore', 
            'arm_angle_zscore'
        ]

    def prepare_data(self, df):
        """One-hot encodes categorical variables (like pitch_type) for XGBoost."""
        print("Preparing features for XGBoost...")
        
        # Generate context features on the fly if missing
        if 'is_lhb' not in df.columns and 'stand' in df.columns:
            df['is_lhb'] = (df['stand'] == 'L').astype(int)
        elif 'is_lhb' not in df.columns:
            df['is_lhb'] = 0 # Fallback
            
        if 'is_stretch' not in df.columns:
            base_cols = [col for col in ['on_1b', 'on_2b', 'on_3b'] if col in df.columns]
            if base_cols:
                df['is_stretch'] = (df[base_cols].notnull().any(axis=1)).astype(int)
            else:
                df['is_stretch'] = 0 # Fallback
                
        if 'score_diff_abs' not in df.columns:
            if 'bat_score' in df.columns and 'fld_score' in df.columns:
                df['score_diff_abs'] = (df['bat_score'] - df['fld_score']).abs()
            elif 'home_score_diff' in df.columns:
                df['score_diff_abs'] = df['home_score_diff'].abs()
            else:
                df['score_diff_abs'] = 0 # Fallback

        
        # We must one-hot encode pitch_type because it is categorical, not ordinal
        if 'pitch_type' in df.columns:
            # pd.get_dummies creates binary columns for every pitch type (e.g., pitch_type_FF, pitch_type_SL)
            df = pd.get_dummies(df, columns=['pitch_type'], prefix='pitch_type', dtype=int)
            
            # Dynamically add these new one-hot columns to our feature list
            pitch_type_cols = [c for c in df.columns if c.startswith('pitch_type_')]
            self.full_feature_cols = self.feature_cols + pitch_type_cols
        else:
            self.full_feature_cols = self.feature_cols
            
        return df

    def train_models_and_get_residuals(self, df):
        """
        Loops through every pitcher, trains their personal model, 
        predicts expected mechanics, and calculates residuals.
        """
        print(f"Training individualized XGBoost models for pitchers with >= {self.min_pitches} pitches...")
        
        # Ensure target columns exist in the dataframe (handle cases where arm_angle might be missing)
        available_targets = [col for col in self.target_cols if col in df.columns]
        
        # Force target columns to standard float64 to avoid Pandas <NA> vs numpy NaN TypeErrors
        for col in available_targets:
            df[col] = df[col].astype('float64')
        
        # Create empty columns to store the predictions and residuals
        for col in available_targets:
            df[f'expected_{col}'] = np.nan
            df[f'residual_{col}'] = np.nan
            
        # Define the base XGBoost model
        # We use a relatively shallow tree (max_depth=3) to prevent overfitting to noise
        base_estimator = xgb.XGBRegressor(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1, 
            n_jobs=-1,
            random_state=42
        )
        
        # MultiOutputRegressor allows one XGBoost model to predict all 3-4 target metrics simultaneously
        multi_model = MultiOutputRegressor(base_estimator)
        
        pitcher_count = 0
        total_pitchers = df['pitcher'].nunique()
        
        # Group by pitcher and train
        for pitcher_id, group_indices in df.groupby('pitcher').groups.items():
            pitcher_count += 1
            if pitcher_count % 100 == 0:
                print(f"  -> Processed {pitcher_count}/{total_pitchers} pitchers...")
                
            pitcher_data = df.loc[group_indices]
            
            # Drop rows where target variables are missing
            valid_idx = pitcher_data[available_targets].dropna().index
            train_data = pitcher_data.loc[valid_idx]
            
            if len(train_data) < self.min_pitches:
                # If a call-up doesn't have enough valid data to train a reliable model, 
                # we assume their expected Z-score is 0.0 (their baseline)
                df.loc[group_indices, [f'expected_{c}' for c in available_targets]] = 0.0
                for col in available_targets:
                    df.loc[group_indices, f'residual_{col}'] = df.loc[group_indices, col] - 0.0
                continue
                
            X_train = train_data[self.full_feature_cols]
            Y_train = train_data[available_targets]
            
            # Train the pitcher's personal model on valid pitches
            multi_model.fit(X_train, Y_train)
            
            # Predict what their mechanics SHOULD be based on the game states
            # Predict on the full pitcher dataset (XGBoost handles missing X variables natively)
            X_all = pitcher_data[self.full_feature_cols]
            predictions = multi_model.predict(X_all)
            
            # Save predictions
            for i, col in enumerate(available_targets):
                df.loc[group_indices, f'expected_{col}'] = predictions[:, i]
                
                # Calculate Residual: Actual Z-Score minus Expected Z-Score
                df.loc[group_indices, f'residual_{col}'] = df.loc[group_indices, col] - predictions[:, i]
                
        print("Model training and residual calculation complete.")
        return df

    def execute_pipeline(self, df):
        """Runs the full Step 3 pipeline."""
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided to ContextualFilter.")
            
        df = self.prepare_data(df)
        residual_df = self.train_models_and_get_residuals(df)
        
        return residual_df


# Example Execution block
if __name__ == "__main__":
    zscored_path = "data/processed/zscored_statcast.parquet"
    
    if not os.path.exists(zscored_path):
        print(f"Processed data not found at {zscored_path}. Please run baseline_scaler.py first.")
    else:
        # Load the Z-scored data
        print("Loading Z-scored dataset...")
        df = pd.read_parquet(zscored_path)
        
        # Run Step 3
        modeler = ContextualFilter()
        residual_df = modeler.execute_pipeline(df)
        
        # Save the dataset with residuals for the Anomaly Detector
        save_path = "data/processed/model_residuals.parquet"
        residual_df.to_parquet(save_path, engine='fastparquet')
        
        print(f"Step 3 Complete: Residuals saved to {save_path}")
        
        # Preview the residuals for a pitcher
        res_cols = [c for c in residual_df.columns if 'residual' in c]
        preview_cols = [c for c in ['pitcher', 'game_date', 'is_stretch'] if c in residual_df.columns]
        print("\nPreview of Contextual Residuals (First 5 Pitches):")
        print(residual_df[preview_cols + res_cols].head())
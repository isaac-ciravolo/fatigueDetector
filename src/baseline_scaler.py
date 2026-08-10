import pandas as pd
import numpy as np

class BaselineScaler:
    """
    Handles Phase 1 & 2 of the pipeline: Intra-Season Normalization.
    Transforms raw biomechanical metrics into Z-scores relative to the 
    pitcher's own seasonal baseline for that specific pitch type.
    """
    
    def __init__(self, target_cols=None, min_sample_size=50):
        if target_cols is None:
            self.target_cols = [
                'release_pos_x', 'release_pos_z', 
                'release_extension', 'arm_angle'
            ]
        else:
            self.target_cols = target_cols
            
        # Minimum pitches thrown in a season required to use a season-specific baseline
        self.min_sample_size = min_sample_size

    def compute_baselines(self, df):
        """
        Calculates both season-specific and multi-year fallback baselines 
        for every pitcher and pitch type combination.
        """
        print("Calculating intra-season baselines for all pitchers...")
        
        # 1. Calculate Season-Specific Baselines (Grouped by Year)
        season_stats = df.groupby(['pitcher', 'pitch_type', 'game_year'])[self.target_cols].agg(
            ['mean', 'std', 'count']
        ).reset_index()
        
        # Flatten the MultiIndex columns created by agg()
        season_stats.columns = ['pitcher', 'pitch_type', 'game_year'] + \
            [f"{col}_{stat}" for col in self.target_cols for stat in ['mean', 'std', 'count']]
        
        # 2. Calculate Multi-Year Fallback Baselines (Ignoring Year)
        # Used for early-season games where a pitcher hasn't thrown a pitch enough times yet
        career_stats = df.groupby(['pitcher', 'pitch_type'])[self.target_cols].agg(
            ['mean', 'std']
        ).reset_index()
        
        career_stats.columns = ['pitcher', 'pitch_type'] + \
            [f"{col}_career_{stat}" for col in self.target_cols for stat in ['mean', 'std']]
        
        # 3. Merge baselines together
        baselines = pd.merge(season_stats, career_stats, on=['pitcher', 'pitch_type'], how='left')
        
        return baselines

    def apply_zscores(self, df, baselines):
        """
        Merges baselines into the main dataset and calculates Z-Scores.
        Applies the fallback logic for small sample sizes.
        """
        print("Standardizing mechanics into Z-scores...")
        
        # Merge the computed baselines back into the pitch-by-pitch dataframe
        df = pd.merge(df, baselines, on=['pitcher', 'pitch_type', 'game_year'], how='left')
        
        for col in self.target_cols:
            # Check if the season sample size is large enough
            # Fill missing counts with 0 to prevent pandas NA ambiguity in np.where
            sufficient_sample = df[f"{col}_count"].fillna(0) >= self.min_sample_size
            
            mean_to_use = np.where(sufficient_sample, df[f"{col}_mean"], df[f"{col}_career_mean"])
            std_to_use = np.where(sufficient_sample, df[f"{col}_std"], df[f"{col}_career_std"])
            
            # Prevent division by zero if standard deviation is exactly 0.0 (e.g., highly imputed data)
            std_to_use = np.where(std_to_use == 0, 0.0001, std_to_use)
            
            # Calculate the Z-score: (Actual - Mean) / Standard Deviation
            zscore_col_name = f"{col}_zscore"
            df[zscore_col_name] = (df[col] - mean_to_use) / std_to_use
            
            # Drop the intermediate baseline columns to keep the dataframe clean
            cols_to_drop = [
                f"{col}_mean", f"{col}_std", f"{col}_count", 
                f"{col}_career_mean", f"{col}_career_std"
            ]
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            
        return df

    def execute_pipeline(self, df):
        """Runs the full Step 2 standardization pipeline."""
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided to BaselineScaler.")
            
        baselines = self.compute_baselines(df)
        standardized_df = self.apply_zscores(df, baselines)
        
        print("Step 2 Complete: Biomechanical metrics successfully standardized into Z-scores.")
        return standardized_df


# Example Execution block
if __name__ == "__main__":
    import os
    
    path = "data/raw/statcast.parquet"
    
    if not os.path.exists(path):
        print(f"Processed data not found at {path}. Please run data_loader.py first.")
    else:
        # Instantly load the cleaned and imputed data
        raw_df = pd.read_parquet(path)
        
        # Run Step 2 to standardize it
        scaler = BaselineScaler()
        scaled_df = scaler.execute_pipeline(raw_df)
        
        # Save the Z-scored data for Step 3
        scaled_df.to_parquet("data/processed/zscored_statcast.parquet", engine='fastparquet')
        
        # Preview the new Z-score columns
        z_cols = [col for col in scaled_df.columns if 'zscore' in col]
        print("\nPreview of Standardized Z-Scores (First 5 Pitches):")
        print(scaled_df[['pitcher', 'pitch_type', 'game_year'] + z_cols].head())
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pybaseball
import os
import time
import datetime

class StatcastDataLoader:
    """
    Handles the ingestion, cleaning, and preprocessing of Statcast pitch data using pybaseball.
    Prepares the dataset for season-specific normalization and time-series modeling.
    """
    
    def __init__(self, start_date, end_date, cache_filepath="data/raw/statcast_cache.parquet"):
        self.start_date = start_date
        self.end_date = end_date
        self.cache_filepath = cache_filepath
        self.raw_data = None
        self.processed_data = None
        
        # The exact variables identified in the data mapping phase
        self.required_columns = [
            'pitcher', 'player_name', 'pitch_type', 'game_pk', 'game_date', 'game_year', 
            'at_bat_number', 'pitch_number', 'stand', 'on_1b', 'on_2b', 'on_3b', 
            'outs_when_up', 'inning', 'home_score_diff', 'release_pos_x', 
            'release_pos_z', 'release_extension', 'arm_angle', 'release_speed', 
            'pfx_x', 'pfx_z', 'events', 'des'
        ]
        
        self.biomech_cols = [
            'release_pos_x', 'release_pos_z', 'release_extension', 'arm_angle'
        ]

    def load_data(self):
        """Loads data using pybaseball or from local cache, then filters down to required columns."""
        if os.path.exists(self.cache_filepath):
            print(f"Loading cached data from {self.cache_filepath}...")
            df = pd.read_parquet(self.cache_filepath)
        else:
            print(f"Fetching data from pybaseball ({self.start_date} to {self.end_date}). This may take a while...")
            # Enable pybaseball's internal caching to speed up subsequent identical requests
            pybaseball.cache.enable() 
            
            # Baseball Savant will abort the connection if we ask for 6 months of data at once 
            # due to thread limits. We mitigate this by chunking the request week-by-week.
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            dfs = []
            current = start
            while current <= end:
                next_date = min(current + pd.Timedelta(days=7), end)
                c_start = current.strftime('%Y-%m-%d')
                c_end = next_date.strftime('%Y-%m-%d')
                
                print(f"  -> Fetching chunk: {c_start} to {c_end}...")
                try:
                    # parallel=False prevents it from spawning too many threads and crashing the socket
                    chunk_df = pybaseball.statcast(start_dt=c_start, end_dt=c_end, parallel=False)
                    if chunk_df is not None and not chunk_df.empty:
                        dfs.append(chunk_df)
                except Exception as e:
                    print(f"  -> Warning: Failed to fetch {c_start} to {c_end}: {e}")
                
                current = next_date + pd.Timedelta(days=1)
                time.sleep(1) # Be nice to the Savant servers to avoid rate limiting
                
            if not dfs:
                raise ValueError("Failed to fetch any data. Please check your connection and dates.")
                
            df = pd.concat(dfs, ignore_index=True)
            
            # Create the directory if it doesn't exist and save for future runs
            os.makedirs(os.path.dirname(self.cache_filepath), exist_ok=True)
            print(f"Saving fetched data to {self.cache_filepath} for future use...")
            df.to_parquet(self.cache_filepath, engine='fastparquet')
        
        # Clean column names (Statcast CSVs sometimes have trailing spaces and casing differences)
        df.columns = df.columns.str.strip().str.lower()
        
        # pybaseball does not return 'home_score_diff' natively, so we engineer it here
        if 'home_score_diff' not in df.columns and 'home_score' in df.columns and 'away_score' in df.columns:
            df['home_score_diff'] = df['home_score'] - df['away_score']
            
        # Keep only required columns that actually exist in the CSV
        available_cols = [col for col in self.required_columns if col in df.columns]
        
        # Print a warning if any required columns are missing
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            print(f"WARNING: The following required columns are missing from the CSV: {missing_cols}")
            
        self.raw_data = df[available_cols].copy()
        
        # Ensure proper data types safely
        if 'game_date' in self.raw_data.columns:
            self.raw_data['game_date'] = pd.to_datetime(self.raw_data['game_date'])
        
        return self.raw_data

    def engineer_context_features(self, df):
        """Creates binary flags and engineered features for the regression model."""
        print("Engineering game-state context features...")
        
        # Batter handedness flag (1 if Left, 0 if Right)
        if 'stand' in df.columns:
            df['is_lhb'] = (df['stand'] == 'L').astype(int)
        
        # Pitching from the stretch flag (runners on base)
        base_cols = [col for col in ['on_1b', 'on_2b', 'on_3b'] if col in df.columns]
        if base_cols:
            df['is_stretch'] = (df[base_cols].notnull().any(axis=1)).astype(int)
        
        # Approximate leverage using absolute score differential 
        if 'home_score_diff' in df.columns:
            df['score_diff_abs'] = df['home_score_diff'].abs()
        
        return df

    def filter_and_clean(self, df):
        """Removes unknown pitch types and invalid game states."""
        print("Filtering invalid pitches...")
        
        # Remove pitches with unknown or unclassified pitch types (e.g., pitchouts 'PO', unknown 'UN')
        valid_pitch_types = ['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'FS', 'KC', 'ST', 'SV']
        if 'pitch_type' in df.columns:
            df = df[df['pitch_type'].isin(valid_pitch_types)]
        else:
            print("WARNING: 'pitch_type' column missing. Skipping pitch classification filter.")
        
        # Drop rows where critical grouping variables are missing
        subset_cols = [col for col in ['pitcher', 'game_pk', 'at_bat_number', 'pitch_number'] if col in df.columns]
        if subset_cols:
            df = df.dropna(subset=subset_cols)
        
        return df

    def impute_missing_biomechanics(self, df):
        """
        Uses K-Nearest Neighbors to impute missing tracking reads.
        We CANNOT drop missing rows because doing so breaks the sequential 
        time-series rolling window needed for fatigue detection.
        """
        print("Imputing missing biomechanical tracking data using KNN...")
        
        # We only impute within the same pitcher's dataset to maintain personal biomechanics
        imputed_dfs = []
        
        # Group by pitcher to keep imputation local to their specific mechanics
        for pitcher_id, group in df.groupby('pitcher'):
            group_copy = group.copy()
            
            # If a pitcher is missing 100% of a specific metric (e.g., pre-2020 no arm_angle), 
            # KNN will fail. We fill with group mean as a fallback.
            for col in self.biomech_cols:
                if group_copy[col].isnull().all():
                    group_copy[col] = 0.0 # Fallback placeholder if no data exists
            
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            
            # Impute only the biomechanical columns
            group_copy[self.biomech_cols] = imputer.fit_transform(group_copy[self.biomech_cols])
            imputed_dfs.append(group_copy)
            
        return pd.concat(imputed_dfs, ignore_index=True)

    def execute_pipeline(self):
        """Runs the full Step 1 pipeline."""
        df = self.load_data()
        df = self.filter_and_clean(df)
        df = self.engineer_context_features(df)
        self.processed_data = self.impute_missing_biomechanics(df)
        
        print("Step 1 Complete: Data loaded, cleaned, engineered, and imputed.")
        return self.processed_data

# Example Execution block (can be commented out in production)
if __name__ == "__main__":
    # Fetching data for a sample window (e.g., the 2024 regular season)
    loader = StatcastDataLoader(
        start_date="2023-03-28", 
        end_date="2026-09-29", 
        cache_filepath="data/raw/statcast.parquet"
    )
    clean_df = loader.execute_pipeline()
    print(clean_df.head())
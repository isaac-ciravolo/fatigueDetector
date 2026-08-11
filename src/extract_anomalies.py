import pandas as pd
import json
import os

def extract_top_anomalies(parquet_path, output_json_path, top_n=10):
    """
    Mines the fully scored parquet file for the top N most severe mechanical breakdowns.
    Automatically determines the primary point of failure by comparing rolling variances.
    """
    print(f"Loading scored dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Format date for JSON serialization
    if pd.api.types.is_datetime64_any_dtype(df['game_date']):
        df['game_date_str'] = df['game_date'].dt.strftime('%b %d, %Y')
    else:
        df['game_date_str'] = df['game_date'].astype(str)

    # Dictionary to translate dataset columns to UI labels
    failure_mapping = {
        'rolling_sd_release_pos_x_zscore': 'Horizontal Release (x-axis)',
        'rolling_sd_release_pos_z_zscore': 'Vertical Release (z-axis)',
        'rolling_sd_release_extension_zscore': 'Release Extension',
        'rolling_sd_arm_angle_zscore': 'Posture (Arm Angle)'
    }
    
    # Find available rolling columns in the dataset
    rolling_cols = [col for col in failure_mapping.keys() if col in df.columns]

    # Find the unique outings with the highest peak Degradation Index
    outings = df.groupby(['player_name', 'game_pk', 'game_date_str']).agg(
        peak_di=('degradation_index', 'max')
    ).reset_index()
    
    top_outings = outings.sort_values(by='peak_di', ascending=False).head(top_n)
    extracted_datasets = {}

    print(f"\nExtracting timeline for top {top_n} anomalies...")
    
    for _, row in top_outings.iterrows():
        pitcher_name = row['player_name']
        game_pk = row['game_pk']
        game_date = row['game_date_str']
        peak_di = row['peak_di']
        
        dropdown_key = f"{pitcher_name} ({game_date})"
        print(f" -> Processing {dropdown_key} (Peak DI: {peak_di:.1f})")
        
        # Extract the chronological sequence for this specific game
        game_df = df[(df['player_name'] == pitcher_name) & (df['game_pk'] == game_pk)].copy()
        game_df = game_df.sort_values(by=['at_bat_number', 'pitch_number'])
        
        #Create a continuous pitch count for the entire outing 
        game_df['cumulative_pitch'] = range(1, len(game_df) + 1)
        
        # Find the exact pitch where the anomaly peaked
        peak_pitch = game_df.loc[game_df['degradation_index'].idxmax()]
        peak_velo = peak_pitch.get('release_speed', 0.0)
        
        #  DETERMINE PRIMARY FAILURE 
        # Find which specific metric had the highest rolling variance at the time of the spike
        primary_failure_label = "Unknown"
        if rolling_cols:
            max_col = peak_pitch[rolling_cols].astype(float).idxmax()
            primary_failure_label = failure_mapping.get(max_col, "Unknown")
        
        # Build the pitch-by-pitch timeline (last 40 pitches for UI performance)
        pitch_sequence = []
        game_df_tail = game_df.tail(40) 
        
        for _, pitch in game_df_tail.iterrows():
            di_val = pitch['degradation_index']
            desc_text = pitch.get('des', '')
            event_text = pitch.get('events', None)
            
            full_desc = str(desc_text) if pd.notna(event_text) and pd.notna(desc_text) else ""
            
            pitch_data = {
                # Map to the new continuous sequence instead of the at-bat pitch number
                "pitch": int(pitch['cumulative_pitch']),
                "velo": float(pitch['release_speed']) if pd.notna(pitch['release_speed']) else 0.0,
                "di": float(di_val) if pd.notna(di_val) else 0.0,
                "desc": full_desc
            }
            
            if pd.notna(di_val) and di_val >= 95.0:
                pitch_data["alert"] = True
                
            pitch_sequence.append(pitch_data)

        # Structure for the JSON file
        extracted_datasets[dropdown_key] = {
            "name": pitcher_name,
            "date": game_date,
            "peakVelo": f"{peak_velo:.1f} mph" if pd.notna(peak_velo) else "N/A",
            "peakDI": f"{peak_di:.1f}",
            "primary_failure": primary_failure_label,  
            "score_before": "TBD", # updated in fix_scores.py
            "score_after": "TBD",
            "data": pitch_sequence
        }

    # Export to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(extracted_datasets, f, indent=4)
        
    print(f"\nSuccess! Saved anomalies to {output_json_path}")

if __name__ == "__main__":
    PARQUET_FILE = "data/processed/final_fatigue_scores.parquet"
    JSON_OUT = "dashboard/anomalies.json"
    
    if os.path.exists(PARQUET_FILE):
        extract_top_anomalies(PARQUET_FILE, JSON_OUT, top_n=10)
    else:
        print(f"Could not find {PARQUET_FILE}.")
import pandas as pd
import json
import os

def append_mechanical_shifts():
    print("Loading parquet dataset...")
    df = pd.read_parquet("./data/processed/final_fatigue_scores.parquet")
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    
    json_path = "dashboard/anomalies.json"
    with open(json_path, 'r') as f:
        anomalies = json.load(f)
        
    # Mapping the UI strings to actual Statcast column names
    statcast_mapping = {
        "Vertical Release (z-axis)": ("release_pos_z", "ft"),
        "Horizontal Release (x-axis)": ("release_pos_x", "ft"),
        "Release Extension": ("release_extension", "ft"),
        "Posture (Arm Angle)": ("release_pos_z", "ft") # Using Z-axis drop as the proxy for arm slot fatigue
    }
        
    print("Extracting mechanical shifts...")
    for key, data in anomalies.items():
        pitcher_name = data['name']
        game_date = data['date']
        primary_failure = data.get('primary_failure', 'Release Extension')
        
        col_name, unit = statcast_mapping.get(primary_failure, ("release_extension", "ft"))
        
        game_df = df[(df['player_name'] == pitcher_name) & (df['game_date'] == game_date)].sort_index()
        
        if not game_df.empty and col_name in game_df.columns:
            # Calculate Baseline: Average of the first 15 pitches of the outing
            baseline_val = game_df.head(15)[col_name].mean()
            
            # Calculate Fatigued: The exact value at the peak DI anomaly
            max_di_idx = game_df['degradation_index'].idxmax()
            fatigued_val = game_df.loc[max_di_idx, col_name]
            
            # Save the formatted values back to the JSON dictionary
            data['breakdown_baseline'] = f"{baseline_val:.2f} {unit}"
            data['breakdown_fatigued'] = f"{fatigued_val:.2f} {unit}"
        else:
            data['breakdown_baseline'] = "N/A"
            data['breakdown_fatigued'] = "N/A"
            
    with open(json_path, 'w') as f:
        json.dump(anomalies, f, indent=4)
        
    print("Successfully updated anomalies.json with before-and-after mechanics.")

if __name__ == "__main__":
    append_mechanical_shifts()
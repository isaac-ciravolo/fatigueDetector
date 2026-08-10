import pandas as pd
import json

def fix_json_scores():
    print("Loading parquet dataset...")
    df = pd.read_parquet("./data/processed/final_fatigue_scores.parquet")
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    
    json_path = "./dashboard/anomalies.json"
    with open(json_path, 'r') as f:
        anomalies = json.load(f)
        
    print("Updating case studies...")
    for key, data in anomalies.items():
        pitcher_name = data['name']
        game_date = data['date']
        
        game_df = df[(df['player_name'] == pitcher_name) & (df['game_date'] == game_date)]
        
        if game_df.empty:
            continue
            
        max_di_idx = game_df['degradation_index'].idxmax()
        target_inning = game_df.loc[max_di_idx, 'inning']
        inning_df = game_df[game_df['inning'] == target_inning].sort_index()
        
        if not inning_df.empty:
            first_pitch = inning_df.iloc[0]
            
            home_team = str(first_pitch.get('home_team', 'HOME'))
            away_team = str(first_pitch.get('away_team', 'AWAY'))
            
            home_score_before = int(first_pitch.get('home_score', 0))
            away_score_before = int(first_pitch.get('away_score', 0))
            
            # Calculate post-play score using post_bat_score
            runs_scored = int(inning_df['post_bat_score'].max() - inning_df['bat_score'].min())
            is_home_batting = (first_pitch.get('inning_topbot') == 'Bot')
            
            home_score_after = home_score_before + (runs_scored if is_home_batting else 0)
            away_score_after = away_score_before + (runs_scored if not is_home_batting else 0)
            
            data['score_before'] = f"{away_team} {away_score_before} - {home_team} {home_score_before}"
            data['score_after'] = f"{away_team} {away_score_after} - {home_team} {home_score_after}"
            
    with open(json_path, 'w') as f:
        json.dump(anomalies, f, indent=4)
        
    print("Successfully updated anomalies.json with post-play scores.")

if __name__ == "__main__":
    fix_json_scores()
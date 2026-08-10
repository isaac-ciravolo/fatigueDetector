import pandas as pd

def check_snell_game_6():
    print("Loading parquet dataset...")
    df = pd.read_parquet("./data/processed/final_fatigue_scores.parquet")
    
    # Ensure the date format is standardized for comparison
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    
    # Filter for Snell's exact Game 6 World Series start
    snell_df = df[(df['player_name'].str.contains('Snell')) & (df['game_date'] == '2020-10-27')].sort_index()
    
    if snell_df.empty:
        print("Could not find Snell's 2020 World Series Game 6 in the dataset.")
        print("Check if your dataset includes the 2020 postseason.")
        return
        
    # Isolate the infamous 6th inning
    inning_6 = snell_df[snell_df['inning'] == 6]
    
    print("-" * 40)
    print("Blake Snell - 2020 WS Game 6 (6th Inning)")
    print("-" * 40)
    
    for idx, row in inning_6.iterrows():
        pitch_num = row.get('pitch_number', '?')
        pitch_type = row.get('pitch_type', '?')
        velo = row.get('release_speed', '?')
        di = row.get('degradation_index', 0)
        batter = row.get('batter', '?')
        
        print(f"Pitch {pitch_num} vs Batter {batter}: {pitch_type} | Velo: {velo} mph | DI: {di:.1f}")

if __name__ == "__main__":
    check_snell_game_6()
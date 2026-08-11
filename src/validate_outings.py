import pandas as pd
import os

def investigate_outing(df, pitcher_name, game_date):
    print(f"\n{'='*60}")
    print(f"PLAY-BY-PLAY RECONSTRUCTION: {pitcher_name} on {game_date}")
    print(f"{'='*60}")
    
    # Filter for the specific game and pitcher
    outing = df[(df['player_name'] == pitcher_name) & (df['game_date'] == game_date)].copy()
    
    if outing.empty:
        print("No data found for this outing.")
        return
        
    # Sort chronologically
    outing = outing.sort_values(by=['at_bat_number', 'pitch_number'])
    
    # Select columns
    cols_to_show = [
        'inning', 'pitch_type', 'release_speed', 
        'degradation_index', 'events', 'des'
    ]
    
    # Ensure columns exist
    available_cols = [c for c in cols_to_show if c in outing.columns]
    
    # Print the timeline, highlighting severe pitches
    for _, pitch in outing.iterrows():
        di = pitch['degradation_index']
        
        # Create a visual alert tag for pitches that crossed the threshold
        alert_tag = " [SEVERE ALERT!]" if pd.notna(di) and di >= 95.0 else ""
        
        di_str = f"{di:.1f}" if pd.notna(di) else "N/A"
        inning = pitch.get('inning', '?')
        velo = pitch.get('release_speed', '?')
        ptype = pitch.get('pitch_type', '?')
        event = pitch.get('events', None)
        desc = pitch.get('des', '')
        
        # Only print the event if something actually happened (e.g., hit, out, walk)
        event_str = f" => Result: {event}" if pd.notna(event) else ""
        
        print(f"Inn {inning} | {ptype} @ {velo}mph | DI: {di_str}{alert_tag}")
        if event_str:
            print(f"  {event_str}")
        if pd.notna(desc) and event_str:
            print(f"  {desc}")


if __name__ == "__main__":
    filepath = "data/processed/final_fatigue_scores.parquet"
    
    if not os.path.exists(filepath):
        print(f"Could not find {filepath}.")
    else:
        # Load the fully scored dataset
        df = pd.read_parquet(filepath)
        
        # Convert game_date to string for easy filtering
        if pd.api.types.is_datetime64_any_dtype(df['game_date']):
            df['game_date'] = df['game_date'].dt.strftime('%Y-%m-%d')
            
        # Investigate our top 2 hits
        investigate_outing(df, "Cortes, Nestor", "2024-08-08")
        investigate_outing(df, "Quantrill, Cal", "2025-06-17")
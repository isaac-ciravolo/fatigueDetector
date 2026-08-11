import pandas as pd
import numpy as np

def evaluate_pipeline(df: pd.DataFrame):
    print("Evaluating Model vs Baseline (Optimized for Late-Game Fatigue)...")

    # Calculate runs allowed in inning
    if 'runs_allowed_in_inning' not in df.columns:
        inning_max_score = df.groupby(['game_pk', 'pitcher', 'inning'])['post_bat_score'].transform('max')
        inning_min_score = df.groupby(['game_pk', 'pitcher', 'inning'])['bat_score'].transform('min')
        df['runs_allowed_in_inning'] = inning_max_score - inning_min_score

    df['damage_occurred'] = df['runs_allowed_in_inning'] > 0

    # Fastball Velocity Drop
    if 'pitch_type' in df.columns:
        fastballs = ['FF', 'SI', 'FC']
        df['is_fastball'] = df['pitch_type'].isin(fastballs)
    else:
        game_max_velo = df.groupby(['game_pk', 'pitcher'])['release_speed'].transform('max')
        df['is_fastball'] = df['release_speed'] >= (game_max_velo - 6.0)
    
    fb_df = df[df['is_fastball']]
    pitcher_avg_fb = fb_df.groupby(['game_pk', 'pitcher']).head(15).groupby(['game_pk', 'pitcher'])['release_speed'].mean().reset_index()
    pitcher_avg_fb.rename(columns={'release_speed': 'avg_early_fb_velo'}, inplace=True)
    
    df = df.merge(pitcher_avg_fb, on=['game_pk', 'pitcher'], how='left')
    df['velo_drop_alert'] = (df['is_fastball']) & ((df['avg_early_fb_velo'] - df['release_speed']) >= 2.5)

    # Localized DI
    df['outing_di_percentile'] = df.groupby(['game_pk', 'pitcher'])['degradation_index'].rank(pct=True)
    df['di_alert_raw'] = df['outing_di_percentile'] >= 0.90 

    # Aggregate Results per Inning (WITH FATIGUE FILTERS)
    inning_summary = df.groupby(['game_pk', 'pitcher', 'inning']).agg({
        'velo_drop_alert': 'max',  
        'di_alert_raw': 'max',         
        'damage_occurred': 'max',
        'pitch_number': 'count' # FIXED: 'count' calculates total pitches in the inning
    }).reset_index()
    
    inning_summary.rename(columns={'pitch_number': 'total_pitches_in_inning'}, inplace=True)

    # Apply Baseball Logic Filters:
    inning_summary = inning_summary[inning_summary['inning'] >= 4].copy()
    inning_summary['di_alert'] = (inning_summary['di_alert_raw'] == True) & (inning_summary['total_pitches_in_inning'] >= 15)

    # Calculate Metrics
    def calc_metrics(alert_col):
        tp = len(inning_summary[(inning_summary[alert_col] == True) & (inning_summary['damage_occurred'] == True)])
        fp = len(inning_summary[(inning_summary[alert_col] == True) & (inning_summary['damage_occurred'] == False)])
        fn = len(inning_summary[(inning_summary[alert_col] == False) & (inning_summary['damage_occurred'] == True)])
        tn = len(inning_summary[(inning_summary[alert_col] == False) & (inning_summary['damage_occurred'] == False)])
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 
        return recall, fpr

    baseline_recall, baseline_fpr = calc_metrics('velo_drop_alert')
    di_recall, di_fpr = calc_metrics('di_alert')

    print("-" * 30)
    print("BASELINE (1.5mph Fastball Drop | Innings 4+)")
    print(f"Recall (Damage Caught): {baseline_recall:.1%}")
    print(f"False Alarm Rate (FPR): {baseline_fpr:.1%}")
    
    print("-" * 30)
    print("PIPELINE (Sustained DI Spike | Innings 4+)")
    print(f"Recall (Damage Caught): {di_recall:.1%}")
    print(f"False Alarm Rate (FPR): {di_fpr:.1%}")

if __name__ == "__main__":
    df = pd.read_parquet("./data/processed/final_fatigue_scores.parquet")
    evaluate_pipeline(df)
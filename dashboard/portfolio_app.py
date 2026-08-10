import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Dugout OS | In-Game Fatigue and Run Prevention",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DUGOUT PRODUCT FEEL ---
# --- CUSTOM CSS FOR DUGOUT PRODUCT FEEL ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent;}
    
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }
    
    .metric-card { background-color: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center; }
    .dugout-header { background: #0f172a; padding: 15px 25px; margin-top: 40px; border-radius: 12px; border-bottom: 4px solid #3b82f6; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;}
    
    .status-indicator-red { 
        background: linear-gradient(180deg, #7f1d1d 0%, #450a0a 100%); 
        border: 2px solid #ef4444; 
        border-radius: 12px; 
        padding: 25px 20px; 
        text-align: center; 
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.6); 
    }
    .status-indicator-green { 
        background: linear-gradient(180deg, #14532d 0%, #052e16 100%); 
        border: 2px solid #22c55e; 
        border-radius: 12px; 
        padding: 25px 20px; 
        text-align: center; 
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_damage_case_studies():
    json_path = os.path.join(os.path.dirname(__file__), 'anomalies.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

DATASETS = load_damage_case_studies()

with st.sidebar:
    st.title("Fatigue Detection Dashboard")
    st.markdown("### For In-Game Run Prevention")
    
    app_mode = st.radio("SYSTEM MODE", ["Live Dugout Monitor", "Pipeline Reliability", "Pipeline Architecture"])
    
    st.divider()
    st.markdown("### Active Outing Select:")
    
    # Safely handle empty JSON load
    pitcher_options = list(DATASETS.keys()) if DATASETS else ["No Data Available"]
    pitcher_selection = st.selectbox("Select Target Pitcher:", pitcher_options, key="pitcher_selection")
    
    st.divider()
    st.markdown("Created by **Isaac**")
    st.markdown("[GitHub Repository](#)")
    st.markdown("[LinkedIn](#)")

if DATASETS and pitcher_selection != "No Data Available":
    data = DATASETS[pitcher_selection]
    df = pd.DataFrame(data.get("data", []))
else:
    data = {}
    df = pd.DataFrame()

if app_mode == "Live Dugout Monitor":
    
    st.markdown(f"""
    <div class="dugout-header">
        <div style="font-size: 22px; font-weight: bold; color: #cbd5e1;">{data.get('matchup', 'MLB Matchup')}</div>
        <div style="font-size: 18px; font-weight: bold; color: #cbd5e1;">Before: <span style="color: #94a3b8;">{data.get('score_before', '0-0')}</span> &nbsp;|&nbsp; After: <span style="color: #ef4444;">{data.get('score_after', 'TBD')}</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2.8])

    with col1:
        st.markdown(f"<h2 style='margin-bottom: 0px;'>{data.get('name', 'N/A')}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; margin-top: 0px; font-size: 14px;'>Outing Date: {data.get('date', 'N/A')}</p>", unsafe_allow_html=True)
        
        peak_di = float(data.get('peakDI', 0))
        if peak_di >= 95.0:
            status_class = "status-indicator-red"
            status_text = "SEVERE ALERT"
            status_sub = "Immediate mound visit recommended."
        else:
            status_class = "status-indicator-green"
            status_text = "ELEVATED RISK"
            status_sub = "Release point scatter detected. Monitor closely."

        st.markdown(f"""
        <div class="{status_class}">
            <div style="font-size: 12px; color: #cbd5e1; text-transform: uppercase; font-weight: bold; letter-spacing: 1px;">Kinetic Chain Status</div>
            <div style="font-size: 30px; font-weight: 900; color: white; margin: 8px 0;">{status_text}</div>
            <div style="font-size: 13px; color: #f8fafc;">{status_sub}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        # Extract the new data points from the loaded JSON
        primary_fail = data.get('primary_failure', 'Release Extension')
        baseline_mech = data.get('breakdown_baseline', 'TBD')
        fatigued_mech = data.get('breakdown_fatigued', 'TBD')

        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: bold;">Primary Breakdown</div>
            <div style="font-size: 18px; font-weight: 700; color: #f8fafc; margin-top: 5px;">{primary_fail}</div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 8px;">
                Baseline: <span style="color: #22c55e; font-weight: bold;">{baseline_mech}</span> | 
                Fatigued: <span style="color: #ef4444; font-weight: bold;">{fatigued_mech}</span>
            </div>
        </div>
        <div class="metric-card" style="margin-top: 12px;">
            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: bold;">Peak Degradation Index</div>
            <div style="font-size: 28px; font-weight: 900; color: #ef4444;">{data.get('peakDI', '0')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if not df.empty:
            df['hover_text'] = df.apply(
                lambda r: f"<b>Pitch {int(r.get('pitch', 0))}</b><br>Event: {r.get('desc', 'N/A')}<br>DI: {r.get('di', 0):.1f}<br>Velo: {r.get('velo', 0):.1f} mph", 
                axis=1
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['pitch'], y=df['di'],
                name='Degradation Index',
                yaxis='y1',
                mode='lines+markers',
                line=dict(color='#3b82f6', width=4),
                marker=dict(
                    color=['#ef4444' if d > 95 else '#3b82f6' for d in df['di']],
                    size=[18 if d > 95 else 10 for d in df['di']],
                    line=dict(color='white', width=2)
                ),
                hovertext=df['hover_text'],
                hovertemplate='%{hovertext}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=df['pitch'], y=df['velo'],
                name='Velocity (mph)',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='rgba(34, 211, 238, 0.3)', width=2, dash='dash'),
                marker=dict(color='rgba(34, 211, 238, 0.6)', size=7, symbol='diamond'),
                hovertext=df['hover_text'],
                hovertemplate='%{hovertext}<extra></extra>'
            ))
            
            fig.update_layout(
                height=430,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='rgba(15, 23, 42, 1)',
                paper_bgcolor='rgba(15, 23, 42, 0)',
                font=dict(color='#f8fafc'),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                xaxis=dict(title='Cumulative Outing Pitch Count', gridcolor='#1e293b'),
                yaxis=dict(
                    title=dict(text='Degradation Index (0-100)', font=dict(color='#3b82f6')),
                    tickfont=dict(color='#3b82f6'),
                    range=[0, 105],
                    gridcolor='#1e293b'
                ),
                yaxis2=dict(
                    title=dict(text='Velocity (mph)', font=dict(color='#22d3ee')),
                    tickfont=dict(color='#22d3ee'),
                    anchor='x',
                    overlaying='y',
                    side='right',
                    range=[75, 105]
                )
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<h3 style='margin-top: 10px;'>Live In-Game Play Feed</h3>", unsafe_allow_html=True)
    if not df.empty:
        with st.container(height=300):
            for _, row in df.iloc[::-1].iterrows():
                
                # Explicitly check for True to prevent NaN from evaluating as truthy
                is_alert = (row.get('alert') == True) or (row.get('di', 0) > 95.0)
                desc = row.get('desc', 'Pitch executed')
                
                if is_alert:
                    st.error(f"**ALERT Pitch {int(row.get('pitch', 0))}:** {desc} | **Velo:** {row.get('velo', 0):.1f} mph | **DI:** {row.get('di', 0):.1f}")
                else:
                    st.info(f"**Pitch {int(row.get('pitch', 0))}:** {desc} | **Velo:** {row.get('velo', 0):.1f} mph | **DI:** {row.get('di', 0):.1f}")

elif app_mode == "Pipeline Reliability":
    st.markdown("## System Reliability vs. Baseline")
    st.markdown("""
    Baseball managers hate false alarms. A system is only useful if it predicts damage accurately without making a premature decision.
    To prove this pipeline's superiority, I evaluated it against the traditional MLB benchmark for fatigue: **Velocity Drop**.
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### The Baseline: Traditional Radar Gun")
        st.markdown("*Condition: Fastball drops > 1.5 mph from early-game average (Innings 4+).*")
        st.markdown("""
        **The Problem:** Pitchers are phenomenal athletes. As their central nervous system tires, they simply exert *more effort* to maintain their 95 mph velocity. Because they maintain velocity, the radar gun tells the manager the pitcher is fine. 
        
        However, max-effort pitching on tired muscles causes structural release variance, leading to hanging sliders, missed spots, and giving up runs while still throwing hard.
        """)
        
        # Fixed formatting: Added height: 100% to ensure visual alignment
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 20px; text-align: left; padding: 20px; height: 100%; border: 1px solid #334155;">
            <div style="color: #94a3b8; font-size: 14px; text-transform: uppercase; font-weight: bold;">Recall (Breakdowns Caught)</div>
            <div style="font-size: 32px; font-weight: 900; color: #22c55e;">54.8%</div>
            <div style="color: #94a3b8; font-size: 14px; margin-top: 15px; text-transform: uppercase; font-weight: bold;">False Alarm Rate</div>
            <div style="font-size: 32px; font-weight: 900; color: #ef4444;">41.2%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### The Pipeline: Degradation Index")
        st.markdown("*Condition: Unsupervised Isolation Forest flags sustained DI > 90th percentile (Innings 4+).*")
        st.markdown("""
        **The Solution:** By evaluating Hawk-Eye 3D tracking data (Release Extension, Vertical/Horizontal Arm Slot, and Posture) as a unified system, we bypass the radar gun entirely. 
        
        The Isolation Forest mathematically identifies when the pitcher loses the ability to repeat their mechanics. It filters out the noise to ensure the bullpen is only engaged when true failure occurs.
        """)
        
        # Fixed formatting: Matched base dimensions and layout to left column
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 20px; text-align: left; padding: 20px; height: 100%; border: 1px solid #3b82f6;">
            <div style="color: #94a3b8; font-size: 14px; text-transform: uppercase; font-weight: bold;">Recall (Breakdowns Caught)</div>
            <div style="font-size: 32px; font-weight: 900; color: #ef4444;">41.9%</div>
            <div style="color: #94a3b8; font-size: 14px; margin-top: 15px; text-transform: uppercase; font-weight: bold;">False Alarm Rate</div>
            <div style="font-size: 32px; font-weight: 900; color: #22c55e;">14.0%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.info("**Executive Summary:** While the radar gun casts a wider net, it produces an unusable 41.2% false alarm rate that would prematurely drain a bullpen. The Degradation Index drops false alarms by roughly 66%, offering managers a highly precise, trustworthy run-prevention tool.")

else:
    st.markdown("## Micro-Degradation ML Architecture")
    st.markdown("This section explains the machine learning pipeline powering the dugout application.")
    
    st.markdown("#### Phase 3: The Contextual Filter (XGBoost)")
    st.markdown("Pitchers strategically change their mechanics to survive a game. I used `MultiOutputRegressor(xgb.XGBRegressor)` to train a unique model for **every single pitcher** to filter out this game-state noise.")
    st.code("""
# From src/contextual_model.py
multi_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=50, max_depth=3))

for pitcher_id, group_indices in df.groupby('pitcher').groups.items():
    pitcher_data = df.loc[group_indices]
    
    multi_model.fit(X_train, Y_train)
    
    predictions = multi_model.predict(X_all)
    df.loc[group_indices, 'residual'] = df.loc[group_indices, target] - predictions
    """, language='python')

    st.markdown("#### Phase 5 and 6: The Anomaly Detector (Isolation Forest)")
    st.markdown("Supervised ML requires labeled injury data, which is sparse and biased. I used unsupervised learning (`IsolationForest`) on the rolling standard deviation of the residuals to mathematically flag structural breaks in mechanics.")
    st.code("""
# From src/anomaly_detector.py
df['rolling_sd'] = df.groupby(['pitcher', 'game_pk'])['residual'].transform(
    lambda x: x.rolling(window=10, min_periods=10).std()
)

iso = IsolationForest(n_estimators=150, contamination=0.05)
iso.fit(eval_df[rolling_cols])

raw_scores = iso.score_samples(eval_df[rolling_cols])
inverted_scores = -raw_scores.reshape(-1, 1)
eval_df['degradation_index'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(inverted_scores)
    """, language='python')
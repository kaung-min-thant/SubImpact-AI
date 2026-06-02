import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# --- 1. Page Config & State Management ---
st.set_page_config(page_title="SubImpact Tactical Board", layout="wide")
st.title("⚽ SubImpact: AI Substitution Assistant")

# NEW: Initialize Streamlit's "Memory Bank" so the app never forgets the prediction!
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.prediction_result = None

# --- 2. Load the Model, Features, & Scenarios ---
@st.cache_resource
def load_model_data():
    return joblib.load('phase2_best_gradient_boosting_model.pkl')

@st.cache_data
def load_feature_names():
    df = pd.read_csv('phase2_feature_columns.csv')
    return df['feature'].tolist() if 'feature' in df.columns else df.iloc[:, 0].tolist()

@st.cache_data
def load_scenarios():
    with open('scenario_baselines.json', 'r') as f:
        return json.load(f)

try:
    model_dict = load_model_data()
    model = model_dict['model']
    feature_cols = load_feature_names()
    scenario_dict = load_scenarios()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Error loading files: {e}")

# --- 3. Sidebar UI (The 3-Tier Architecture) ---
st.sidebar.header("1️⃣ Match Context")
time_remaining = st.sidebar.slider("Time Remaining (mins)", 1, 45, 20)
score_diff = st.sidebar.slider("Score Difference (us vs them)", -3, 3, -1)
sub_position = st.sidebar.radio("Player Position to Sub In", ["Forward", "Midfielder", "Defender"])

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ Substitution Details")
pass_drop = st.sidebar.slider("Outgoing Player Pass Drop (%)", 0.0, 1.0, 0.15)
action_drop = st.sidebar.slider("Outgoing Player Action Drop (%)", 0.0, 1.0, 0.20)

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ Match Momentum")
momentum = st.sidebar.selectbox("How has the last 15 mins looked?", list(scenario_dict.keys()))

baseline = scenario_dict[momentum]

with st.sidebar.expander("⚙️ Advanced Tactical Tuners (Optional)", expanded=False):
    st.write("These auto-update based on Momentum, but you can override them.")
    team_xg = st.slider("Team xG (Last 15m)", 0.0, 2.0, float(round(baseline['team_xg_prev15'], 2)))
    opp_xg = st.slider("Opponent xG (Last 15m)", 0.0, 2.0, float(round(baseline['opp_xg_prev15'], 2)))
    passes = st.slider("Passes (Last 15m)", 0, 200, int(baseline['passes_prev15']))
    shots = st.slider("Shots (Last 15m)", 0, 15, int(baseline['shots_prev15']))

# --- 4. Main Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🤖 AI Prediction Engine")
    
    position_mapping = {"Defender": 0, "Forward": 1, "Midfielder": 2}
    ai_position_code = position_mapping[sub_position]
    
    # THE BUTTON IS BACK!
    if st.button("Calculate SubImpact Score", type="primary", use_container_width=True):
        if model_loaded:
            # Prepare the data
            input_dict = baseline.copy() 
            input_dict['team_xg_prev15'] = team_xg 
            input_dict['opp_xg_prev15'] = opp_xg
            input_dict['xg_diff_prev15'] = team_xg - opp_xg 
            input_dict['passes_prev15'] = passes
            input_dict['shots_prev15'] = shots
            
            input_dict['time_remaining'] = time_remaining 
            input_dict['score_diff'] = score_diff
            input_dict['pass_success_rate_drop'] = pass_drop
            input_dict['action_rate_drop'] = action_drop
            input_dict['position_group_enc'] = ai_position_code
            
            input_dict['abs_score_diff'] = abs(score_diff)
            input_dict['is_leading'] = 1 if score_diff > 0 else 0
            input_dict['is_trailing'] = 1 if score_diff < 0 else 0

            for col in feature_cols:
                if col not in input_dict:
                    input_dict[col] = 0
            input_data = pd.DataFrame([input_dict])[feature_cols]
            
            # Predict and SAVE to memory!
            st.session_state.prediction_result = model.predict(input_data)[0]
            st.session_state.prediction_run = True
            
        else:
            st.error("Cannot predict. Model is missing.")

    # Only show the results if the button has been pressed at least once
    if st.session_state.prediction_run:
        prediction = st.session_state.prediction_result
        
        st.markdown(f"**Scenario:** You are making a substitution with **{time_remaining} minutes** left.")
        if score_diff < 0 and sub_position == "Forward":
            st.info("🧠 Tactical Intuition: Chasing the game. Attempting to increase Team xG.")
        elif score_diff > 0 and sub_position == "Defender":
            st.info("🧠 Tactical Intuition: Defending the lead. Attempting to decrease Opponent xG.")
        
        st.markdown("---")
        
        if prediction == 2:
            st.success(f"🟢 **POSITIVE IMPACT**\n\nBringing on a {sub_position} here is highly recommended by the AI.")
        elif prediction == 0:
            st.error(f"🔴 **NEGATIVE IMPACT**\n\nWarning. Bringing on a {sub_position} may backfire in this exact game state.")
        else:
            st.warning(f"🟡 **NEUTRAL IMPACT**\n\nA {sub_position} sub here is unlikely to change the momentum.")

with col2:
    st.subheader("📍 Tactical Pitch Map")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0e1117')
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1e1e1e', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    
    if sub_position == "Forward":
        x = np.random.normal(100, 10, 100) 
        y = np.random.normal(40, 15, 100)
        color_map = 'inferno' 
        label_x = 100
    elif sub_position == "Defender":
        x = np.random.normal(20, 10, 100) 
        y = np.random.normal(40, 15, 100)
        color_map = 'mako' 
        label_x = 20
    else: 
        x = np.random.normal(60, 10, 100) 
        y = np.random.normal(40, 15, 100)
        color_map = 'viridis' 
        label_x = 60
        
    pitch.kdeplot(x, y, ax=ax, fill=True, levels=100, thresh=0.1, cmap=color_map, alpha=0.6)
    pitch.scatter(label_x, 40, ax=ax, color='white', edgecolors='black', s=300, marker='*', zorder=3)
    ax.text(label_x, 20, f"Predicted {sub_position} Impact Zone", color='white', ha='center', fontsize=12, fontweight='bold')
    
    st.pyplot(fig)
    plt.close(fig)
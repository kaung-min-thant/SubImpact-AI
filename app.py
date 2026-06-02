import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch


# --- 1. Page Config ---

st.set_page_config(page_title="SubImpact AI", layout="wide")
st.title("⚽ SubImpact AI: Football Substitution Assistant")


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
    default_momentum = list(scenario_dict.keys())[0]
except Exception as e:
    model_loaded = False
    default_momentum = None
    st.error(f"⚠️ Error loading files: {e}")

# --- 3. State Initialization ---
# We must inject the very first JSON values into memory before the page loads
if 'app_init' not in st.session_state and model_loaded:
    st.session_state.prediction_run = False
    st.session_state.active_position = "Forward"
    
    # Load initial baseline values into the tuner memory
    baseline = scenario_dict[default_momentum]
    st.session_state.tuner_team_xg = float(round(baseline['team_xg_prev15'], 2))
    st.session_state.tuner_opp_xg = float(round(baseline['opp_xg_prev15'], 2))
    st.session_state.tuner_passes = int(baseline['passes_prev15'])
    st.session_state.tuner_shots = int(baseline['shots_prev15'])
    
    st.session_state.app_init = True


# --- 4. The Magic Callback ---

# This function runs ONLY when the dropdown changes. It injects new JSON math into the frozen sliders.
def update_tuners():
    selected = st.session_state.momentum_dropdown
    new_base = scenario_dict[selected]
    st.session_state.tuner_team_xg = float(round(new_base['team_xg_prev15'], 2))
    st.session_state.tuner_opp_xg = float(round(new_base['opp_xg_prev15'], 2))
    st.session_state.tuner_passes = int(new_base['passes_prev15'])
    st.session_state.tuner_shots = int(new_base['shots_prev15'])


# --- 5. Sidebar UI ---

st.sidebar.header("Tactical Controls")

st.sidebar.subheader("Match Momentum")
momentum = st.sidebar.selectbox(
    "How has the last 15 mins looked?", 
    list(scenario_dict.keys()), 
    key="momentum_dropdown", 
    on_change=update_tuners 
)

st.sidebar.markdown("---")

with st.sidebar.form(key="tactical_form"):
    
    st.subheader("Match Context")
    time_remaining = st.slider("Time Remaining (mins)", 1, 45, 20)
    score_diff = st.slider("Score Difference", -10, 10, 0)

    st.markdown("---")
    st.subheader("Substitution Details")
    sub_position = st.radio("Player Position to Sub In", ["Forward", "Midfielder", "Defender"])
    pass_drop = st.slider("Outgoing Player Pass Drop (%)", 0.0, 1.0, 0.15)
    action_drop = st.slider("Outgoing Player Action Drop (%)", 0.0, 1.0, 0.20)

    st.markdown("---")
    # These sliders read directly from the injected memory bank (st.session_state.tuner_...)
    with st.expander("⚙️ Advanced Tactical Tuners (Optional)"):
        st.caption("These settings auto-update based on Momentum, but you can override them manually.")
        team_xg = st.slider("Team's xG", 0.0, 2.0, key="tuner_team_xg")
        opp_xg = st.slider("Opponent's xG", 0.0, 2.0, key="tuner_opp_xg")
        passes = st.slider("Passes", 0, 200, key="tuner_passes")
        shots = st.slider("Shots", 0, 15, key="tuner_shots")

    submit_button = st.form_submit_button("**Calculate SubImpact**", use_container_width=True, type="primary")

# --- 6. Main Layout ---

col1, col2 = st.columns([1, 1]) 

with col1:
    st.subheader("🔁 SubImpact Engine")
    
    if submit_button:
        if model_loaded:
            position_mapping = {"Defender": 0, "Forward": 1, "Midfielder": 2}
            
            # Start with the baseline 46 features
            input_dict = scenario_dict[momentum].copy() 
            
            # Apply exactly what is on the UI sliders
            input_dict['team_xg_prev15'] = team_xg 
            input_dict['opp_xg_prev15'] = opp_xg
            input_dict['xg_diff_prev15'] = team_xg - opp_xg 
            input_dict['passes_prev15'] = passes
            input_dict['shots_prev15'] = shots
            
            input_dict['time_remaining'] = time_remaining 
            input_dict['score_diff'] = score_diff
            input_dict['pass_success_rate_drop'] = pass_drop
            input_dict['action_rate_drop'] = action_drop
            input_dict['position_group_enc'] = position_mapping[sub_position]
            
            input_dict['abs_score_diff'] = abs(score_diff)
            input_dict['is_leading'] = 1 if score_diff > 0 else 0
            input_dict['is_trailing'] = 1 if score_diff < 0 else 0

            for col in feature_cols:
                if col not in input_dict:
                    input_dict[col] = 0
            input_data = pd.DataFrame([input_dict])[feature_cols]
            
            # Save final results to memory
            st.session_state.prediction_result = model.predict(input_data)[0]
            st.session_state.prediction_run = True
            st.session_state.active_position = sub_position
            
        else:
            st.error("Cannot predict. Model is missing.")

    # Render Prediction from memory
    if st.session_state.prediction_run:
        prediction = st.session_state.prediction_result
        
        st.markdown(f"**Scenario:** You are substituting in a **{st.session_state.active_position}** with **{time_remaining} minutes** left.")
        if score_diff < 0 and st.session_state.active_position == "Forward":
            st.info("🧠 Tactical Intuition: Attacking the opponent. Attempting to increase Team's xG.")
        elif score_diff > 0 and st.session_state.active_position == "Defender":
            st.info("🧠 Tactical Intuition: Defending the lead. Attempting to decrease Opponent's xG.")
        
        st.markdown("---")
        
        if prediction == 2:
            st.success(f"🟢 **POSITIVE IMPACT**\n\nBringing on a {st.session_state.active_position} here is highly recommended.")
        elif prediction == 0:
            st.error(f"🔴 **NEGATIVE IMPACT**\n\nWarning. Bringing on a {st.session_state.active_position} may backfire in this game state.")
        else:
            st.warning(f"🟡 **NEUTRAL IMPACT**\n\nA {st.session_state.active_position} sub here is unlikely to change the momentum.")

with col2:
    st.subheader("📍 Impact Zone")
    
    @st.cache_resource
    def draw_pitch_map(position):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('none') 
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white', stripe=True)
        pitch.draw(ax=ax)
        
        np.random.seed(42)
        if position == "Forward":
            x = np.random.normal(100, 10, 100) 
            y = np.random.normal(40, 15, 100)
            color_map = 'inferno' 
            label_x = 100
        elif position == "Defender":
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
        
        return fig

    cached_fig = draw_pitch_map(st.session_state.active_position)
    st.pyplot(cached_fig, transparent=True)
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# --- 1. Page Config & State Management ---
st.set_page_config(page_title="SubImpact AI", layout="wide")
st.title("⚽ SubImpact AI: Football Substitution Assistant")

if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.prediction_result = None
    st.session_state.active_position = "Forward"

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

# --- 3. Sidebar UI (THE FROZEN FORM) ---
st.sidebar.header("Tactical Controls")

# Wrapping everything in a form forces Streamlit to DO NOTHING until submitted.
with st.sidebar.form(key="tactical_form"):
    
    st.subheader("1️⃣ Match Context")
    time_remaining = st.slider("Time Remaining (mins)", 1, 45, 20)
    score_diff = st.slider("Score Difference (us vs them)", -3, 3, -1)

    st.markdown("---")
    st.subheader("2️⃣ Substitution Details")
    sub_position = st.radio("Player Position to Sub In", ["Forward", "Midfielder", "Defender"])
    pass_drop = st.slider("Outgoing Player Pass Drop (%)", 0.0, 1.0, 0.15)
    action_drop = st.slider("Outgoing Player Action Drop (%)", 0.0, 1.0, 0.20)

    st.markdown("---")
    st.subheader("3️⃣ Match Momentum")
    momentum = st.selectbox("How has the last 15 mins looked?", list(scenario_dict.keys()))

    with st.expander("⚙️ Advanced Tactical Tuners (Optional)"):
        st.caption("Because the form is frozen, these sliders won't auto-update. Check the box below to override the JSON medians with custom math.")
        force_custom = st.checkbox("Enable Custom Tuners", value=False)
        team_xg = st.slider("Team xG", 0.0, 2.0, 1.0)
        opp_xg = st.slider("Opponent xG", 0.0, 2.0, 1.0)
        passes = st.slider("Passes", 0, 200, 50)
        shots = st.slider("Shots", 0, 15, 5)

    # THE BUTTON MUST BE INSIDE THE FORM!
    submit_button = st.form_submit_button("Calculate SubImpact Score", use_container_width=True, type="primary")


# --- 4. Main Layout ---
# FIX: The 1.2 to 1.0 ratio makes the Prediction UI wider!
col1, col2 = st.columns([1.2, 1.0]) 

with col1:
    st.subheader("🤖 AI Prediction Engine")
    
    if submit_button:
        if model_loaded:
            position_mapping = {"Defender": 0, "Forward": 1, "Midfielder": 2}
            
            # Load the perfect 46 features for the chosen scenario
            input_dict = scenario_dict[momentum].copy() 
            
            # If the user wants to break the rules and force custom numbers:
            if force_custom:
                input_dict['team_xg_prev15'] = team_xg 
                input_dict['opp_xg_prev15'] = opp_xg
                input_dict['xg_diff_prev15'] = team_xg - opp_xg 
                input_dict['passes_prev15'] = passes
                input_dict['shots_prev15'] = shots
            
            # Apply Tier 1 Sliders
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
            
            # Update memory bank ONLY when button is clicked
            st.session_state.prediction_result = model.predict(input_data)[0]
            st.session_state.prediction_run = True
            st.session_state.active_position = sub_position
            
        else:
            st.error("Cannot predict. Model is missing.")

    # Render Prediction
    if st.session_state.prediction_run:
        prediction = st.session_state.prediction_result
        
        st.markdown(f"**Scenario:** You are making a substitution with **{time_remaining} minutes** left.")
        if score_diff < 0 and st.session_state.active_position == "Forward":
            st.info("🧠 Tactical Intuition: Chasing the game. Attempting to increase Team xG.")
        elif score_diff > 0 and st.session_state.active_position == "Defender":
            st.info("🧠 Tactical Intuition: Defending the lead. Attempting to decrease Opponent xG.")
        
        st.markdown("---")
        
        if prediction == 2:
            st.success(f"🟢 **POSITIVE IMPACT**\n\nBringing on a {st.session_state.active_position} here is highly recommended by the AI.")
        elif prediction == 0:
            st.error(f"🔴 **NEGATIVE IMPACT**\n\nWarning. Bringing on a {st.session_state.active_position} may backfire in this exact game state.")
        else:
            st.warning(f"🟡 **NEUTRAL IMPACT**\n\nA {st.session_state.active_position} sub here is unlikely to change the momentum.")

with col2:
    st.subheader("📍 Predicted Impact Zone")
    
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

    # Render pitch map using the LOCKED state from the memory bank
    cached_fig = draw_pitch_map(st.session_state.active_position)
    st.pyplot(cached_fig, transparent=True)
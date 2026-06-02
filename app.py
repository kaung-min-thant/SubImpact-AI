import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# --- 1. Page Configuration ---
st.set_page_config(page_title="SubImpact Tactical Board", layout="wide")
st.title("⚽ SubImpact: AI Substitution Assistant")

# --- 2. Load the Model & Features ---
@st.cache_resource
def load_model_data():
    import joblib
    # Load the dictionary that Minn and Tun created
    return joblib.load('phase2_best_gradient_boosting_model.pkl')

@st.cache_data
def load_feature_names():
    df = pd.read_csv('phase2_feature_columns.csv')
    if 'feature' in df.columns:
        return df['feature'].tolist()
    else:
        return df.iloc[:, 0].tolist()

try:
    # 1. Load the dictionary from the .pkl file
    model_dict = load_model_data()
    
    # 2. Extract ONLY the AI model using the exact key we found
    model = model_dict['model']
    
    # 3. Load the feature columns
    feature_cols = load_feature_names()
    
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Error loading model files. Error: {e}")

# --- 3. Sidebar UI (Clean & Simple) ---
st.sidebar.header("Tactical Situation")
time_remaining = st.sidebar.slider("Time Remaining (mins)", 1, 45, 15)
score_diff = st.sidebar.slider("Score Difference (us vs them)", -3, 3, -1)

st.sidebar.markdown("---")
st.sidebar.header("Outgoing Player Fatigue")
pass_drop = st.sidebar.slider("Pass Success Drop (%)", 0.0, 1.0, 0.15)
action_drop = st.sidebar.slider("Action Rate Drop (%)", 0.0, 1.0, 0.20)

# --- 4. Main Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("AI Prediction Engine")
    
    if st.button("Calculate SubImpact Score", type="primary"):
        if model_loaded:
            # THE FIX: Replace the 0s with realistic League Averages
            # These values simulate a 0-0 game with average possession and pressure
            input_dict = {
                'score_diff': 0, 'is_home': 1, 'time_remaining': 20, 'sub_sequence': 1, 
                'game_phase_enc': 3, 'position_group_enc': 2, 'team_xg_prev15': 0.25, 
                'opp_xg_prev15': 0.25, 'xg_diff_prev15': 0.0, 'shots_prev15': 3, 
                'passes_prev15': 60, 'pressures_prev15': 25, 'competition_enc': 1, 
                'abs_score_diff': 0, 'is_trailing': 0, 'is_leading': 0, 'late_sub': 1, 
                'very_late_sub': 0, 'xg_per_shot_prev15': 0.08, 'pressure_per_pass_prev15': 0.4, 
                'passes_per_min_prev15': 4.0, 'pressures_per_min_prev15': 1.6, 'shots_per_min_prev15': 0.2, 
                'player_passes_prev15': 10, 'pass_success_rate_drop': 0.05, 'player_pressures_prev15': 5, 
                'pressure_activity_drop': 0.1, 'player_actions_prev15': 15, 'action_rate_drop': 0.1, 
                'player_involvement_share_prev15': 0.08, 'FW_passes_prev15': 15, 'FW_pressures_prev15': 8, 
                'FW_shots_prev15': 2, 'FW_xg_prev15': 0.2, 'MF_passes_prev15': 25, 'MF_pressures_prev15': 10, 
                'MF_shots_prev15': 1, 'MF_xg_prev15': 0.05, 'DF_passes_prev15': 20, 'DF_pressures_prev15': 7, 
                'DF_shots_prev15': 0, 'DF_xg_prev15': 0.0, 'GK_passes_prev15': 5, 'GK_pressures_prev15': 0, 
                'GK_shots_prev15': 0, 'GK_xg_prev15': 0.0
            }
            
            # Make sure any missing columns are set to 0 just to be safe
            for col in feature_cols:
                if col not in input_dict:
                    input_dict[col] = 0

            # Overwrite the defaults with OUR slider values
            input_dict['time_remaining'] = time_remaining
            input_dict['score_diff'] = score_diff
            input_dict['pass_success_rate_drop'] = pass_drop
            input_dict['action_rate_drop'] = action_drop
            
            # Recalculate contextual stats based on the slider
            input_dict['abs_score_diff'] = abs(score_diff)
            input_dict['is_leading'] = 1 if score_diff > 0 else 0
            input_dict['is_trailing'] = 1 if score_diff < 0 else 0

            # Convert to DataFrame
            input_data = pd.DataFrame([input_dict])
            
            # Reorder columns to exactly match what the model expects
            input_data = input_data[feature_cols]
            
            # Ask the AI to predict!
            prediction = model.predict(input_data)[0]
            
            # UI LOGIC: Translate the prediction number (0, 1, 2) into text
            if prediction == 2:
                st.success("🟢 AI Prediction: POSITIVE IMPACT")
                st.write("This substitution is highly likely to improve team momentum.")
            elif prediction == 0:
                st.error("🔴 AI Prediction: NEGATIVE IMPACT")
                st.write("Warning: This substitution may disrupt team structure.")
            else:
                st.warning("🟡 AI Prediction: NEUTRAL IMPACT")
                st.write("This substitution is not expected to significantly alter the game flow.")

        else:
            st.error("Cannot predict. Model is missing.")
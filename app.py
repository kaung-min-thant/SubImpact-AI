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
def load_model():
    return joblib.load('phase2_best_gradient_boosting_model.pkl')

@st.cache_data
def load_feature_names():
    # Reading the exact 46 column names the model needs
    df = pd.read_csv('phase2_feature_columns.csv')
    if 'feature' in df.columns:
        return df['feature'].tolist()
    else:
        return df.iloc[:, 0].tolist()

try:
    model = load_model()
    feature_cols = load_feature_names()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Error loading model files. Did you name them correctly? Error: {e}")

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
            # THE MAGIC TRICK: Create a dictionary where all 46 inputs are 0 by default
            input_dict = {col: 0 for col in feature_cols}
            
            # Overwrite only our 4 slider values!
            if 'time_remaining' in input_dict: input_dict['time_remaining'] = time_remaining
            if 'score_diff' in input_dict: input_dict['score_diff'] = score_diff
            if 'pass_success_rate_drop' in input_dict: input_dict['pass_success_rate_drop'] = pass_drop
            if 'action_rate_drop' in input_dict: input_dict['action_rate_drop'] = action_drop
            
            # Convert to DataFrame (The format the AI model demands)
            input_data = pd.DataFrame([input_dict])
            
            # Ask the AI to predict!
            prediction = model.predict(input_data)[0]
            
            # Display the result
            st.metric(label="Predicted Tactical Impact", value=str(prediction))
            st.success("Analysis Complete!")
        else:
            st.error("Cannot predict. Model is missing.")

with col2:
    st.subheader("Tactical Pitch Map")
    fig, ax = plt.subplots(figsize=(6, 4))
    pitch = Pitch(pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    st.pyplot(fig)
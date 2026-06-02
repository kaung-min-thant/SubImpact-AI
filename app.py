import streamlit as st
import pandas as pd
import joblib
from mplsoccer import Pitch
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="SubImpact Tactical Board", layout="wide")
st.title("⚽ SubImpact: AI Substitution Assistant")

# 2. Sidebar for User Inputs
st.sidebar.header("Match Context")
home_team = st.sidebar.selectbox("Home Team", ["Arsenal", "Man City", "Liverpool"])
minute = st.sidebar.slider("Current Minute", 45, 90, 70)
st.sidebar.markdown("---")
st.sidebar.header("Bench Players")
selected_sub = st.sidebar.selectbox("Select Player to Sub In", ["Martinelli", "Trossard", "Nketiah"])

# 3. Load the Model (You will need the .pkl file from your team here)
# @st.cache_resource # This makes sure the model only loads once
# def load_model():
#     return joblib.load('subimpact_model.pkl')
# model = load_model()

# 4. Main Layout (Dividing the screen into two columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted xG Impact")
    st.metric(label=f"Impact Score: {selected_sub}", value="+0.14 xG", delta="High Impact")
    st.write("*(Note: Once the model is linked, this will be a live prediction!)*")

with col2:
    st.subheader("Tactical Pitch Map")
    # Step 4: Drawing the Pitch using mplsoccer
    fig, ax = plt.subplots(figsize=(6, 4))
    pitch = Pitch(pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    
    # We will eventually plot the data here
    
    st.pyplot(fig)
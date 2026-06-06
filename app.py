import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import shap
import base64
import os
import gdown


# --- 1. Page Config ---

st.set_page_config(page_title="SubImpact AI", page_icon="logo.png", layout="wide")


def get_base64_image(image_path):
    """Convert a local image file to a base64 string so HTML can embed it directly."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    logo_base64 = get_base64_image("logo.png")
    st.markdown(
        f"""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <img src="data:image/png;base64,{logo_base64}" width="70" style="margin-right: 15px; border-radius: 12px;">
    <div>
        <h1 style="margin: 0; padding: 0; line-height: 1.1; font-size: clamp(24px, 6vw, 48px); white-space: nowrap;">SubImpact AI</h1>
        <h4 style="margin: 0; padding: 0; font-weight: 400; color: #888; font-size: clamp(14px, 3vw, 18px);">Your Football Substitution Assistant</h4>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )
except Exception:
    # Fallback if logo.png is missing
    st.title("⚽ SubImpact AI: Football Substitution Assistant")


# --- 2. Load the Model, Features, & Scenarios ---

@st.cache_resource
def load_model_data():
    """Download model from Google Drive if not cached locally, then load it."""
    file_id   = '1A-5Ru9HoZN6eWEvUV6Z23SW0dayP8_oS'
    model_path = 'final_best_model.pkl'
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    return joblib.load(model_path)

@st.cache_resource
def load_explainer(_model):
    """Build and cache a SHAP TreeExplainer for the loaded model."""
    return shap.TreeExplainer(_model)

@st.cache_data
def load_feature_names(_model):
    """Return the ordered feature list the model was trained on."""
    return _model.feature_names_in_.tolist()

@st.cache_data
def load_scenarios():
    """Load the pre-built scenario baseline dicts from JSON."""
    with open('scenario_baselines.json', 'r') as f:
        return json.load(f)

# --- Pitch visualisation (cached at module level, not inside a column block) ---

@st.cache_resource
def draw_pitch_map(position):
    """
    Draw a KDE heatmap on a football pitch showing the expected
    zone of influence for the incoming player's position.
    """
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white', stripe=True)
    fig, ax = pitch.draw(figsize=(9, 6))
    fig.patch.set_facecolor('none')

    np.random.seed(42)
    if position == "Forward":
        x         = np.random.normal(100, 10, 100)
        y         = np.random.normal(40, 15, 100)
        color_map = 'inferno'
    elif position == "Defender":
        x         = np.random.normal(20, 10, 100)
        y         = np.random.normal(40, 15, 100)
        color_map = 'mako'
    else:  # Midfielder
        x         = np.random.normal(60, 10, 100)
        y         = np.random.normal(40, 15, 100)
        color_map = 'viridis'

    pitch.kdeplot(x, y, ax=ax, fill=True, levels=100, thresh=0.1, cmap=color_map, alpha=0.6)
    return fig


try:
    model_dict       = load_model_data()
    model            = model_dict['model']
    feature_cols     = load_feature_names(model)   # FIX 1: pass model as arg, not global reference
    scenario_dict    = load_scenarios()
    model_loaded     = True
    default_momentum = list(scenario_dict.keys())[0]
except Exception as e:
    model_loaded     = False
    default_momentum = None
    st.error(f"⚠️ Error loading files: {e}")


# --- 3. State Initialization ---

if 'app_init' not in st.session_state and model_loaded:
    st.session_state.prediction_run    = False
    st.session_state.active_position   = "Forward"

    baseline = scenario_dict[default_momentum]
    st.session_state.tuner_team_xg = float(round(baseline['team_xg_prev15'], 2))
    st.session_state.tuner_opp_xg  = float(round(baseline['opp_xg_prev15'],  2))
    st.session_state.tuner_passes  = int(baseline['passes_prev15'])
    st.session_state.tuner_shots   = int(baseline['shots_prev15'])

    st.session_state.app_init = True


# --- 4. Callback ---

def update_tuners():
    """Fires when the momentum dropdown changes — resets the advanced sliders."""
    selected   = st.session_state.momentum_dropdown
    new_base   = scenario_dict[selected]
    st.session_state.tuner_team_xg = float(round(new_base['team_xg_prev15'], 2))
    st.session_state.tuner_opp_xg  = float(round(new_base['opp_xg_prev15'],  2))
    st.session_state.tuner_passes  = int(new_base['passes_prev15'])
    st.session_state.tuner_shots   = int(new_base['shots_prev15'])


# --- 5. Sidebar UI ---

st.sidebar.header("Match Momentum")
momentum = st.sidebar.selectbox(
    "How has the last 15 mins looked?",
    list(scenario_dict.keys()),
    key="momentum_dropdown",
    on_change=update_tuners,
)

with st.sidebar.form(key="tactical_form"):

    st.subheader("Match Context")
    time_remaining = st.slider("Time Remaining (mins)", 1, 45, 20)
    score_diff     = st.slider("Score Difference", -10, 10, 0)

    st.subheader("Substitution Details")
    sub_position = st.radio("Player Position to Sub In", ["Forward", "Midfielder", "Defender"])
    pass_drop    = st.slider("Outgoing Player Pass Drop (%)", 0.0, 1.0, 0.15)
    action_drop  = st.slider("Outgoing Player Action Drop (%)", 0.0, 1.0, 0.20)

    with st.expander("⚙️ Advanced Settings (Optional)"):
        st.markdown(
            "<span style='font-size: 0.85em; color: gray;'>These settings auto-update based on the "
            "**Match Momentum** you selected, but you can override them manually here.</span>",
            unsafe_allow_html=True,
        )
        team_xg = st.slider("Team's xG (Last 15m)",      0.0, 2.0,  key="tuner_team_xg")
        opp_xg  = st.slider("Opponent's xG (Last 15m)",  0.0, 2.0,  key="tuner_opp_xg")
        passes  = st.slider("Passes (Last 15m)",          0,   200,  key="tuner_passes")
        shots   = st.slider("Shots (Last 15m)",           0,   15,   key="tuner_shots")

    submit_button = st.form_submit_button(
        "**Calculate SubImpact**", use_container_width=True, type="primary"
    )


# --- 6. Main Layout ---

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Engine")

    if submit_button:
        if model_loaded:
            position_mapping = {"Defender": 0, "Forward": 1, "Midfielder": 2}

            input_dict = scenario_dict[momentum].copy()

            input_dict['team_xg_prev15']       = team_xg
            input_dict['opp_xg_prev15']        = opp_xg
            input_dict['xg_diff_prev15']       = team_xg - opp_xg
            input_dict['passes_prev15']        = passes
            input_dict['shots_prev15']         = shots
            input_dict['time_remaining']       = time_remaining
            input_dict['score_diff']           = score_diff
            input_dict['pass_success_rate_drop'] = pass_drop
            input_dict['action_rate_drop']     = action_drop
            input_dict['position_group_enc']   = position_mapping[sub_position]
            input_dict['abs_score_diff']       = abs(score_diff)
            input_dict['is_leading']           = 1 if score_diff > 0 else 0
            input_dict['is_trailing']          = 1 if score_diff < 0 else 0

            for col in feature_cols:
                if col not in input_dict:
                    input_dict[col] = 0
            input_data = pd.DataFrame([input_dict])[feature_cols]

            st.session_state.prediction_result = model.predict(input_data)[0]
            st.session_state.prediction_run    = True
            st.session_state.active_position   = sub_position
            st.session_state.last_input_data   = input_data
        else:
            st.error("Cannot predict. Model is missing.")

    # FIX 2: use .get() so this never crashes if model failed to load
    if st.session_state.get("prediction_run", False):
        prediction = st.session_state.prediction_result

        st.markdown(
            f"**Scenario:** You are substituting in a **{st.session_state.active_position}** "
            f"with **{time_remaining} minutes** left."
        )

        # Tactical intuition — all position/score combinations covered
        active_pos = st.session_state.active_position
        if score_diff < 0 and active_pos == "Forward":
            st.info("**Tactical Intuition:** Attacking the opponent. Attempting to increase Team's xG.")
        elif score_diff > 0 and active_pos == "Defender":
            st.info("**Tactical Intuition:** Defending the lead. Attempting to decrease Opponent's xG.")
        elif score_diff < 0 and active_pos == "Defender":
            st.info("**Tactical Intuition:** Defensive stability while chasing the game.")
        elif score_diff > 0 and active_pos == "Forward":
            st.info("**Tactical Intuition:** Pressing for a second goal to seal the result.")
        elif active_pos == "Midfielder":
            st.info("**Tactical Intuition:** Midfield control — influencing tempo and transitions.")
        else:
            st.info("**Tactical Intuition:** Balanced substitution in this game state.")

        if prediction == 2:
            st.success(f"🟢 **POSITIVE IMPACT**\n\nBringing on a {active_pos} here is highly recommended.")
        elif prediction == 0:
            st.error(f"🔴 **NEGATIVE IMPACT**\n\nBringing on a {active_pos} may backfire in this game state.")
        else:
            st.warning(f"🟡 **NEUTRAL IMPACT**\n\nA {active_pos} sub here is unlikely to change the momentum.")

with col2:
    st.subheader("Impact Zone")
    st.caption("⚠️ **Beta Feature:** This visualization is based purely on the selected position, independent of the ML prediction.")

    cached_fig = draw_pitch_map(st.session_state.get("active_position", "Forward"))
    st.pyplot(cached_fig, transparent=True, use_container_width=True, bbox_inches='tight', pad_inches=0)


# --- 7. xAI (SHAP) Layout ---

# FIX 2 (same): use .get() to guard against uninitialised state
if st.session_state.get("prediction_run", False):
    with st.expander("**Why did SubImpact AI make this decision? (SHAP Explanation)**", expanded=False):
        st.write(
            "This **Waterfall Chart** shows exactly how the specific match momentum and tactical tuners pushed the AI toward its final conclusion."
        )

        explainer   = load_explainer(model)
        shap_values = explainer(st.session_state.last_input_data)
        pred_class  = st.session_state.prediction_result

        # FIX 3: shap.plots.waterfall draws on its own internal figure.
        # Use plt.gcf() AFTER the call to capture that figure, not a manually created one.
        try:
            shap.plots.waterfall(shap_values[0, :, pred_class], show=False)
        except Exception:
            shap.plots.waterfall(shap_values[0], show=False)

        fig = plt.gcf()
        fig.patch.set_facecolor('white')
        st.pyplot(fig, transparent=False)
        plt.close(fig)
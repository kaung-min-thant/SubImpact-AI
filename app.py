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
    st.title("⚽ SubImpact AI: Football Substitution Assistant")


# --- 2. Load the Model, Features, & Scenarios ---

@st.cache_resource
def load_model_data():
    """Download model from Google Drive if not cached locally, then load it.
    Falls back to local backup model if Google Drive is unavailable."""
    
    # --- Plan A: Google Drive ---
    file_id    = '1A-5Ru9HoZN6eWEvUV6Z23SW0dayP8_oS'
    model_path = 'final_best_model.pkl'
    
    if not os.path.exists(model_path):
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.warning(f"⚠️ Could not download primary model from Google Drive: {e}")
    
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"⚠️ Primary model file is corrupted or unreadable: {e}")
            os.remove(model_path)  # Remove bad file so next run retries

    # --- Plan B: Local backup ---
    backup_path = 'phase2_best_gradient_boosting_model.pkl'
    
    if not os.path.exists(backup_path):
        st.error("❌ Backup model not found. Please ensure the file is in the repo.")
        raise FileNotFoundError(f"Backup model not found at: {backup_path}")
    
    st.warning("⚠️ Running on the **backup model** (lower accuracy). Primary model unavailable.")
    return joblib.load(backup_path)

@st.cache_resource
def load_explainer(_model):
    """Build and cache a SHAP TreeExplainer for the loaded model."""
    return shap.TreeExplainer(_model)

@st.cache_data
def load_scenarios():
    """Load the pre-built scenario baseline dicts from JSON."""
    with open('scenario_baselines.json', 'r') as f:
        return json.load(f)


def resolve_model(model_dict):
    raw = model_dict.get('model')

    # Case 2: ensemble dict
    if isinstance(raw, dict) and raw.get('type') == 'soft_voting_ensemble':
        return raw, True

    # Case 3: trained_models entry — one more level deep
    if isinstance(raw, dict) and 'model' in raw:
        return raw['model'], False

    # Case 1: plain estimator
    return raw, False


def get_feature_cols(model_dict, model_obj, is_ensemble):
    # Preferred: explicit key saved in the bundle
    if 'feature_cols' in model_dict and model_dict['feature_cols']:
        return model_dict['feature_cols']
    # Sklearn estimator attribute
    if not is_ensemble and hasattr(model_obj, 'feature_names_in_'):
        return model_obj.feature_names_in_.tolist()
    # trained_models entry has its own 'features' key
    raw = model_dict.get('model')
    if isinstance(raw, dict) and 'features' in raw:
        return raw['features']
    # Ensemble fallback
    if is_ensemble and 'features' in model_dict.get('model', {}):
        return model_dict['model']['features']
    return None

# --- Pitch visualisation (module-level so cache works correctly) ---

@st.cache_data
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
        x, y, color_map = np.random.normal(100, 10, 100), np.random.normal(40, 15, 100), 'inferno'
    elif position == "Defender":
        x, y, color_map = np.random.normal(20,  10, 100), np.random.normal(40, 15, 100), 'mako'
    else:
        x, y, color_map = np.random.normal(60,  10, 100), np.random.normal(40, 15, 100), 'viridis'

    pitch.kdeplot(x, y, ax=ax, fill=True, levels=100, thresh=0.1, cmap=color_map, alpha=0.6)
    return fig


def predict_ensemble(ensemble_dict, input_data):
    """Run soft-voting prediction across all models inside the ensemble dict."""
    sub_models = ensemble_dict.get('models', {})
    probs = []
    for name, info in sub_models.items():
        m        = info['model']
        features = info.get('features', input_data.columns.tolist())
        X        = input_data[features]
        probs.append(m.predict_proba(X))
    avg_prob = np.mean(probs, axis=0)
    return int(np.argmax(avg_prob, axis=1)[0]), avg_prob


# --- Load everything, guard sidebar against undefined variables ---

model_loaded     = False
model_obj        = None
is_ensemble      = False
feature_cols     = []
scenario_dict    = {}
default_momentum = None

try:
    model_dict       = load_model_data()
    model_obj, is_ensemble = resolve_model(model_dict)
    feature_cols     = get_feature_cols(model_dict, model_obj, is_ensemble)
    scenario_dict    = load_scenarios()
    default_momentum = list(scenario_dict.keys())[0]

    if feature_cols is None:
        st.error("⚠️ Could not determine feature columns from the saved model bundle.")
    else:
        model_loaded = True

except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")


# --- 3. State Initialization ---

if 'app_init' not in st.session_state and model_loaded:
    st.session_state.prediction_run  = False
    st.session_state.active_position = "Forward"

    baseline = scenario_dict[default_momentum]
    st.session_state.tuner_team_xg = float(round(baseline['team_xg_prev15'], 2))
    st.session_state.tuner_opp_xg  = float(round(baseline['opp_xg_prev15'],  2))
    st.session_state.tuner_passes  = int(baseline['passes_prev15'])
    st.session_state.tuner_shots   = int(baseline['shots_prev15'])

    st.session_state.app_init = True


# --- 4. Callback ---

def update_tuners():
    """Fires when the momentum dropdown changes — resets the advanced sliders."""
    selected = st.session_state.momentum_dropdown
    new_base = scenario_dict[selected]
    st.session_state.tuner_team_xg = float(round(new_base['team_xg_prev15'], 2))
    st.session_state.tuner_opp_xg  = float(round(new_base['opp_xg_prev15'],  2))
    st.session_state.tuner_passes  = int(new_base['passes_prev15'])
    st.session_state.tuner_shots   = int(new_base['shots_prev15'])


# --- 5. Sidebar UI ---

# Guard: don't render sidebar widgets if scenario_dict is empty
if not model_loaded:
    st.stop()

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
        team_xg = st.slider("Team's xG (Last 15m)",     0.0, 2.0, key="tuner_team_xg")
        opp_xg  = st.slider("Opponent's xG (Last 15m)", 0.0, 2.0, key="tuner_opp_xg")
        passes  = st.slider("Passes (Last 15m)",         0,   200, key="tuner_passes")
        shots   = st.slider("Shots (Last 15m)",          0,   15,  key="tuner_shots")

    submit_button = st.form_submit_button(
        "**Calculate SubImpact**", use_container_width=True, type="primary"
    )


# --- 6. Main Layout ---

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Engine")

    if submit_button:
        position_mapping = {"Defender": 0, "Forward": 1, "Midfielder": 2}

        input_dict = scenario_dict[momentum].copy()

        input_dict['team_xg_prev15']         = team_xg
        input_dict['opp_xg_prev15']          = opp_xg
        input_dict['xg_diff_prev15']         = team_xg - opp_xg
        input_dict['passes_prev15']          = passes
        input_dict['shots_prev15']           = shots
        input_dict['time_remaining']         = time_remaining
        input_dict['score_diff']             = score_diff
        input_dict['pass_success_rate_drop'] = pass_drop
        input_dict['action_rate_drop']       = action_drop
        input_dict['position_group_enc']     = position_mapping[sub_position]
        input_dict['abs_score_diff']         = abs(score_diff)
        input_dict['is_leading']             = 1 if score_diff > 0 else 0
        input_dict['is_trailing']            = 1 if score_diff < 0 else 0

        for col in feature_cols:
            if col not in input_dict:
                input_dict[col] = 0
        input_data = pd.DataFrame([input_dict])[feature_cols]

        if is_ensemble:
            pred, _ = predict_ensemble(model_dict['model'], input_data)
        else:
            pred = int(model_obj.predict(input_data)[0])

        st.session_state.prediction_result = pred
        st.session_state.prediction_run    = True
        st.session_state.active_position   = sub_position
        st.session_state.last_input_data   = input_data

    if st.session_state.get("prediction_run", False):
        prediction = st.session_state.prediction_result
        active_pos = st.session_state.active_position

        st.markdown(
            f"**Scenario:** You are substituting in a **{active_pos}** "
            f"with **{time_remaining} minutes** left."
        )

        # Tactical intuition — all position / score combinations
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

if st.session_state.get("prediction_run", False):
    with st.expander("**Why did SubImpact AI make this decision? (SHAP Explanation)**", expanded=False):
        st.write(
            "This **Waterfall Chart** shows exactly how the specific match momentum and "
            "tactical tuners pushed the AI toward its final conclusion."
        )

        # SHAP only works on a single sklearn estimator, not the ensemble dict.
        # If the best model is the ensemble, use the first sub-model for explanation.
        if is_ensemble:
            sub_models   = model_dict['model'].get('models', {})
            explain_name = list(sub_models.keys())[0]
            explain_model   = sub_models[explain_name]['model']
            explain_features = sub_models[explain_name].get('features', feature_cols)
            explain_data  = st.session_state.last_input_data[explain_features]
            st.caption(f"ℹ️ SHAP explanation uses **{explain_name}** (first model in the ensemble).")
        else:
            explain_model    = model_obj
            explain_data     = st.session_state.last_input_data

        explainer   = load_explainer(explain_model)
        shap_values = explainer(explain_data)
        pred_class  = st.session_state.prediction_result

        try:
            shap.plots.waterfall(shap_values[0, :, pred_class], show=False)
        except (IndexError, ValueError, TypeError):
            try:
            # Shape is (samples, features) — no class dimension
                shap.plots.waterfall(shap_values[0], show=False)
            except (IndexError, ValueError, TypeError):
                st.warning("⚠️ SHAP explanation could not be displayed for this model.")
                st.code(f"shap_values shape: {shap_values.shape}\nExpected value: {explainer.expected_value}")

        fig = plt.gcf()
        fig.patch.set_facecolor('white')
        st.pyplot(fig, transparent=False)
        plt.close(fig)
import pandas as pd
import json

print("Loading dataset...")

# 1. Load the full dataset instead of the columns file
df = pd.read_csv('substitutions_phase2_features.csv')

# 2. Extract just the feature columns (drop the target label and identifiers)
cols_to_drop = ['match_id', 'substitution_id', 'team', 'player_out', 'player_in', 'impact_score', 'impact_label']
feature_cols = [col for col in df.columns if col not in cols_to_drop]

# 3. Create a filtered DataFrame using only the feature columns
features_df = df[feature_cols].copy()

# Get the exact 46 features the model needs
features = features_df.values.flatten().tolist() if 'feature' in features_df.columns else features_df.columns.tolist()
num_df = df.select_dtypes(include=['number'])
valid_features = [f for f in features if f in num_df.columns]

# Define the exact same tactical scenarios
cond_dominating = df['xg_diff_prev15'] > 0.5
cond_siege = df['xg_diff_prev15'] < -0.5
cond_chaos = (df['team_xg_prev15'] > 0.4) & (df['opp_xg_prev15'] > 0.4)
cond_sterile = (df['passes_prev15'] > 100) & (df['team_xg_prev15'] < 0.1)
cond_neutral = pd.Series(True, index=df.index) # The baseline average for the whole dataset

# Function to grab the median of ALL 46 features for a specific condition
def get_all_medians(condition):
    # Calculate medians, fill any missing data with 0, and turn it into a dictionary
    return num_df[condition][valid_features].median().fillna(0).to_dict()

# Map the UI dropdown names to the exact calculated dictionaries
scenarios = {
    "Neutral / Midfield Battle": get_all_medians(cond_neutral),
    "Sustained Pressure (We are dominating)": get_all_medians(cond_dominating),
    "Under Siege (Opponent is dominating)": get_all_medians(cond_siege),
    "End-to-End Chaos (Open game, high xG)": get_all_medians(cond_chaos),
    "Sterile Possession (Lots of passes, no shots)": get_all_medians(cond_sterile)
}

# Save it all to a clean JSON file
with open("scenario_baselines.json", "w") as f:
    json.dump(scenarios, f, indent=4)

print("✅ SUCCESS: Extracted all 46 features and saved to 'scenario_baselines.json'")
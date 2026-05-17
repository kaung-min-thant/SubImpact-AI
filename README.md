# ⚽ SubImpact AI: Tactical Substitution Monitor

> This repository contains the team project for the **AI Programming** course at **Kyung Hee University**.

## 📖 About The Project
**SubImpact AI** is a machine learning model that predicts and explains whether a football substitution has a positive, neutral, or negative tactical impact based on match-state and player event data. 

In modern football, substitutions are critical tactical decisions, but evaluating their actual impact is difficult. A goal after a substitution doesn't automatically prove the substitution caused the improvement. This project moves beyond basic intuition, using **StatsBomb Open Data** to analyze match flow, substitution timing, pre/post Expected Goals (xG), and player fatigue proxies to provide data-driven, explainable insights into coaching decisions.

## ✨ Key Features
* **Event Data Processing:** Direct integration with the `statsbombpy` API to extract and format historical match logs.
* **Complex Feature Engineering:** Calculates match momentum and outgoing player fatigue proxies (e.g., dropping pass accuracy, pressure declines).
* **Predictive Modeling:** Utilizes algorithms like Random Forest, XGBoost, and LightGBM to classify substitution impact.
* **Explainable AI (XAI):** Implements SHAP (SHapley Additive exPlanations) to visually explain *why* the model made a specific prediction, preventing a "black box" scenario.
* **Interactive Dashboard:** A Streamlit-powered Web UI acting as a "Manager's Tablet" to visualize pitch maps and prediction rationales.

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, statsbombpy
* **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest), XGBoost, LightGBM
* **Explainability & Visualization:** SHAP, Matplotlib, mplsoccer
* **Frontend UI:** Streamlit

## 👥 Team Members & Roles (Team 6)

| Name | Primary Role | Core Responsibilities |
| :--- | :--- | :--- |
| **Minn Thwin Khant** | Data Pipeline & Baselines | StatsBomb API extraction, dataset formatting, and Baseline Modeling (Logistic Regression/Random Forest). |
| **Tun Zaw Lin** | Feature Engineering & ML | Engineering fatigue proxies, positional event windowing, and advanced modeling (XGBoost/LightGBM). |
| **Kaung Min Thant** | Evaluation, XAI & Web UI | Target variable calculation, Evaluation Metrics, SHAP implementation, and Streamlit UI development. |

## 🚀 How to Run (Local Development)
1. Clone the repository:
   ```bash
   git clone https://github.com/kaung-min-thant/SubImpact-AI.git
   cd SubImpact-AI
2. Install the required dependencies:
   ```bash
   pip install pandas statsbombpy scikit-learn xgboost shap streamlit mplsoccer
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py

# ⚽ SubImpact AI: Your Football Substitution Assistant

> This repository contains the final team project for the **AI Programming** course at **Kyung Hee University**. 
> 
> 🔴 **Live Web Dashboard:** [https://subimpact-ai.streamlit.app/](https://subimpact-ai.streamlit.app/)

## 📖 About The Project
**SubImpact AI** is a machine learning model that predicts and explains whether a football substitution has a positive, neutral, or negative tactical impact based on match-state and player event data. 

In modern football, substitutions are critical tactical decisions, but evaluating their actual impact is difficult because a goal after a substitution doesn't automatically prove causality. This project moves beyond basic intuition, using **StatsBomb Open Data** to analyze match momentum, substitution timing, pre/post Expected Goals (xG), and player fatigue proxies to provide a data-driven, explainable decision-support tool for coaching staffs. 

## ✨ Key Features & Final Results
* **Event Data Pipeline:** Extracted and processed 5,207 independent substitution samples from ~1,900 matches across 6 major competitions using the `statsbombpy` API.
* **4-Layer Feature Engineering:** Designed 40+ granular features categorized into Match Context, Momentum, Fatigue Proxies (e.g., dropping pass accuracy, pressure declines), and Position-Specific metrics.
* **Advanced Predictive Engine:** Executed a model tournament, with the **ExtraTrees Classifier Ensemble** emerging as the global winner, achieving a **68.04% Accuracy (0.8431 Macro AUC)** on highly noisy football event data.
* **Explainable AI (XAI):** Implemented heavily cached SHAP (SHapley Additive exPlanations) Waterfall plots to visually explain *why* the model made a specific prediction in real-time.
* **Zero-Latency Cloud Dashboard:** Deployed a Streamlit-powered "Manager's Tablet" featuring an interactive UI, state-managed momentum scenarios, and a dynamic Google Drive (`gdown`) architecture to bypass deployment limits for the 271MB model file. Includes a Kernel Density Estimation (KDE) Impact Zone pitch map using `mplsoccer`.

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy, statsbombpy
* **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest, ExtraTrees), XGBoost, LightGBM
* **Explainability & Visualization:** SHAP, Matplotlib, mplsoccer
* **Frontend & Cloud Deployment:** Streamlit, gdown, joblib

## 👥 Team Members & Roles (Team 6)

| Name | Primary Role | Core Responsibilities |
| :--- | :--- | :--- |
| **Minn Thwin Khant** | Data Extraction & Baselines | `statsbombpy` API event data parsing, 15-minute window preprocessing, and initial Baseline Modeling (Logistic Regression/Random Forest). |
| **Tun Zaw Lin** | Feature Engineering & ML Training | Designing the 4-layer features (Fatigue, Momentum, etc.), and training initial XGBoost/LightGBM and ExtraTrees ensemble comparisons. |
| **Kaung Min Thant** | Project Lead, ML Refactoring & Full-Stack Deployment | **Project Management:** Established sprint timelines and finalized the presentation narrative. <br>**Pipeline Refactoring:** Debugged and refactored scattered Jupyter notebooks into a single, optimized ML pipeline. <br>**XAI & Web UI:** Built the Streamlit dashboard, integrated SHAP visualizations, and managed the cloud deployment architecture. |

## 🚀 How to Run (Local Development)
1. Clone the repository:
   ```bash
   git clone https://github.com/kaung-min-thant/SubImpact-AI.git
   cd SubImpact-AI
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

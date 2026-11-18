# IPL Player Performance Prediction Web Application

A machine learning web application analyzing IPL player performances across 16 seasons (2008–2023) using a trained **Random Forest Regressor**, featuring an IPL dark stadium interface, interactive Chart.js visualizations, and precomputed prediction analytics.

---

## 🏏 Project Overview & Operational Mode

* **Model Used:** Precomputed test evaluation outputs from the trained **Random Forest Regressor**.
* **Integrity & Policy (Option C):** No models were retrained, modified, or overwritten. The application reuses the authentic outputs and data artifacts (`IPL_Random_Forest_Predictions.csv`, `IPL_Player_Match_Performance.csv`, `IPL_Feature_Importance.csv`, `IPL_Model_Evaluation_Metrics.csv`, and `IPL_Model_Comparison.csv`).
* **Transparency Notice:** Because the binary serialized model file (`.pkl` / `.joblib`) was not included in the repository, hypothetical real-time input inference is explicitly disabled to avoid generating unverified predictions. Only the genuine, verified model evaluation results on the 2021–2023 test set are displayed.

---

## 📊 Dataset & Model Specifications

1. **Cleaned Ball-by-Ball Data (`IPL_Final_Project_Cleaned_Data.csv`):** 201,115 ball records across 848 matches.
2. **Player Match Performance (`IPL_Player_Match_Performance.csv`):** 21,716 player-match entries covering 695 unique players across 2008–2023.
3. **Random Forest Predictions (`IPL_Random_Forest_Predictions.csv`):** 4,085 prediction rows covering 261 players across 2021–2023.
4. **Target Metric:** $\text{performance\_score} = \text{batting\_runs} + (20 \times \text{wickets})$.
5. **Model Evaluation Metrics:**
   * **$R^2$ Score:** `0.00274062`
   * **MAE (Mean Absolute Error):** `18.63698904`
   * **RMSE (Root Mean Squared Error):** `23.42119164`

---

## 🌐 Deployment on Render (Step-by-Step)

The project is fully prepared for instant deployment on **Render**:

### Method 1: Web Service Setup (Manual)
1. Push this repository to **GitHub**.
2. Go to your [Render Dashboard](https://dashboard.render.com/) and click **New + > Web Service**.
3. Connect your GitHub repository.
4. Configure the service settings:
   * **Name:** `ipl-player-performance-prediction`
   * **Region:** Any (e.g., Oregon / Frankfurt / Singapore)
   * **Branch:** `main` (or your default branch)
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app`
   * **Plan:** Free
5. Click **Create Web Service**.

### Method 2: Blueprint Deployment (Automatic via `render.yaml`)
1. Push this repository to GitHub.
2. In Render, select **New + > Blueprint**.
3. Select your repository. Render will automatically read `render.yaml`, configure the build command, start command, and environment variables.

---

## 🚀 Running Locally

```powershell
# Install requirements
python -m pip install -r requirements.txt

# Run the Flask app
python app.py
```

The application will bind to `0.0.0.0` and be accessible at **`http://localhost:5000`** (or `$PORT` if set).

---

## 📁 Project Directory Layout

```
IPL PLAYER PERFORMANCE PREDICTION USING ML/
│
├── IPL_Feature_Importance.csv         # 12 features & relative importance weights
├── IPL_Final_Project_Cleaned_Data.csv # Ball-by-ball IPL dataset
├── IPL_Model_Comparison.csv           # Model benchmark comparison metrics
├── IPL_Model_Evaluation_Metrics.csv   # R2, MAE, RMSE metrics
├── IPL_Player_Match_Performance.csv   # Aggregated player performance & lag features
├── IPL_Random_Forest_Predictions.csv  # 4,085 test predictions (2021-2023)
│
├── app.py                             # Flask entry point (binds to 0.0.0.0, PORT)
├── data_service.py                    # In-memory cached dataset & querying service
├── requirements.txt                   # Dependencies (flask, pandas, numpy, gunicorn)
├── Procfile                           # Gunicorn start process definition
├── render.yaml                        # Render Blueprint configuration
├── .python-version                    # Python runtime specification (3.11.9)
├── .gitignore                         # Git ignore configuration
├── README.md                          # Documentation
│
├── static/
│   ├── css/
│   │   └── style.css                  # Custom Dark IPL Stadium theme styling
│   └── js/
│       └── main.js                    # Chart.js visualization logic & interactive filters
│
└── templates/
    ├── base.html                      # Base layout with navbar, banner, and footer
    ├── index.html                     # Main dashboard & high-level statistics
    ├── predictions.html               # Random Forest predictions & 12-feature inspector
    ├── players.html                   # All 695 player career catalog & search
    ├── player_detail.html             # Individual player career profile & breakdown
    ├── compare.html                   # Head-to-head player comparison tool
    ├── model_analysis.html            # Feature importances & residual error analytics
    └── 404.html                       # 404 error template
```

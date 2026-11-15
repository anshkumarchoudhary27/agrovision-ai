# AgriVision AI — Beginner project

## Use only Google Colab, GitHub and Streamlit

### Step 1 — Train the model in Colab
1. Open `Colab_Train_Model.ipynb` in Google Colab.
2. Run every cell from top to bottom.
3. The notebook downloads the dataset, trains a Random Forest regression pipeline, evaluates it and downloads `yield_model.joblib` and `metadata.json`.

### Step 2 — Create a GitHub repository
Create a public repository named `agrivision-ai`. Upload these four files to the repository root:
- `app.py`
- `requirements.txt`
- `yield_model.joblib` (downloaded from Colab)
- `metadata.json` (downloaded from Colab)

### Step 3 — Deploy with Streamlit
1. Open https://share.streamlit.io and sign in using GitHub.
2. Click **Create app**.
3. Select your repository, branch `main`, and main file path `app.py`.
4. Click **Deploy**.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Dataset
The notebook uses `yield_df.csv` from https://github.com/kdeeey/cropai .
Fields used: country (`Area`), crop (`Item`), year, average annual rainfall, pesticide use, average temperature and yield (`hg/ha_yield`).

The dataset contains historical country-level observations (1990–2013), not farm-level measurements. The ML prediction is an educational estimate, not a validated local farm forecast. The calculator is arithmetic only.

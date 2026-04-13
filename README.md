# Bana7075_MVP
Machine learning model for Bana 7075 Machine Learning course

Machine learning model trained to predict demand for rental bikes. Based off of dataset pulled from [Kagle Bike Rental](https://www.kaggle.com/competitions/bike-sharing-demand/data)

# Setup
Utilizing Python version 3.13

## Install Dependancies
Project dependancies are listed in the requirements.txt file. Install dependancies from file:
`pip install -r requirements.txt`

# Development Notes
## Update Dependancies
When new dependancies are added to the project, make sure to add them to the requirements file:
`pip freeze > requirements.txt`

# Model Experiments
MLFlow is the primary tool to manage experiment tracking and model versioning

Utilize the MLFlow ui by running the following command in terminal:
`mlflow ui`

Then navigate to the locally hosted webpage
(http://127.0.0.1:5000)

# Usage
Run full pipeline
`python main.py`

Run data quality tests
`python -m tests.data_quality.test_data_quality`

# DevOps Pipeline

## Infrastructure Overview
```
Your machine
  → push to production
        │
        ├── GitHub Actions runner (Ubuntu VM, free)
        │     ├── run data quality tests
        │     ├── train model (smoke test)
        │     └── deploy docs/ → GitHub Pages
        │
        └── Render server (Linux, free tier)
              ├── train model (for real)
              └── run FastAPI server ← web UI calls this
```

## Branches
All production-ready code lives on the `production` branch. The CI/CD pipeline triggers automatically on every push to `production`.

## GitHub Actions (CI/CD)
On every push to `production`, the pipeline runs two jobs:

**validate** — runs first and must pass before anything deploys:
1. Installs all dependencies from `requirements.txt`
2. Runs data quality tests against known bad input files
3. Runs the full model training pipeline end-to-end as a smoke test

**deploy-pages** — runs after `validate` passes:
1. Deploys the `docs/` folder to GitHub Pages (the web UI)

The pipeline can also be triggered manually from the Actions tab on GitHub.

## API Hosting (Render)
The FastAPI prediction API is hosted on [Render](https://render.com) at:

**`https://bana7075-mvp.onrender.com`**

Render watches the `production` branch. On every new commit it:
1. Installs dependencies
2. Retrains the LightGBM model from `train.csv`
3. Starts the FastAPI server

Endpoints:
- `GET /health` — confirms API is live and model is loaded
- `POST /predict` — accepts bike rental conditions, returns predicted hourly count

## Web UI (GitHub Pages)
A simple prediction form is hosted on GitHub Pages at:

**`https://jamesallen74.github.io/Bana7075_MVP`**

The UI is a single HTML page (`docs/index.html`) that calls the Render API. No build tools or frameworks — plain HTML, CSS, and JavaScript.

## Retraining with New Data
To retrain the model with new data:
1. Replace `train.csv` with the updated dataset
2. Commit and push to `production`
3. GitHub Actions validates the pipeline
4. Render automatically retrains and redeploys the API

# Contributors
- James Allen
- Lee Brodbeck-Moore
- Rachael Rahe
- Brett Toothman
- Sagar Vedantam
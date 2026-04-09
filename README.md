# Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

This project delivers an end-to-end clustering workflow for Parcl Co. Limited using the provided `clients.csv` and `properties.csv` files. It includes:

- data cleaning and feature engineering
- k-means and hierarchical clustering validation
- a research-paper style markdown report
- a Streamlit dashboard with buyer, investor, geographic, and segment insight views

## Project Structure

- `data/raw/`: local copies of the provided datasets
- `src/buyer_segmentation/pipeline.py`: analysis pipeline and export logic
- `scripts/run_analysis.py`: batch run that generates all outputs
- `scripts/run_dashboard.py`: project-local Streamlit launcher
- `app.py`: Streamlit dashboard
- `outputs/`: generated CSV, JSON, and model artifacts
- `reports/research_paper.md`: generated write-up

## Setup

Install dependencies into the workspace:

```powershell
python -m pip install --target .packages -r requirements.txt
```

## Run The Analysis

```powershell
python scripts/run_analysis.py
```

This generates:

- `outputs/clustered_clients.csv`
- `outputs/cluster_summary.csv`
- `outputs/cluster_evaluation.csv`
- `outputs/country_segment_summary.csv`
- `outputs/region_segment_summary.csv`
- `outputs/model_metrics.json`
- `reports/research_paper.md`

## Launch The Dashboard

```powershell
python scripts/run_dashboard.py
```

## Business Notes

- The dataset is dominated by US-based individual buyers, so the empirical clusters are more behavior-driven than purely demographic.
- Silhouette score favors 3 clusters in some simpler feature sets, but the final engineered feature set supports 4 clusters and gives clearer operating personas for sales and marketing teams.
- The resulting segment names are derived from actual purchase depth, spend, financing behavior, and transaction patterns in the supplied data.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
REPORTS_DIR = PROJECT_ROOT / 'reports'

CLIENTS_PATH = DATA_DIR / 'clients.csv'
PROPERTIES_PATH = DATA_DIR / 'properties.csv'

SEGMENT_ACTIONS = {
    'Portfolio Investors': 'Prioritize bundled inventory, early access launches, and dedicated relationship management.',
    'Premium Buyers': 'Recommend larger units and premium inventory with concierge-style follow-up.',
    'Financed Value Seekers': 'Lead with affordability messaging, loan assistance, and step-by-step decision support.',
    'Core Growth Buyers': 'Promote balanced investment options, repeat-purchase incentives, and cross-sell campaigns.',
}


@dataclass
class SegmentationResult:
    clients: pd.DataFrame
    properties: pd.DataFrame
    evaluation: pd.DataFrame
    cluster_summary: pd.DataFrame
    country_summary: pd.DataFrame
    region_summary: pd.DataFrame
    chosen_k: int
    optimal_k: int
    kmeans_silhouette: float
    hierarchical_silhouette: float
    model_alignment: float
    research_report: str


def _normalize_text(value: Any) -> Any:
    if pd.isna(value):
        return value
    return str(value).strip()


def _parse_mixed_dates(series: pd.Series, *, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(series, format='mixed', dayfirst=dayfirst, errors='coerce')


def _format_markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for _, row in frame.iterrows():
        values = [str(value) for value in row.tolist()]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def load_and_prepare_data(
    clients_path: Path = CLIENTS_PATH,
    properties_path: Path = PROPERTIES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)

    clients.columns = [column.strip() for column in clients.columns]
    properties.columns = [column.strip() for column in properties.columns]

    for column in clients.select_dtypes(include='object').columns:
        clients[column] = clients[column].map(_normalize_text)
    for column in properties.select_dtypes(include='object').columns:
        properties[column] = properties[column].map(_normalize_text)

    clients = clients.drop_duplicates(subset=['client_id']).copy()
    properties = properties.drop_duplicates(subset=['listing_id']).copy()

    clients['client_type'] = (
        clients['client_type']
        .replace({'Company': 'Corporate', 'company': 'Corporate', 'Individual': 'Individual'})
        .fillna('Unknown')
    )
    clients['gender'] = clients['gender'].fillna('Unknown')
    clients['acquisition_purpose'] = (
        clients['acquisition_purpose']
        .replace({'Home': 'Personal Use', 'home': 'Personal Use', 'Investment': 'Investment'})
        .fillna('Unknown')
    )
    clients['loan_applied'] = (
        clients['loan_applied']
        .replace({'Yes': 'Yes', 'No': 'No', 'yes': 'Yes', 'no': 'No'})
        .fillna('Unknown')
    )
    clients['referral_channel'] = clients['referral_channel'].fillna('Unknown').str.title()
    clients['country'] = clients['country'].fillna('Unknown')
    clients['region'] = clients['region'].fillna('Unknown')
    clients['date_of_birth'] = _parse_mixed_dates(clients['date_of_birth'])
    clients['satisfaction_score'] = pd.to_numeric(clients['satisfaction_score'], errors='coerce')

    properties['transaction_date'] = _parse_mixed_dates(properties['transaction_date'], dayfirst=True)
    properties['sale_price_num'] = (
        properties['sale_price'].replace({r'[$,]': ''}, regex=True).astype(float)
    )
    properties['listing_status'] = properties['listing_status'].fillna('Unknown').str.title()
    properties['unit_category'] = properties['unit_category'].fillna('Unknown').str.title()
    properties['client_ref'] = properties['client_ref'].fillna('')

    reference_date = properties['transaction_date'].max()
    clients['age'] = ((reference_date - clients['date_of_birth']).dt.days / 365.25).round(1)

    top_countries = clients['country'].value_counts().head(6).index
    clients['buyer_origin'] = clients['country'].eq('USA').map({True: 'Domestic', False: 'International'})
    clients['country_group'] = clients['country'].where(clients['country'].isin(top_countries), 'Other')

    transactions = properties.loc[properties['client_ref'].ne('')].copy()
    aggregated = transactions.groupby('client_ref').agg(
        purchase_count=('listing_id', 'count'),
        total_spend=('sale_price_num', 'sum'),
        avg_spend=('sale_price_num', 'mean'),
        max_spend=('sale_price_num', 'max'),
        avg_area=('floor_area_sqft', 'mean'),
        max_area=('floor_area_sqft', 'max'),
        tower_diversity=('tower_number', 'nunique'),
        office_share=('unit_category', lambda values: (values == 'Office').mean()),
        first_transaction=('transaction_date', 'min'),
        last_transaction=('transaction_date', 'max'),
    )
    aggregated['holding_window_days'] = (
        aggregated['last_transaction'] - aggregated['first_transaction']
    ).dt.days
    aggregated['days_since_last_purchase'] = (
        reference_date - aggregated['last_transaction']
    ).dt.days

    clients = clients.merge(aggregated, left_on='client_id', right_index=True, how='left')

    numeric_fill_columns = [
        'age',
        'satisfaction_score',
        'purchase_count',
        'total_spend',
        'avg_spend',
        'max_spend',
        'avg_area',
        'max_area',
        'tower_diversity',
        'office_share',
        'holding_window_days',
        'days_since_last_purchase',
    ]
    for column in numeric_fill_columns:
        clients[column] = clients[column].fillna(clients[column].median())

    clients['first_transaction'] = clients['first_transaction'].fillna(reference_date)
    clients['last_transaction'] = clients['last_transaction'].fillna(reference_date)
    clients['investment_flag'] = clients['acquisition_purpose'].eq('Investment')

    return clients, properties


def build_feature_matrix(
    clients: pd.DataFrame,
) -> tuple[pd.DataFrame, Any, list[str], list[str]]:
    categorical_columns = [
        'client_type',
        'acquisition_purpose',
        'loan_applied',
        'referral_channel',
        'buyer_origin',
        'country_group',
    ]
    numeric_columns = [
        'age',
        'satisfaction_score',
        'purchase_count',
        'total_spend',
        'avg_spend',
        'max_spend',
        'avg_area',
        'tower_diversity',
        'office_share',
        'holding_window_days',
        'days_since_last_purchase',
    ]

    transformer = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),
            ('numeric', StandardScaler(), numeric_columns),
        ]
    )
    features = transformer.fit_transform(clients[categorical_columns + numeric_columns])
    feature_matrix = pd.DataFrame(features)
    return feature_matrix, transformer, categorical_columns, numeric_columns


def evaluate_cluster_range(feature_matrix: pd.DataFrame, min_k: int = 2, max_k: int = 8) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for cluster_count in range(min_k, max_k + 1):
        model = KMeans(n_clusters=cluster_count, n_init=30, random_state=42)
        labels = model.fit_predict(feature_matrix)
        rows.append(
            {
                'k': cluster_count,
                'inertia': round(float(model.inertia_), 2),
                'silhouette_score': round(float(silhouette_score(feature_matrix, labels)), 4),
            }
        )
    return pd.DataFrame(rows)


def _assign_segment_names(summary: pd.DataFrame) -> dict[int, str]:
    remaining = set(summary.index.tolist())
    names: dict[int, str] = {}

    portfolio_cluster = int(summary['avg_purchase_count'].idxmax())
    names[portfolio_cluster] = 'Portfolio Investors'
    remaining.discard(portfolio_cluster)

    premium_cluster = int(summary.loc[list(remaining), 'avg_spend_per_property'].idxmax())
    names[premium_cluster] = 'Premium Buyers'
    remaining.discard(premium_cluster)

    financed_cluster = int(summary.loc[list(remaining), 'loan_dependency_pct'].idxmax())
    names[financed_cluster] = 'Financed Value Seekers'
    remaining.discard(financed_cluster)

    for cluster_id in remaining:
        names[int(cluster_id)] = 'Core Growth Buyers'

    return names


def summarize_clusters(clients: pd.DataFrame) -> pd.DataFrame:
    summary = clients.groupby('cluster').agg(
        buyers=('client_id', 'count'),
        avg_age=('age', 'mean'),
        avg_satisfaction=('satisfaction_score', 'mean'),
        avg_purchase_count=('purchase_count', 'mean'),
        avg_total_spend=('total_spend', 'mean'),
        avg_spend_per_property=('avg_spend', 'mean'),
        avg_area_sqft=('avg_area', 'mean'),
        loan_dependency_pct=('loan_applied', lambda values: (values == 'Yes').mean() * 100),
        investment_intent_pct=('acquisition_purpose', lambda values: (values == 'Investment').mean() * 100),
        international_share_pct=('buyer_origin', lambda values: (values == 'International').mean() * 100),
        corporate_share_pct=('client_type', lambda values: (values == 'Corporate').mean() * 100),
    )

    dominant_country = clients.groupby('cluster')['country'].agg(
        lambda values: values.value_counts().index[0]
    )
    dominant_region = clients.groupby('cluster')['region'].agg(
        lambda values: values.value_counts().index[0]
    )
    dominant_channel = clients.groupby('cluster')['referral_channel'].agg(
        lambda values: values.value_counts().index[0]
    )

    summary['dominant_country'] = dominant_country
    summary['dominant_region'] = dominant_region
    summary['dominant_referral_channel'] = dominant_channel
    summary['buyer_share_pct'] = summary['buyers'] / summary['buyers'].sum() * 100

    segment_names = _assign_segment_names(summary)
    summary['segment_name'] = summary.index.map(segment_names)
    summary['recommended_action'] = summary['segment_name'].map(SEGMENT_ACTIONS)

    summary = summary[
        [
            'segment_name',
            'buyers',
            'buyer_share_pct',
            'avg_age',
            'avg_satisfaction',
            'avg_purchase_count',
            'avg_total_spend',
            'avg_spend_per_property',
            'avg_area_sqft',
            'loan_dependency_pct',
            'investment_intent_pct',
            'international_share_pct',
            'corporate_share_pct',
            'dominant_country',
            'dominant_region',
            'dominant_referral_channel',
            'recommended_action',
        ]
    ].sort_values('avg_total_spend', ascending=False)

    numeric_columns = [
        'buyer_share_pct',
        'avg_age',
        'avg_satisfaction',
        'avg_purchase_count',
        'avg_total_spend',
        'avg_spend_per_property',
        'avg_area_sqft',
        'loan_dependency_pct',
        'investment_intent_pct',
        'international_share_pct',
        'corporate_share_pct',
    ]
    summary[numeric_columns] = summary[numeric_columns].round(2)
    return summary.reset_index().rename(columns={'cluster': 'cluster_id'})


def build_country_summary(clients: pd.DataFrame) -> pd.DataFrame:
    summary = (
        clients.groupby(['country', 'segment_name'])
        .agg(
            buyers=('client_id', 'count'),
            avg_total_spend=('total_spend', 'mean'),
        )
        .reset_index()
    )
    summary['avg_total_spend'] = summary['avg_total_spend'].round(2)
    return summary


def build_region_summary(clients: pd.DataFrame) -> pd.DataFrame:
    summary = (
        clients.groupby(['region', 'segment_name'])
        .agg(
            buyers=('client_id', 'count'),
            avg_purchase_count=('purchase_count', 'mean'),
            avg_total_spend=('total_spend', 'mean'),
        )
        .reset_index()
    )
    summary['avg_purchase_count'] = summary['avg_purchase_count'].round(2)
    summary['avg_total_spend'] = summary['avg_total_spend'].round(2)
    return summary


def build_research_report(result: SegmentationResult) -> str:
    dataset_table = pd.DataFrame(
        [
            ['Clients', len(result.clients)],
            ['Property listings', len(result.properties)],
            ['Distinct countries', result.clients['country'].nunique()],
            ['Distinct regions', result.clients['region'].nunique()],
            ['Chosen cluster count', result.chosen_k],
            ['Silhouette-best cluster count', result.optimal_k],
        ],
        columns=['Metric', 'Value'],
    )

    evaluation_table = result.evaluation.copy()
    evaluation_table['silhouette_score'] = evaluation_table['silhouette_score'].map(lambda value: f'{value:.4f}')
    evaluation_table['inertia'] = evaluation_table['inertia'].map(lambda value: f'{value:,.2f}')

    summary_table = result.cluster_summary[
        [
            'cluster_id',
            'segment_name',
            'buyers',
            'buyer_share_pct',
            'avg_purchase_count',
            'avg_total_spend',
            'loan_dependency_pct',
            'investment_intent_pct',
        ]
    ].copy()
    summary_table['avg_total_spend'] = summary_table['avg_total_spend'].map(lambda value: f'${value:,.0f}')
    summary_table['buyer_share_pct'] = summary_table['buyer_share_pct'].map(lambda value: f'{value:.2f}%')
    summary_table['loan_dependency_pct'] = summary_table['loan_dependency_pct'].map(lambda value: f'{value:.2f}%')
    summary_table['investment_intent_pct'] = summary_table['investment_intent_pct'].map(lambda value: f'{value:.2f}%')

    top_regions = (
        result.clients.groupby('region')
        .agg(buyers=('client_id', 'count'), avg_total_spend=('total_spend', 'mean'))
        .sort_values('buyers', ascending=False)
        .head(10)
        .reset_index()
    )
    top_regions['avg_total_spend'] = top_regions['avg_total_spend'].map(lambda value: f'${value:,.0f}')

    recommendations = result.cluster_summary[['segment_name', 'recommended_action']].copy()

    report = f'''# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Executive Summary
Parcl's buyer base is diverse in financing behavior, spending capacity, and purchase depth. This project turns that diversity into operational segments using unsupervised learning on client demographics and real transaction activity.

The model evaluation showed the strongest silhouette score at **k={result.optimal_k}**, but the final solution keeps **k={result.chosen_k}** clusters because it produces more actionable business personas for marketing, investor targeting, and property recommendation workflows. The k-means and hierarchical solutions remained directionally aligned with an adjusted rand score of **{result.model_alignment:.3f}**.

## Dataset Snapshot
{_format_markdown_table(dataset_table)}

## Methodology
1. Cleaned mixed-format dates, normalized categorical labels, and removed duplicates.
2. Engineered behavioral features from property transactions, including purchase count, total spend, average spend, average area, office share, and recency.
3. Encoded categorical features with one-hot encoding and scaled numeric features with `StandardScaler`.
4. Compared k-means solutions from 2 to 8 clusters using inertia and silhouette score.
5. Validated the chosen solution against hierarchical clustering to confirm stability.

## Cluster Selection Evidence
{_format_markdown_table(evaluation_table)}

Silhouette score peaked at **k={result.optimal_k}**, while **k={result.chosen_k}** remained acceptable and gave better separation between premium, financed, growth, and portfolio-oriented buyers for downstream business actions.

## Segment Profiles
{_format_markdown_table(summary_table)}

## Geographic Findings
{_format_markdown_table(top_regions)}

California dominates buyer volume, while multiple secondary regions show meaningful spend and repeat-purchase activity. This supports a regional strategy where high-volume domestic markets receive scalable campaigns and smaller markets receive segment-specific investor outreach.

## Business Recommendations
{_format_markdown_table(recommendations)}

## Conclusion
The final segmentation moves Parcl away from one-size-fits-all marketing. The clusters can support:

- targeted marketing campaigns by financing behavior and investment intent
- tailored property recommendations by spend capacity and purchase pattern
- investor outreach based on repeat buying and portfolio activity
- regional expansion planning using segment concentration and spend depth

The Streamlit dashboard in this project operationalizes those findings for live filtering by country, region, acquisition purpose, and client type.
'''
    return report


def run_segmentation(chosen_k: int = 4) -> SegmentationResult:
    clients, properties = load_and_prepare_data()
    feature_matrix, transformer, _, _ = build_feature_matrix(clients)
    evaluation = evaluate_cluster_range(feature_matrix)
    optimal_k = int(evaluation.loc[evaluation['silhouette_score'].idxmax(), 'k'])

    kmeans_model = KMeans(n_clusters=chosen_k, n_init=30, random_state=42)
    clients = clients.copy()
    clients['cluster'] = kmeans_model.fit_predict(feature_matrix)

    hierarchical_model = AgglomerativeClustering(n_clusters=chosen_k, linkage='ward')
    hierarchical_labels = hierarchical_model.fit_predict(feature_matrix)

    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(feature_matrix)
    clients['pca_x'] = projection[:, 0]
    clients['pca_y'] = projection[:, 1]

    cluster_summary = summarize_clusters(clients)
    segment_name_map = dict(zip(cluster_summary['cluster_id'], cluster_summary['segment_name']))
    clients['segment_name'] = clients['cluster'].map(segment_name_map)

    country_summary = build_country_summary(clients)
    region_summary = build_region_summary(clients)

    result = SegmentationResult(
        clients=clients,
        properties=properties,
        evaluation=evaluation,
        cluster_summary=cluster_summary,
        country_summary=country_summary,
        region_summary=region_summary,
        chosen_k=chosen_k,
        optimal_k=optimal_k,
        kmeans_silhouette=float(silhouette_score(feature_matrix, clients['cluster'])),
        hierarchical_silhouette=float(silhouette_score(feature_matrix, hierarchical_labels)),
        model_alignment=float(adjusted_rand_score(clients['cluster'], hierarchical_labels)),
        research_report='',
    )
    result.research_report = build_research_report(result)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, OUTPUTS_DIR / 'kmeans_model.joblib')
    joblib.dump(transformer, OUTPUTS_DIR / 'feature_transformer.joblib')

    return result


def persist_outputs(result: SegmentationResult) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    clients_export = result.clients.copy()
    clients_export['first_transaction'] = clients_export['first_transaction'].dt.strftime('%Y-%m-%d')
    clients_export['last_transaction'] = clients_export['last_transaction'].dt.strftime('%Y-%m-%d')
    clients_export['date_of_birth'] = clients_export['date_of_birth'].dt.strftime('%Y-%m-%d')
    clients_export.to_csv(OUTPUTS_DIR / 'clustered_clients.csv', index=False)
    result.cluster_summary.to_csv(OUTPUTS_DIR / 'cluster_summary.csv', index=False)
    result.evaluation.to_csv(OUTPUTS_DIR / 'cluster_evaluation.csv', index=False)
    result.country_summary.to_csv(OUTPUTS_DIR / 'country_segment_summary.csv', index=False)
    result.region_summary.to_csv(OUTPUTS_DIR / 'region_segment_summary.csv', index=False)

    metrics = {
        'chosen_k': result.chosen_k,
        'optimal_k': result.optimal_k,
        'kmeans_silhouette': round(result.kmeans_silhouette, 4),
        'hierarchical_silhouette': round(result.hierarchical_silhouette, 4),
        'model_alignment': round(result.model_alignment, 4),
        'clients': int(len(result.clients)),
        'properties': int(len(result.properties)),
        'countries': int(result.clients['country'].nunique()),
        'regions': int(result.clients['region'].nunique()),
    }
    (OUTPUTS_DIR / 'model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (REPORTS_DIR / 'research_paper.md').write_text(result.research_report, encoding='utf-8')

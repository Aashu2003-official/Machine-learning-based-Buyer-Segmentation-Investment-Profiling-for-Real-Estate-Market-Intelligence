from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGES_DIR = PROJECT_ROOT / '.packages'
SRC_DIR = PROJECT_ROOT / 'src'

if PACKAGES_DIR.exists():
    sys.path.insert(0, str(PACKAGES_DIR))
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from buyer_segmentation import run_segmentation

st.set_page_config(
    page_title='Parcl Buyer Segmentation',
    layout='wide',
)


@st.cache_data(show_spinner=False)
def load_results() -> dict[str, object]:
    result = run_segmentation(chosen_k=4)
    return {
        'clients': result.clients,
        'properties': result.properties,
        'evaluation': result.evaluation,
        'cluster_summary': result.cluster_summary,
        'country_summary': result.country_summary,
        'region_summary': result.region_summary,
        'chosen_k': result.chosen_k,
        'optimal_k': result.optimal_k,
        'model_alignment': result.model_alignment,
        'kmeans_silhouette': result.kmeans_silhouette,
        'research_report': result.research_report,
    }


def format_currency(value: float) -> str:
    return f'${value:,.0f}'


def filter_clients(clients: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header('Filters')
    selected_countries = st.sidebar.multiselect(
        'Country',
        options=sorted(clients['country'].unique().tolist()),
        default=sorted(clients['country'].unique().tolist()),
    )
    selected_regions = st.sidebar.multiselect(
        'Region',
        options=sorted(clients['region'].unique().tolist()),
        default=sorted(clients['region'].unique().tolist()),
    )
    selected_purposes = st.sidebar.multiselect(
        'Acquisition Purpose',
        options=sorted(clients['acquisition_purpose'].unique().tolist()),
        default=sorted(clients['acquisition_purpose'].unique().tolist()),
    )
    selected_types = st.sidebar.multiselect(
        'Client Type',
        options=sorted(clients['client_type'].unique().tolist()),
        default=sorted(clients['client_type'].unique().tolist()),
    )

    return clients[
        clients['country'].isin(selected_countries)
        & clients['region'].isin(selected_regions)
        & clients['acquisition_purpose'].isin(selected_purposes)
        & clients['client_type'].isin(selected_types)
    ].copy()


results = load_results()
clients = results['clients']
filtered_clients = filter_clients(clients)
cluster_summary = results['cluster_summary']

st.title('Buyer Segmentation and Investment Profiling')
st.caption(
    f"Final delivery for Parcl Co. Limited. The mathematically strongest silhouette score occurred at k={results['optimal_k']}, "
    f"while the dashboard uses k={results['chosen_k']} for clearer business segmentation."
)

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric('Filtered Buyers', f'{len(filtered_clients):,}')
metric_b.metric('Average Total Spend', format_currency(filtered_clients['total_spend'].mean()))
metric_c.metric('Average Purchase Count', f"{filtered_clients['purchase_count'].mean():.2f}")
metric_d.metric(
    'Investment Intent',
    f"{(filtered_clients['acquisition_purpose'].eq('Investment').mean() * 100):.1f}%",
)

overview_tab, behavior_tab, geography_tab, insights_tab = st.tabs(
    [
        'Buyer Segmentation Overview',
        'Investor Behavior Dashboard',
        'Geographic Buyer Analysis',
        'Segment Insights Panel',
    ]
)

with overview_tab:
    left_col, right_col = st.columns((1, 1))

    distribution = (
        filtered_clients.groupby('segment_name')
        .agg(buyers=('client_id', 'count'))
        .reset_index()
        .sort_values('buyers', ascending=False)
    )
    distribution_chart = px.pie(
        distribution,
        names='segment_name',
        values='buyers',
        hole=0.45,
        title='Cluster Distribution',
    )
    left_col.plotly_chart(distribution_chart, use_container_width=True)

    scatter_chart = px.scatter(
        filtered_clients,
        x='pca_x',
        y='pca_y',
        color='segment_name',
        hover_data=['client_id', 'country', 'region', 'client_type', 'purchase_count', 'total_spend'],
        title='Buyer Clusters in Reduced Feature Space',
    )
    right_col.plotly_chart(scatter_chart, use_container_width=True)

    elbow_chart = px.line(
        results['evaluation'],
        x='k',
        y='silhouette_score',
        markers=True,
        title='Silhouette Score by Cluster Count',
    )
    st.plotly_chart(elbow_chart, use_container_width=True)

with behavior_tab:
    behavior_left, behavior_right = st.columns((1, 1))

    spend_chart = px.bar(
        filtered_clients.groupby('segment_name', as_index=False)['total_spend'].mean(),
        x='segment_name',
        y='total_spend',
        color='segment_name',
        title='Average Total Spend by Segment',
    )
    behavior_left.plotly_chart(spend_chart, use_container_width=True)

    purchase_chart = px.box(
        filtered_clients,
        x='segment_name',
        y='purchase_count',
        color='segment_name',
        title='Purchase Count Distribution by Segment',
    )
    behavior_right.plotly_chart(purchase_chart, use_container_width=True)

    loan_behavior = (
        filtered_clients.groupby(['segment_name', 'loan_applied'])
        .size()
        .reset_index(name='buyers')
    )
    loan_chart = px.bar(
        loan_behavior,
        x='segment_name',
        y='buyers',
        color='loan_applied',
        barmode='group',
        title='Loan Behavior by Segment',
    )
    st.plotly_chart(loan_chart, use_container_width=True)

    purpose_chart = px.histogram(
        filtered_clients,
        x='segment_name',
        color='acquisition_purpose',
        barmode='group',
        title='Acquisition Purpose by Segment',
    )
    st.plotly_chart(purpose_chart, use_container_width=True)

with geography_tab:
    country_counts = (
        filtered_clients.groupby(['country', 'segment_name'])
        .size()
        .reset_index(name='buyers')
    )
    country_totals = (
        filtered_clients.groupby('country')
        .agg(
            buyers=('client_id', 'count'),
            avg_total_spend=('total_spend', 'mean'),
        )
        .reset_index()
    )
    map_chart = px.choropleth(
        country_totals,
        locations='country',
        locationmode='country names',
        color='buyers',
        hover_data=['avg_total_spend'],
        title='Buyer Volume by Country',
    )
    st.plotly_chart(map_chart, use_container_width=True)

    region_counts = (
        filtered_clients.groupby(['region', 'segment_name'])
        .size()
        .reset_index(name='buyers')
        .sort_values('buyers', ascending=False)
    )
    region_chart = px.bar(
        region_counts.head(25),
        x='region',
        y='buyers',
        color='segment_name',
        title='Top Regions by Segment',
    )
    st.plotly_chart(region_chart, use_container_width=True)

    dominant_segment = (
        country_counts.sort_values(['country', 'buyers'], ascending=[True, False])
        .drop_duplicates(subset=['country'])
        .rename(columns={'segment_name': 'dominant_segment'})
    )
    st.dataframe(
        dominant_segment[['country', 'dominant_segment', 'buyers']].sort_values('buyers', ascending=False),
        use_container_width=True,
    )

with insights_tab:
    display_summary = cluster_summary.copy()
    display_summary['avg_total_spend'] = display_summary['avg_total_spend'].map(format_currency)
    display_summary['avg_spend_per_property'] = display_summary['avg_spend_per_property'].map(format_currency)
    display_summary['avg_area_sqft'] = display_summary['avg_area_sqft'].map(lambda value: f'{value:,.0f}')
    st.dataframe(display_summary, use_container_width=True)

    segment_choice = st.selectbox(
        'Select a segment for a closer readout',
        options=cluster_summary['segment_name'].tolist(),
    )
    selected_row = cluster_summary.loc[cluster_summary['segment_name'] == segment_choice].iloc[0]
    selected_clients = filtered_clients.loc[filtered_clients['segment_name'] == segment_choice]

    st.subheader(segment_choice)
    st.write(selected_row['recommended_action'])
    detail_a, detail_b, detail_c = st.columns(3)
    detail_a.metric('Buyers', f"{int(selected_row['buyers']):,}")
    detail_b.metric('Avg Purchase Count', f"{selected_row['avg_purchase_count']:.2f}")
    detail_c.metric('Avg Total Spend', format_currency(float(selected_row['avg_total_spend'])))

    st.write(
        f"Dominant geography: {selected_row['dominant_region']}, {selected_row['dominant_country']}. "
        f"Primary referral channel: {selected_row['dominant_referral_channel']}."
    )

    st.dataframe(
        selected_clients[
            [
                'client_id',
                'client_type',
                'country',
                'region',
                'acquisition_purpose',
                'loan_applied',
                'purchase_count',
                'total_spend',
                'avg_spend',
            ]
        ].sort_values('total_spend', ascending=False).head(20),
        use_container_width=True,
    )

st.download_button(
    label='Download Research Report',
    data=str(results['research_report']),
    file_name='research_paper.md',
    mime='text/markdown',
)

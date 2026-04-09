# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Executive Summary
Parcl's buyer base is diverse in financing behavior, spending capacity, and purchase depth. This project turns that diversity into operational segments using unsupervised learning on client demographics and real transaction activity.

The model evaluation showed the strongest silhouette score at **k=4**, but the final solution keeps **k=4** clusters because it produces more actionable business personas for marketing, investor targeting, and property recommendation workflows. The k-means and hierarchical solutions remained directionally aligned with an adjusted rand score of **0.536**.

## Dataset Snapshot
| Metric | Value |
| --- | --- |
| Clients | 2000 |
| Property listings | 10000 |
| Distinct countries | 10 |
| Distinct regions | 57 |
| Chosen cluster count | 4 |
| Silhouette-best cluster count | 4 |

## Methodology
1. Cleaned mixed-format dates, normalized categorical labels, and removed duplicates.
2. Engineered behavioral features from property transactions, including purchase count, total spend, average spend, average area, office share, and recency.
3. Encoded categorical features with one-hot encoding and scaled numeric features with `StandardScaler`.
4. Compared k-means solutions from 2 to 8 clusters using inertia and silhouette score.
5. Validated the chosen solution against hierarchical clustering to confirm stability.

## Cluster Selection Evidence
| k | inertia | silhouette_score |
| --- | --- | --- |
| 2 | 22,351.24 | 0.1458 |
| 3 | 19,492.79 | 0.1618 |
| 4 | 17,494.82 | 0.1713 |
| 5 | 16,291.80 | 0.1486 |
| 6 | 15,506.28 | 0.1286 |
| 7 | 14,729.60 | 0.1325 |
| 8 | 14,269.36 | 0.1253 |

Silhouette score peaked at **k=4**, while **k=4** remained acceptable and gave better separation between premium, financed, growth, and portfolio-oriented buyers for downstream business actions.

## Segment Profiles
| cluster_id | segment_name | buyers | buyer_share_pct | avg_purchase_count | avg_total_spend | loan_dependency_pct | investment_intent_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | Portfolio Investors | 44 | 2.20% | 7.5 | $2,477,339 | 34.09% | 36.36% |
| 2 | Premium Buyers | 826 | 41.30% | 3.65 | $1,442,020 | 35.11% | 29.54% |
| 0 | Core Growth Buyers | 357 | 17.85% | 3.37 | $1,175,395 | 36.97% | 30.81% |
| 1 | Financed Value Seekers | 773 | 38.65% | 3.57 | $1,036,253 | 38.68% | 31.69% |

## Geographic Findings
| region | buyers | avg_total_spend |
| --- | --- | --- |
| California | 633 | $1,304,664 |
| Nevada | 143 | $1,280,537 |
| Colorado | 118 | $1,234,178 |
| Arizona | 108 | $1,237,170 |
| Oregon | 96 | $1,272,468 |
| Utah | 87 | $1,230,653 |
| Washington | 73 | $1,193,662 |
| Virginia | 65 | $1,269,861 |
| Texas | 56 | $1,198,793 |
| Florida | 50 | $1,333,435 |

California dominates buyer volume, while multiple secondary regions show meaningful spend and repeat-purchase activity. This supports a regional strategy where high-volume domestic markets receive scalable campaigns and smaller markets receive segment-specific investor outreach.

## Business Recommendations
| segment_name | recommended_action |
| --- | --- |
| Portfolio Investors | Prioritize bundled inventory, early access launches, and dedicated relationship management. |
| Premium Buyers | Recommend larger units and premium inventory with concierge-style follow-up. |
| Core Growth Buyers | Promote balanced investment options, repeat-purchase incentives, and cross-sell campaigns. |
| Financed Value Seekers | Lead with affordability messaging, loan assistance, and step-by-step decision support. |

## Conclusion
The final segmentation moves Parcl away from one-size-fits-all marketing. The clusters can support:

- targeted marketing campaigns by financing behavior and investment intent
- tailored property recommendations by spend capacity and purchase pattern
- investor outreach based on repeat buying and portfolio activity
- regional expansion planning using segment concentration and spend depth

The Streamlit dashboard in this project operationalizes those findings for live filtering by country, region, acquisition purpose, and client type.

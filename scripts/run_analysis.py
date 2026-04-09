from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = PROJECT_ROOT / '.packages'
SRC_DIR = PROJECT_ROOT / 'src'

if PACKAGES_DIR.exists():
    sys.path.insert(0, str(PACKAGES_DIR))
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from buyer_segmentation import persist_outputs, run_segmentation


def main() -> None:
    result = run_segmentation(chosen_k=4)
    persist_outputs(result)

    print('Analysis completed.')
    print(f'Chosen k: {result.chosen_k}')
    print(f'Silhouette-best k: {result.optimal_k}')
    print(f'K-means silhouette: {result.kmeans_silhouette:.4f}')
    print(f'Hierarchical silhouette: {result.hierarchical_silhouette:.4f}')
    print(f'Model alignment: {result.model_alignment:.4f}')
    print(f"Clustered clients export: {PROJECT_ROOT / 'outputs' / 'clustered_clients.csv'}")
    print(f"Research paper: {PROJECT_ROOT / 'reports' / 'research_paper.md'}")


if __name__ == '__main__':
    main()

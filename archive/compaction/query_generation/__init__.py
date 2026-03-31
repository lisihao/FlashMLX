"""
Query Generation Module

Implements paper's query selection algorithms:
- Self-Study: K-means clustering or importance sampling
- OMP: Orthogonal Matching Pursuit for query refinement
"""

from .self_study import (
    self_study_kmeans,
    self_study_importance_sampling,
    self_study_auto,
)

from .omp import (
    omp_refine_queries,
    omp_refine_queries_fast,
    compute_attention_output,
)

__all__ = [
    # Self-Study
    "self_study_kmeans",
    "self_study_importance_sampling",
    "self_study_auto",

    # OMP
    "omp_refine_queries",
    "omp_refine_queries_fast",
    "compute_attention_output",
]

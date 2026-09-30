"""
External Benchmark Module for Gym Exercise Classification.
Supports cross-dataset generalization evaluation on Fit3D and MM-Fit.
"""

from src.external.base import ExternalRecord, BaseExternalDataset, QualityControlGate
from src.external.canonical_pose import (
    openpose18_to_skelgym13,
    mediapipe33_to_skelgym13,
    fit3d_to_skelgym13,
    canonical_geometric_normalization,
    resample_temporal_sequence,
    skeleton_13_to_dataframe
)
from src.external.class_mapping import (
    get_class_mapping,
    get_target_skelgym_classes,
    get_target_skelgym_indices,
    get_closed_set_index_map
)
from src.external.mmfit import MMFitExternalDataset
from src.external.fit3d import Fit3DExternalDataset
from src.external.metrics import (
    evaluate_window_level,
    aggregate_hierarchical_predictions,
    compute_recording_level_bootstrap_ci,
    compute_paired_bootstrap_difference,
    compute_domain_gap_diagnostics,
    plot_external_confusion_matrix,
    generate_error_analysis_table
)
from src.external.fewshot import extract_penultimate_embeddings, simulate_one_shot_transfer

__all__ = [
    "ExternalRecord",
    "BaseExternalDataset",
    "QualityControlGate",
    "openpose18_to_skelgym13",
    "mediapipe33_to_skelgym13",
    "fit3d_to_skelgym13",
    "canonical_geometric_normalization",
    "resample_temporal_sequence",
    "skeleton_13_to_dataframe",
    "get_class_mapping",
    "get_target_skelgym_classes",
    "get_target_skelgym_indices",
    "get_closed_set_index_map",
    "MMFitExternalDataset",
    "Fit3DExternalDataset",
    "evaluate_window_level",
    "aggregate_hierarchical_predictions",
    "compute_recording_level_bootstrap_ci",
    "compute_paired_bootstrap_difference",
    "compute_domain_gap_diagnostics",
    "plot_external_confusion_matrix",
    "generate_error_analysis_table",
    "extract_penultimate_embeddings",
    "simulate_one_shot_transfer"
]

"""
External Dataset Class Mapping and Semantic Alignment.
Defines explicit, reproducible class correspondences between external benchmarks (Fit3D, MM-Fit)
and the canonical 22-class SkelGym taxonomy.
"""

from typing import Dict, List, Tuple, Optional
from src.constants import ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION

# MM-Fit Class Mappings (Official 10-class taxonomy -> SkelGym 22-class)
MMFIT_CORE4: Dict[str, str] = {
    "squats": "squat",
    "pushups": "push-up",
    "dumbbell_shoulder_press": "shoulder press",
    "lateral_shoulder_raises": "lateral raise",
}

MMFIT_EXTENDED5: Dict[str, str] = {
    **MMFIT_CORE4,
    "bicep_curls": "barbell biceps curl",
}

# Fit3D Class Mappings (Official 47-exercise taxonomy -> SkelGym 22-class)
FIT3D_CORE6: Dict[str, str] = {
    "deadlift": "deadlift",
    "squat": "squat",
    "pushup": "push-up",
    "side_lateral_raise": "lateral raise",
    "dumbbell_overhead_shoulder_press": "shoulder press",
    "neutral_overhead_shoulder_press": "shoulder press",
    "dumbbell_hammer_curls": "hammer curl",
}

FIT3D_EXTENDED7: Dict[str, str] = {
    **FIT3D_CORE6,
    "dumbbell_biceps_curls": "barbell biceps curl",
}

# Non-mapped distractor classes explicitly excluded with scientific rationale
MMFIT_EXCLUDED_CLASSES = {
    "lunges": "No exact matching lower-body unilateral exercise in SkelGym",
    "situps": "Abdominal floor exercise not present in SkelGym",
    "dumbbell_rows": "Dumbbell row kinematics differ substantially from barbell t-bar row",
    "tricep_extensions": "Overhead extension mechanics differ from cable tricep pushdown / dips",
    "jumping_jacks": "Calisthenic cardio movement not present in SkelGym"
}

FIT3D_EXCLUDED_CLASSES = [
    # Fit3D classes outside overlapping resistance training taxonomy
]

def get_class_mapping(dataset: str, class_set: str = "core") -> Dict[str, str]:
    """
    Returns dictionary mapping external raw class name -> canonical SkelGym class name.
    """
    ds = dataset.lower()
    cs = class_set.lower()
    if ds == "mmfit":
        if "ext" in cs or "5" in cs:
            return MMFIT_EXTENDED5
        return MMFIT_CORE4
    elif ds == "fit3d":
        if "ext" in cs or "7" in cs:
            return FIT3D_EXTENDED7
        return FIT3D_CORE6
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mmfit' or 'fit3d'.")

def get_target_skelgym_classes(dataset: str, class_set: str = "core") -> List[str]:
    """
    Returns sorted list of unique SkelGym class names involved in this benchmark.
    """
    mapping = get_class_mapping(dataset, class_set)
    unique_skel = sorted(list(set(mapping.values())))
    return unique_skel

def get_target_skelgym_indices(dataset: str, class_set: str = "core") -> List[int]:
    """
    Returns list of 0-based indices into canonical 22 SkelGym classes for the benchmark.
    """
    classes = get_target_skelgym_classes(dataset, class_set)
    return [ACTION_TO_IDX[c] for c in classes]

def get_closed_set_index_map(dataset: str, class_set: str = "core") -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns (skelgym_to_closed, closed_to_skelgym) index mapping dictionaries.
    """
    skel_indices = get_target_skelgym_indices(dataset, class_set)
    skel_to_closed = {skel_idx: closed_idx for closed_idx, skel_idx in enumerate(skel_indices)}
    closed_to_skel = {closed_idx: skel_idx for closed_idx, skel_idx in enumerate(skel_indices)}
    return skel_to_closed, closed_to_skel

"""
Core constants for Gym Exercise Classification.
Includes 22 action classes, keypoint definitions, skeleton graph connections, and feature dimensions.
"""

from typing import List, Tuple, Dict

# Canonical 22 Action Classes (sorted as in metadata / publication)
ACTIONS: List[str] = [
    "barbell biceps curl",
    "bench press",
    "chest fly machine",
    "deadlift",
    "decline bench press",
    "hammer curl",
    "hip thrust",
    "incline bench press",
    "lat pulldown",
    "lateral raise",
    "leg extension",
    "leg raises",
    "plank",
    "pull Up",
    "push-up",
    "romanian deadlift",
    "russian twist",
    "shoulder press",
    "squat",
    "t bar row",
    "tricep Pushdown",
    "tricep dips"
]

NUM_CLASSES: int = len(ACTIONS)
ACTION_TO_IDX: Dict[str, int] = {act: i for i, act in enumerate(ACTIONS)}
IDX_TO_ACTION: Dict[int, str] = {i: act for i, act in enumerate(ACTIONS)}

# MediaPipe 33 Landmark Names
RAW_POINTS_33: List[str] = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX"
]

# 13 Key Body Joints (Core posture joints)
RAW_POINTS_13: List[str] = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

# 12 Relative Body Joints (Legacy: Excluding NOSE which served as coordinate origin)
REL_POINTS_12: List[str] = [j for j in RAW_POINTS_13 if j != "NOSE"]

# Hip joints used to compute the hip-midpoint origin for relative features
HIP_MIDPOINT_JOINTS: Tuple[str, str] = ("LEFT_HIP", "RIGHT_HIP")

# Skeleton Graph Connections for 33 Joints (MediaPipe POSE_CONNECTIONS)
POSE_CONNECTIONS_33: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),
    (15, 21), (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (31, 27), (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)
]

# Skeleton Graph Connections for 13 Joints
EDGES_13: List[Tuple[int, int]] = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8),
    (7, 8),
    (7, 9), (9, 11),
    (8, 10), (10, 12)
]

# Feature Dimensions Mapping
# - raw_2d: 13 * 2 (x, y) = 26
# - raw_3d: 13 * 3 (x, y, z) = 39
# - rel_2d: 13 * 2 (hip-midpoint-relative x, y) = 26
# - rel_3d: 13 * 3 (hip-midpoint-relative x, y, z) = 39
# - angle_2d: C(13, 3) = 286 planar triplet angles
# - angle_3d: C(13, 3) = 286 3D spatial vector angles
# - mix: 39 (rel_3d) + 286 (angle_3d) = 325
# - Legacy:
#   - full_4: 33 * 4 = 132
#   - full_rel_4: 33 * 4 + 1 = 133 (hip-midpoint-relative, all 33 joints)
#   - 13_4: 13 * 4 = 52
#   - 12rel_4: 13 * 4 + 1 = 53 (hip-midpoint-relative, all 13 joints)
#   - angle3: 286, angle2: 78, direct_concat: 339
# Kinematic Parent Mapping for 13 Joints: joint_idx -> parent_joint_idx (None for root)
KINEMATIC_TREE_13: Dict[int, Optional[int]] = {
    0: None,  # NOSE (Root reference)
    1: 0,     # LEFT_SHOULDER -> NOSE
    2: 0,     # RIGHT_SHOULDER -> NOSE
    3: 1,     # LEFT_ELBOW -> LEFT_SHOULDER
    4: 2,     # RIGHT_ELBOW -> RIGHT_SHOULDER
    5: 3,     # LEFT_WRIST -> LEFT_ELBOW
    6: 4,     # RIGHT_WRIST -> RIGHT_ELBOW
    7: 1,     # LEFT_HIP -> LEFT_SHOULDER
    8: 2,     # RIGHT_HIP -> RIGHT_SHOULDER
    9: 7,     # LEFT_KNEE -> LEFT_HIP
    10: 8,    # RIGHT_KNEE -> RIGHT_HIP
    11: 9,    # LEFT_ANKLE -> LEFT_KNEE
    12: 10    # RIGHT_ANKLE -> RIGHT_KNEE
}

FEATURE_DIMS: Dict[str, int] = {
    "raw_2d": 26,
    "raw_3d": 39,
    "rel_2d": 26,
    "rel_3d": 39,
    "bone_2d": 26,
    "bone_3d": 39,
    "angle_2d": 286,
    "angle_3d": 286,
    "mix": 325,
    "full_4": 132,
    "full_rel_4": 133,
    "13_4": 52,
    "12rel_4": 53,
    "angle3": 286,
    "angle2": 78,
    "direct_concat": 339,
    "branch_concat": -1  # Dual branch tuple (53, 286)
}

# Standard Sliding Window defaults
DEFAULT_SEQ_LEN: int = 32
DEFAULT_TRAIN_STRIDE: int = 16
DEFAULT_VAL_TEST_STRIDE: int = 32

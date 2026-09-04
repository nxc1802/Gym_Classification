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

# 12 Relative Body Joints (Excluding NOSE which serves as coordinate origin)
REL_POINTS_12: List[str] = [j for j in RAW_POINTS_13 if j != "NOSE"]

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
# - full_4: 33 * 4 (x, y, z, visibility) = 132
# - full_rel_4: 32 * 4 (rel_x, rel_y, rel_z, visibility) + 1 (nose_vis) = 129
# - 13_4: 13 * 4 (x, y, z, visibility) = 52
# - 12rel_4: 12 * 4 (rel_x, rel_y, rel_z, visibility) = 48 (or 49 if nose_vis included)
# - angle3: C(13, 3) = 286 triplet angles
# - angle2: C(13, 2) = 78 pair absolute angles
# - direct_concat: 48 (or 49) + 286 = 334 (or 335)
FEATURE_DIMS: Dict[str, int] = {
    "full_4": 132,
    "full_rel_4": 129,
    "13_4": 52,
    "12rel_4": 49,
    "angle3": 286,
    "angle2": 78,
    "direct_concat": 335,
    "branch_concat": -1  # Dual branch tuple (49, 286)
}

# Standard Sliding Window defaults
DEFAULT_SEQ_LEN: int = 32
DEFAULT_TRAIN_STRIDE: int = 16
DEFAULT_VAL_TEST_STRIDE: int = 32

"""
MediaPipe 33-Joint Skeleton Graph Definition for BlockGCN (CVPR 2024).
Computes 3-subset spatial adjacency matrix (self-link, inward, outward) and k-hop distance matrices.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch

from src.constants import POSE_CONNECTIONS_33

NUM_NODE_33 = 33

# Auxiliary anatomical edges connecting disjoint components (face, mouth, and shoulders)
# As specified in docs/external_baseline.md Section 1:
# - Connect Nose (0) to Left Shoulder (11) and Right Shoulder (12)
# - Connect Mouth Left (9) to Nose (0) and Mouth Right (10) to Nose (0)
AUXILIARY_EDGES_33: List[Tuple[int, int]] = [
    (0, 11),  # Nose -> Left Shoulder
    (0, 12),  # Nose -> Right Shoulder
    (9, 0),   # Mouth Left -> Nose
    (10, 0),  # Mouth Right -> Nose
]

def build_mediapipe33_edges() -> List[Tuple[int, int]]:
    """
    Returns canonical directed inward edges for 33 MediaPipe joints.
    Edges point inward towards the torso/hips center.
    """
    all_edges = set(POSE_CONNECTIONS_33 + AUXILIARY_EDGES_33)
    inward = list(all_edges)
    return sorted(inward)

def edge2mat(link: List[Tuple[int, int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1.0
    return A

def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = np.sum(A, axis=0)
    w = A.shape[1]
    Dn = np.zeros((w, w), dtype=np.float32)
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn).astype(np.float32)

def get_spatial_graph(num_node: int, self_link: List[Tuple[int, int]], inward: List[Tuple[int, int]], outward: List[Tuple[int, int]]) -> np.ndarray:
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack([I, In, Out], axis=0).astype(np.float32)

def compute_hop_distance_matrix(num_node: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Computes shortest path graph distance between all joint pairs using Floyd-Warshall.
    Returns integer matrix of shape (num_node, num_node).
    """
    dist = np.full((num_node, num_node), fill_value=999, dtype=np.int64)
    np.fill_diagonal(dist, 0)
    for u, v in edges:
        dist[u, v] = 1
        dist[v, u] = 1

    for k in range(num_node):
        for i in range(num_node):
            for j in range(num_node):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

class MediaPipe33Graph:
    """
    Graph adapter for MediaPipe 33 joints compatible with BlockGCN.
    Provides:
      - self.A: (3, 33, 33) 3-subset spatial adjacency matrix.
      - self.hops: (33, 33) shortest path hop distance matrix.
    """
    def __init__(self, labeling_mode: str = "spatial"):
        self.num_node = NUM_NODE_33
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = build_mediapipe33_edges()
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

        self.A = self.get_adjacency_matrix(labeling_mode)
        self.hops = compute_hop_distance_matrix(self.num_node, self.neighbor)

    def get_adjacency_matrix(self, labeling_mode: str = "spatial") -> np.ndarray:
        if labeling_mode == "spatial":
            return get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError(f"Unsupported labeling_mode: {labeling_mode}")

    def get_hops_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.hops).long()

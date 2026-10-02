"""
BlockGCN External Baseline Subsystem for SkelGym.
Provides:
  - BlockGCNModel
  - MediaPipe33Graph
  - BlockGCNDataset, get_blockgcn_dataloaders
  - BlockGCNTrainer
"""

from .graph import MediaPipe33Graph, NUM_NODE_33
from .model import BlockGCNModel
from .dataset import BlockGCNDataset, get_blockgcn_dataloaders
from .trainer import BlockGCNTrainer

__all__ = [
    "MediaPipe33Graph",
    "NUM_NODE_33",
    "BlockGCNModel",
    "BlockGCNDataset",
    "get_blockgcn_dataloaders",
    "BlockGCNTrainer"
]

"""
Isolated Unit Tests for BlockGCN External Baseline Subsystem.
Verifies:
  1. MediaPipe 33 Graph construction and connectivity (no disconnected islands)
  2. BlockGCN architecture forward pass and gradient flow
  3. 33-joint 5D tensor compatibility and parameter count
"""

import unittest
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import NUM_CLASSES
from src.external.blockgcn import MediaPipe33Graph, BlockGCNModel


class TestBlockGCNIsolated(unittest.TestCase):
    def test_01_graph_generation(self):
        graph = MediaPipe33Graph()
        # Spatial adjacency matrix: 3 subsets (self-link, inward, outward)
        self.assertEqual(graph.A.shape, (3, 33, 33))
        self.assertEqual(graph.hops.shape, (33, 33))

        # Ensure all node pairs are connected (finite hop distance <= 32)
        max_dist = int(graph.hops.max())
        self.assertLess(max_dist, 50, f"Graph has disconnected components! Max hop distance: {max_dist}")
        self.assertEqual(int(graph.hops.diagonal().max()), 0, "Diagonal of hop distance matrix must be 0")

    def test_02_model_forward_pass(self):
        model = BlockGCNModel(num_class=NUM_CLASSES, num_point=33, num_person=1, in_channels=3)
        model.eval()

        # Canonical 5D input: (B=2, C=3, T=32, V=33, M=1)
        x_5d = torch.randn(2, 3, 32, 33, 1)
        with torch.no_grad():
            out = model(x_5d)
        self.assertEqual(out.shape, (2, NUM_CLASSES))
        self.assertFalse(torch.isnan(out).any())

    def test_03_backward_gradient_flow(self):
        model = BlockGCNModel(num_class=NUM_CLASSES, num_point=33, num_person=1, in_channels=3)
        model.train()

        x = torch.randn(2, 3, 32, 33, 1)
        y = torch.tensor([0, 1], dtype=torch.long)
        criterion = torch.nn.CrossEntropyLoss()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Check gradients in input projection and GCN blocks
        self.assertIsNotNone(model.to_joint_embedding.weight.grad)
        self.assertFalse(torch.isnan(model.to_joint_embedding.weight.grad).any())
        self.assertIsNotNone(model.fc.weight.grad)

    def test_04_parameter_count(self):
        model = BlockGCNModel(num_class=NUM_CLASSES, num_point=33, num_person=1, in_channels=3)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Original BlockGCN has 10 GCN-TCN blocks (~1.5M to 2.5M parameters)
        self.assertGreater(n_params, 1_000_000, f"BlockGCN params {n_params} unexpectedly too small")
        self.assertLess(n_params, 5_000_000, f"BlockGCN params {n_params} unexpectedly too large")


if __name__ == "__main__":
    unittest.main()

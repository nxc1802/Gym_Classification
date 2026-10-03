"""
Dedicated Training Engine for BlockGCN (CVPR 2024).
Implements the exact upstream recipe:
  - SGD with Nesterov momentum 0.9, weight decay 4e-4
  - 5-epoch linear warmup + MultiStepLR at [110, 120]
  - 140 epochs (no early stopping)
  - Validation-accuracy checkpoint selection
  - Full dual-level evaluation: Window-Level and Video-Level Consensus
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.constants import NUM_CLASSES, ACTIONS
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions
from src.utils.reproducibility import get_git_commit_hash, is_git_repo_dirty, compute_file_sha256, save_provenance_metadata


class BlockGCNTrainer:
    """
    Independent Trainer for BlockGCN adhering strictly to the author's optimization protocol.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        base_lr: float = 0.05,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0004,
        warm_up_epoch: int = 5,
        steps: Optional[List[int]] = None,
        lr_decay_rate: float = 0.1,
        max_epochs: int = 140,
        checkpoint_dir: str = "checkpoints/external",
        model_name: str = "BlockGCN",
        seed: Optional[int] = 42,
        smoke_test: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.base_lr = base_lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.warm_up_epoch = warm_up_epoch
        self.steps = steps or [110, 120]
        self.lr_decay_rate = lr_decay_rate
        self.max_epochs = max_epochs
        self.smoke_test = smoke_test
        self.seed = seed
        self.model_name = model_name

        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.smoke_test:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint_path = self.checkpoint_dir / f"best_{model_name}_seed{seed}.pt"
            self.last_checkpoint_path = self.checkpoint_dir / f"last_{model_name}_seed{seed}.pt"
        else:
            self.best_checkpoint_path = None
            self.last_checkpoint_path = None

        # Optimizer: SGD Nesterov
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=base_lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )

        # Learning Rate Schedule: Warmup + MultiStepLR
        def lr_lambda(epoch: int) -> float:
            if epoch < self.warm_up_epoch:
                return float(epoch + 1) / float(max(1, self.warm_up_epoch))
            factor = 1.0
            for step_ep in self.steps:
                if epoch >= step_ep:
                    factor *= self.lr_decay_rate
            return factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.criterion = nn.CrossEntropyLoss()

        # AMP
        self.use_amp = (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(X)
                    loss = self.criterion(out, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(X)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for X, y in val_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(X)
                    loss = self.criterion(out, y)
            else:
                out = self.model(X)
                loss = self.criterion(out, y)

            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

        val_loss = total_loss / max(1, total)
        val_acc = correct / max(1, total)
        val_metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        val_macro_f1 = float(val_metrics["macro_f1"])
        return val_loss, val_acc, val_macro_f1

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 140,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_macro_f1": [],
            "lr": []
        }
        best_val_acc = -1.0

        if verbose:
            mode_str = "[SMOKE TEST]" if self.smoke_test else "[FULL RUN]"
            print(f"{mode_str} Training BlockGCN on {self.device} for {epochs} epochs (Target: Best Val Accuracy) ...")

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = self.train_epoch(train_loader)
            v_loss, v_acc, v_f1 = self.validate(val_loader)
            self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_macro_f1"].append(v_f1)
            history["lr"].append(cur_lr)

            elapsed = time.perf_counter() - t0
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] - "
                    f"loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - "
                    f"val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - val_f1: {v_f1:.4f} - "
                    f"lr: {cur_lr:.2e}"
                )

            # Checkpoint logic (Skipped entirely if smoke_test=True)
            if not self.smoke_test:
                last_payload = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "history": history
                }
                torch.save(last_payload, self.last_checkpoint_path)

                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    provenance = {
                        "model_name": self.model_name,
                        "model_class": "BlockGCNModel",
                        "feature_method": "raw_33_xyz",
                        "seed": self.seed,
                        "git_sha": get_git_commit_hash(),
                        "git_dirty": is_git_repo_dirty(),
                        "saved_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "best_metric_value": float(best_val_acc),
                        "val_loss": float(v_loss),
                        "val_acc": float(v_acc),
                        "val_macro_f1": float(v_f1),
                        "hyperparameters": {
                            "optimizer": "SGD",
                            "base_lr": self.base_lr,
                            "momentum": self.momentum,
                            "nesterov": self.nesterov,
                            "weight_decay": self.weight_decay,
                            "warm_up_epoch": self.warm_up_epoch,
                            "steps": self.steps
                        }
                    }
                    best_payload = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "best_metric": float(best_val_acc),
                        "history": history,
                        "provenance": provenance
                    }
                    torch.save(best_payload, self.best_checkpoint_path)
                    sha256 = compute_file_sha256(self.best_checkpoint_path)
                    provenance["checkpoint_file"] = self.best_checkpoint_path.name
                    provenance["checkpoint_sha256"] = sha256
                    sidecar_path = self.best_checkpoint_path.with_suffix(".provenance.json")
                    save_provenance_metadata(provenance, sidecar_path)

        # Reload best weights if saved
        if not self.smoke_test and self.best_checkpoint_path.exists():
            payload = torch.load(self.best_checkpoint_path, map_location=self.device)
            state = payload.get("model_state_dict", payload)
            self.model.load_state_dict(state)

        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_targets = []
        all_probs = []

        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(X)
            else:
                logits = self.model(X)

            probs = F.softmax(logits, dim=1).float().cpu().numpy()
            all_probs.append(probs)
            all_targets.extend(y.cpu().numpy().tolist())

        y_prob = np.concatenate(all_probs, axis=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.array(all_targets, dtype=np.int64)
        return y_true, y_pred, y_prob

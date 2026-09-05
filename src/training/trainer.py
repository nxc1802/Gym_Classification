"""
PyTorch Training and Evaluation Engine.
Supports:
  - Automatic Mixed Precision (AMP) with bfloat16 / float16 for massive Blackwell / CUDA acceleration
  - Early Stopping, ReduceLROnPlateau scheduler, Class Weighting
  - Dual checkpointing ('best' and 'last')
  - Direct upload of checkpoints to Hugging Face Hub
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.training.metrics import compute_metrics
from src.utils.hf_hub import upload_checkpoints_to_hf, DEFAULT_MODEL_REPO

class Trainer:
    """
    High-Throughput Trainer for gym exercise classification models.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 20,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
        use_class_weights: bool = False,
        label_smoothing: float = 0.0,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        push_to_hf: bool = False,
        hf_repo: str = DEFAULT_MODEL_REPO,
        hf_token: Optional[str] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.best_checkpoint_path = self.checkpoint_dir / f"best_{model_name}.pt"
        self.last_checkpoint_path = self.checkpoint_dir / f"last_{model_name}.pt"
        self.use_class_weights = use_class_weights
        self.label_smoothing = label_smoothing

        # AMP Configuration
        self.use_amp = use_amp and (device.type == "cuda")
        if self.use_amp:
            if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bfloat16 has same dynamic range as float32, no scaler needed
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        # HF Configuration
        self.push_to_hf = push_to_hf
        self.hf_repo = hf_repo
        self.hf_token = hf_token

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def _get_criterion(self, train_loader: DataLoader) -> nn.Module:
        if self.use_class_weights:
            from src.constants import NUM_CLASSES
            all_labels = []
            for _, y in train_loader:
                all_labels.extend(y.tolist())
            classes = np.unique(all_labels)
            if len(classes) > 1:
                weights = compute_class_weight("balanced", classes=classes, y=all_labels)
                weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32)
                for c, w in zip(classes, weights):
                    if c < NUM_CLASSES:
                        weight_tensor[c] = float(w)
                return nn.CrossEntropyLoss(
                    weight=weight_tensor.to(self.device),
                    label_smoothing=self.label_smoothing
                )
            else:
                return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            if isinstance(X, (tuple, list)):
                X = tuple(x_item.to(self.device, non_blocking=True) for x_item in X)
            else:
                X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    out = self.model(X)
                    loss = criterion(out, y)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                out = self.model(X)
                loss = criterion(out, y)
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
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in val_loader:
            if isinstance(X, (tuple, list)):
                X = tuple(x_item.to(self.device, non_blocking=True) for x_item in X)
            else:
                X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    out = self.model(X)
                    loss = criterion(out, y)
            else:
                out = self.model(X)
                loss = criterion(out, y)

            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        val_loss = total_loss / max(1, total)
        val_acc = correct / max(1, total)
        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Executes full training loop with early stopping and checkpoint saving.
        Saves both best checkpoint and last checkpoint.
        """
        criterion = self._get_criterion(train_loader)
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

        best_val_loss = float("inf")
        patience_counter = 0

        amp_info = f" [AMP: {self.amp_dtype}]" if self.use_amp else " [FP32]"
        if verbose:
            print(f"Starting training on {self.device}{amp_info} for {epochs} epochs ...")

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = self.train_epoch(train_loader, criterion)
            v_loss, v_acc = self.validate(val_loader, criterion)
            self.scheduler.step(v_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["lr"].append(current_lr)

            elapsed = time.perf_counter() - t0

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] - "
                    f"loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - "
                    f"val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - "
                    f"lr: {current_lr:.1e}"
                )

            # Save last checkpoint every epoch
            torch.save(self.model.state_dict(), self.last_checkpoint_path)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
                if verbose:
                    print(f"  --> Best checkpoint saved: {self.best_checkpoint_path.name}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Reload best weights
        if self.best_checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))

        # Push to Hugging Face Hub if requested
        if self.push_to_hf:
            print(f"[HF Hub] Uploading best and last checkpoints to {self.hf_repo} ...")
            upload_checkpoints_to_hf(
                best_ckpt_path=str(self.best_checkpoint_path),
                last_ckpt_path=str(self.last_checkpoint_path),
                model_name=self.model_name,
                repo_id=self.hf_repo,
                token=self.hf_token
            )

        return history

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates predictions and softmax probability distributions.
        Returns: (y_true, y_pred, y_prob)
        """
        self.model.eval()
        all_y = []
        all_preds = []
        all_probs = []

        for X, y in dataloader:
            if isinstance(X, (tuple, list)):
                X = tuple(x_item.to(self.device, non_blocking=True) for x_item in X)
            else:
                X = X.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    out = self.model(X)
            else:
                out = self.model(X)

            probs = torch.softmax(out, dim=1).float().cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

            all_y.extend(y.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        return np.array(all_y), np.array(all_preds), np.array(all_probs)

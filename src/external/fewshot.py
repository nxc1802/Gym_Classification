"""
One-Shot Cross-Dataset Metric Transfer Evaluation.
Simulates 1-shot transfer learning across N trials (default 100) using frozen penultimate embeddings.
Enforces strict subject/workout group isolation: support and query samples NEVER share the same workout/recording.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.constants import ACTIONS

def extract_penultimate_embeddings(
    model: Any,
    features: np.ndarray,
    model_type: str,
    device: Any,
    batch_size: int = 32
) -> np.ndarray:
    """
    Extracts penultimate representation vectors for input windows.
    Input features: (N, T, D).
    """
    import torch
    model.eval()
    embeddings = []
    N = len(features)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(features[i:i + batch_size]).float().to(device)
            if model_type == "Transformer":
                x_norm = model.in_norm(batch)
                h = model.input_proj(x_norm)
                h = model.input_drop(h)
                h = model.pos_encoder(h)
                enc = model.norm(model.transformer_encoder(h))
                emb = enc.mean(dim=1)
            elif model_type == "AAGCN":
                B, T, D = batch.shape
                V = model.num_joints
                C = model.c_per_joint
                expected_len = V * C
                if D >= expected_len:
                    x_joints = batch[:, :, :expected_len].reshape(B, T, V, C)
                else:
                    pad = torch.zeros(B, T, expected_len - D, device=batch.device, dtype=batch.dtype)
                    x_padded = torch.cat([batch, pad], dim=-1)
                    x_joints = x_padded.reshape(B, T, V, C)
                x_in = x_joints.permute(0, 3, 1, 2).contiguous()
                out = x_in
                for block in model.blocks:
                    out = block(out)
                emb = model.gap(out).squeeze(-1).squeeze(-1)
            elif model_type == "STGCN":
                B, T, D = batch.shape
                V = model.num_joints
                C = model.c_per_joint
                expected_len = V * C
                if D >= expected_len:
                    x_joints = batch[:, :, :expected_len].reshape(B, T, V, C)
                else:
                    pad = torch.zeros(B, T, expected_len - D, device=batch.device, dtype=batch.dtype)
                    x_padded = torch.cat([batch, pad], dim=-1)
                    x_joints = x_padded.reshape(B, T, V, C)
                x_in = x_joints.permute(0, 3, 1, 2).contiguous()
                out = model.data_bn(x_in)
                for block in model.blocks:
                    out = block(out)
                emb = model.gap(out).squeeze(-1).squeeze(-1)
            else:
                emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 128), dtype=np.float32)

def simulate_one_shot_transfer(
    embeddings_by_record: Dict[str, np.ndarray],
    record_to_class: Dict[str, int],
    record_to_subject: Dict[str, str],
    target_class_indices: List[int],
    n_trials: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Simulates 1-shot transfer across n_trials.
    In each trial:
      - 1 support recording per target class is sampled.
      - All remaining recordings act as query.
      - Isolation Rule: If query shares subject/workout with support, it is excluded to prevent leakage.
      - Nearest Neighbor classification based on Cosine Similarity.
    """
    rng = np.random.default_rng(seed)
    
    # Organize records by target class
    records_by_class = {c: [] for c in target_class_indices}
    for rec_id, c_idx in record_to_class.items():
        if c_idx in records_by_class:
            records_by_class[c_idx].append(rec_id)

    # Ensure all classes have at least 2 records for 1-shot (1 support, >=1 query)
    valid_classes = [c for c in target_class_indices if len(records_by_class[c]) >= 2]
    if len(valid_classes) < len(target_class_indices):
        print(f"[Warning] Some classes have < 2 records for 1-shot evaluation: "
              f"valid={len(valid_classes)}/{len(target_class_indices)}")

    trial_accuracies = []
    trial_records = []

    for trial_idx in range(n_trials):
        support_records = {}
        support_subjects = set()

        # Sample 1 support recording per class
        for c in valid_classes:
            cand_records = records_by_class[c]
            chosen_rec = rng.choice(cand_records)
            support_records[c] = chosen_rec
            support_subjects.add(record_to_subject.get(chosen_rec, chosen_rec))

        # Query records: all remaining records from valid classes
        # Strict global workout-disjoint isolation: any workout present in support is completely excluded from query
        query_records = []
        query_labels = []
        for c in valid_classes:
            for r in records_by_class[c]:
                if r == support_records[c]:
                    continue
                subj = record_to_subject.get(r, r)
                if subj in support_subjects:
                    continue
                query_records.append(r)
                query_labels.append(c)

        if len(query_records) == 0:
            continue

        # Build support vectors (K_classes, D)
        support_vecs = np.stack([
            embeddings_by_record[support_records[c]] / (np.linalg.norm(embeddings_by_record[support_records[c]]) + 1e-12)
            for c in valid_classes
        ])

        # Build query vectors (N_query, D)
        query_vecs = np.stack([
            embeddings_by_record[r] / (np.linalg.norm(embeddings_by_record[r]) + 1e-12)
            for r in query_records
        ])

        # Cosine Similarity: (N_query, K_classes)
        sim_matrix = np.dot(query_vecs, support_vecs.T)
        pred_indices = np.argmax(sim_matrix, axis=1)
        pred_classes = np.array([valid_classes[p] for p in pred_indices])

        acc = float(np.mean(pred_classes == np.array(query_labels)) * 100.0)
        trial_accuracies.append(acc)
        trial_records.append({
            "trial": trial_idx,
            "accuracy": acc,
            "n_query": len(query_records)
        })

    trial_accuracies = np.array(trial_accuracies)
    return {
        "mean": float(np.mean(trial_accuracies)),
        "std": float(np.std(trial_accuracies)),
        "ci_95": [float(np.percentile(trial_accuracies, 2.5)), float(np.percentile(trial_accuracies, 97.5))],
        "n_trials_completed": len(trial_accuracies),
        "trials_df": pd.DataFrame(trial_records)
    }

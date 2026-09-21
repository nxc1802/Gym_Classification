#!/usr/bin/env python3
"""
Reproducible evaluation script for the external S&C benchmark (Deyzel et al. CVPRW 2023 protocol).
Covers:
1. Closed-set & Open-set evaluation across the 4 shared S&C exercises
   (squat, deadlift, barbell biceps curl, lateral raise) across 54 held-out test videos (N=529 windows).
2. Squat vs. Deadlift Ambiguity Breakdown (The Deyzel Dilemma) across 25 held-out test videos (N=305 windows).
3. 1-Shot Transfer Learning simulation across 100 trials:
   K=1 random support video per class (4 support videos), 50 query videos,
   nearest-neighbor cosine similarity on video mean embeddings.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.models.ensemble import WeightedSoftVotingEnsemble, aggregate_video_level_predictions

SC_CLASS_NAMES = ['barbell biceps curl', 'deadlift', 'lateral raise', 'squat']

def load_model(model_type, feat_type, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    m = build_model(model_type, feat_type, num_classes=22)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    return m

def predict_model(model, loader, device):
    trainer = Trainer(model=model, device=device)
    y_true, y_pred, y_probs = trainer.predict(loader)
    return y_probs, y_true

def get_embeddings(model, loader, device, model_type):
    """Extract penultimate pooled embeddings per window."""
    embeds = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch["features"]
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            elif isinstance(x, (list, tuple)):
                x = [xi.to(device) for xi in x]
            
            if model_type == "Transformer":
                x_norm = model.in_norm(x)
                h = model.input_proj(x_norm)
                h = model.input_drop(h)
                h = model.pos_encoder(h)
                enc = model.norm(model.transformer_encoder(h))
                emb = enc.mean(dim=1)
            elif model_type == "AAGCN":
                # x shape: (B, T, D)
                B, T, D = x.shape
                V = model.num_joints
                C = model.c_per_joint
                expected_len = V * C
                if D >= expected_len:
                    x_joints = x[:, :, :expected_len].reshape(B, T, V, C)
                else:
                    pad = torch.zeros(B, T, expected_len - D, device=x.device, dtype=x.dtype)
                    x_padded = torch.cat([x, pad], dim=-1)
                    x_joints = x_padded.reshape(B, T, V, C)
                x_in = x_joints.permute(0, 3, 1, 2).contiguous()
                out = x_in
                for block in model.blocks:
                    out = block(out)
                emb = model.gap(out).squeeze(-1).squeeze(-1)
            else:
                # Default to logits as feature vector
                emb = model(x)
            embeds.append(emb.cpu().numpy())
    return np.concatenate(embeds, axis=0)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running External S&C Benchmark on device: {device}")

    metadata_path = "data/Final_dataset_metadata.csv" if os.path.exists("data/Final_dataset_metadata.csv") else "Final_dataset_metadata.csv"
    meta_df = pd.read_csv(metadata_path)
    all_classes = sorted(meta_df['class'].unique())
    sc_class_indices = [all_classes.index(c) for c in SC_CLASS_NAMES]
    print(f"S&C Classes & Indices: {dict(zip(SC_CLASS_NAMES, sc_class_indices))}")

    # Load dataloaders
    print("Loading test datasets...")
    _, val_l_mix, test_l_mix = get_dataloaders(
        metadata_path=metadata_path, feature_method="mix", batch_size=32, seq_len=32, stride=32, val_test_stride=32,
        landmark_dir="data/landmarks", num_workers=0, in_memory=True
    )
    _, val_l_rel, test_l_rel = get_dataloaders(
        metadata_path=metadata_path, feature_method="rel_3d", batch_size=32, seq_len=32, stride=32, val_test_stride=32,
        landmark_dir="data/landmarks", num_workers=0, in_memory=True
    )
    _, val_l_bone, test_l_bone = get_dataloaders(
        metadata_path=metadata_path, feature_method="bone_3d", batch_size=32, seq_len=32, stride=32, val_test_stride=32,
        landmark_dir="data/landmarks", num_workers=0, in_memory=True
    )
    _, val_l_jm, test_l_jm = get_dataloaders(
        metadata_path=metadata_path, feature_method="joint_motion_3d", batch_size=32, seq_len=32, stride=32, val_test_stride=32,
        landmark_dir="data/landmarks", num_workers=0, in_memory=True
    )
    _, val_l_bm, test_l_bm = get_dataloaders(
        metadata_path=metadata_path, feature_method="bone_motion_3d", batch_size=32, seq_len=32, stride=32, val_test_stride=32,
        landmark_dir="data/landmarks", num_workers=0, in_memory=True
    )

    y_test_true = np.array(test_l_mix.dataset.labels)
    test_vids = np.array(test_l_mix.dataset.video_ids)
    y_val_true = np.array(val_l_mix.dataset.labels)

    # Filter indices for 4 S&C classes
    sc_mask_win = np.isin(y_test_true, sc_class_indices)
    sc_win_indices = np.where(sc_mask_win)[0]
    sc_vids = np.unique(test_vids[sc_mask_win])
    print(f"Total S&C windows: {len(sc_win_indices)}, S&C unique videos: {len(sc_vids)}")

    # Load models
    models = {
        "ST-GCN (Rel 3D)": (load_model("STGCN", "rel_3d", "checkpoints/best_STGCN_T3.2_rel_3d.pt", device), test_l_rel, val_l_rel),
        "Transformer (Mix)": (load_model("Transformer", "mix", "checkpoints/best_Transformer_T2.2_mix.pt", device), test_l_mix, val_l_mix),
        "AAGCN (Bone 3D)": (load_model("AAGCN", "bone_3d", "checkpoints/best_AAGCN_T4.2_bone_3d.pt", device), test_l_bone, val_l_bone),
        "AAGCN (Rel 3D)": (load_model("AAGCN", "rel_3d", "checkpoints/best_AAGCN_T4.3_rel_3d.pt", device), test_l_rel, val_l_rel),
        "AAGCN (Joint Mot)": (load_model("AAGCN", "joint_motion_3d", "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt", device), test_l_jm, val_l_jm),
        "AAGCN (Bone Mot)": (load_model("AAGCN", "bone_motion_3d", "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt", device), test_l_bm, val_l_bm),
    }

    # Predict test probabilities
    test_probs = {}
    val_probs = {}
    for name, (m, tl, vl) in models.items():
        print(f"Predicting {name}...")
        tp, _ = predict_model(m, tl, device)
        vp, _ = predict_model(m, vl, device)
        test_probs[name] = tp
        val_probs[name] = vp

    # Ensembles
    # SkelGym-Lite
    lite_ens = WeightedSoftVotingEnsemble()
    lite_ens.fit_window([val_probs["Transformer (Mix)"], val_probs["AAGCN (Bone 3D)"]], y_val_true)
    test_probs["SkelGym-Lite"] = lite_ens.weights[0] * test_probs["Transformer (Mix)"] + lite_ens.weights[1] * test_probs["AAGCN (Bone 3D)"]

    # SkelGym-Full
    full_ens = WeightedSoftVotingEnsemble()
    five_val = [val_probs["Transformer (Mix)"], val_probs["AAGCN (Bone 3D)"], val_probs["AAGCN (Rel 3D)"], val_probs["AAGCN (Joint Mot)"], val_probs["AAGCN (Bone Mot)"]]
    five_test = [test_probs["Transformer (Mix)"], test_probs["AAGCN (Bone 3D)"], test_probs["AAGCN (Rel 3D)"], test_probs["AAGCN (Joint Mot)"], test_probs["AAGCN (Bone Mot)"]]
    full_ens.fit_window(five_val, y_val_true)
    test_probs["SkelGym-Full"] = sum(w * p for w, p in zip(full_ens.weights, five_test))

    # =========================================================================
    # 1. Closed-Set vs Open-Set Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("TABLE 8: EXTERNAL S&C BENCHMARK (DEYZEL ET AL. PROTOCOL)")
    print("="*80)
    print(f"{'Model':<25} | {'Open Win':<10} | {'Open Vid':<10} | {'Closed Win':<10} | {'Closed Vid':<10} | {'Closed F1':<10}")
    print("-" * 85)

    eval_models = ["ST-GCN (Rel 3D)", "Transformer (Mix)", "AAGCN (Bone 3D)", "SkelGym-Lite", "SkelGym-Full"]
    
    # Map global class index to closed set index (0..3)
    global_to_sc = {idx: i for i, idx in enumerate(sc_class_indices)}
    y_true_sc_win = np.array([global_to_sc[y] for y in y_test_true[sc_win_indices]])
    
    # Video level true labels for S&C videos
    vid_to_true = {}
    for v, y in zip(test_vids, y_test_true):
        if v in sc_vids:
            vid_to_true[v] = global_to_sc[y]

    for name in eval_models:
        prob = test_probs[name]
        
        # Open-set evaluation: argmax over all 22 classes, evaluate only on S&C ground truth samples
        pred_win_open = np.argmax(prob[sc_win_indices], axis=1)
        open_win_acc = np.mean(pred_win_open == y_test_true[sc_win_indices]) * 100.0
        
        # Video level open-set
        vid_preds_open = {}
        for v in sc_vids:
            mask = (test_vids == v)
            v_prob = np.mean(prob[mask], axis=0)
            vid_preds_open[v] = np.argmax(v_prob)
        open_vid_acc = np.mean([vid_preds_open[v] == y_test_true[test_vids == v][0] for v in sc_vids]) * 100.0
        
        # Closed-set evaluation: restrict columns to sc_class_indices and re-normalize
        prob_sc = prob[:, sc_class_indices]
        prob_sc_norm = prob_sc / np.sum(prob_sc, axis=1, keepdims=True)
        
        pred_win_closed = np.argmax(prob_sc_norm[sc_win_indices], axis=1)
        closed_win_acc = accuracy_score(y_true_sc_win, pred_win_closed) * 100.0
        
        # Video level closed-set
        vid_preds_closed = {}
        for v in sc_vids:
            mask = (test_vids == v)
            v_prob = np.mean(prob_sc_norm[mask], axis=0)
            vid_preds_closed[v] = np.argmax(v_prob)
        y_true_v = [vid_to_true[v] for v in sc_vids]
        y_pred_v = [vid_preds_closed[v] for v in sc_vids]
        closed_vid_acc = accuracy_score(y_true_v, y_pred_v) * 100.0
        closed_vid_f1 = f1_score(y_true_v, y_pred_v, average="macro")
        
        print(f"{name:<25} | {open_win_acc:>9.2f}% | {open_vid_acc:>9.2f}% | {closed_win_acc:>9.2f}% | {closed_vid_acc:>9.2f}% | {closed_vid_f1:>10.4f}")

    # =========================================================================
    # 2. Squat vs Deadlift Ambiguity Breakdown (The Deyzel Dilemma)
    # =========================================================================
    print("\n" + "="*80)
    print("SQUAT VS DEADLIFT AMBIGUITY (DEYZEL DILEMMA, N=25 VIDEOS)")
    print("="*80)
    
    squat_idx = all_classes.index("squat")
    deadlift_idx = all_classes.index("deadlift")
    sq_dl_mask = np.isin(y_test_true, [squat_idx, deadlift_idx])
    sq_dl_vids = np.unique(test_vids[sq_dl_mask])
    
    print(f"{'Model':<25} | {'Squat Win Rec':<13} | {'DL Win Rec':<11} | {'Squat Vid Rec':<13} | {'DL Vid Rec':<11}")
    print("-" * 80)
    
    for name in ["ST-GCN (Rel 3D)", "Transformer (Mix)", "AAGCN (Bone 3D)", "SkelGym-Full"]:
        prob = test_probs[name]
        preds_win = np.argmax(prob, axis=1)
        
        sq_win_mask = (y_test_true == squat_idx)
        dl_win_mask = (y_test_true == deadlift_idx)
        sq_win_rec = np.mean(preds_win[sq_win_mask] == squat_idx) * 100.0
        dl_win_rec = np.mean(preds_win[dl_win_mask] == deadlift_idx) * 100.0
        
        # Video level
        sq_vid_correct, sq_vid_total = 0, 0
        dl_vid_correct, dl_vid_total = 0, 0
        for v in sq_dl_vids:
            mask = (test_vids == v)
            true_cls = y_test_true[mask][0]
            v_pred = np.argmax(np.mean(prob[mask], axis=0))
            if true_cls == squat_idx:
                sq_vid_total += 1
                if v_pred == squat_idx:
                    sq_vid_correct += 1
            elif true_cls == deadlift_idx:
                dl_vid_total += 1
                if v_pred == deadlift_idx:
                    dl_vid_correct += 1
        sq_vid_rec = (sq_vid_correct / sq_vid_total) * 100.0
        dl_vid_rec = (dl_vid_correct / dl_vid_total) * 100.0
        
        print(f"{name:<25} | {sq_win_rec:>12.1f}% | {dl_win_rec:>10.1f}% | {sq_vid_rec:>12.1f}% | {dl_vid_rec:>10.1f}%")

    # =========================================================================
    # 3. One-Shot Transfer Learning (Deyzel One-Shot Protocol)
    # =========================================================================
    print("\n" + "="*80)
    print("ONE-SHOT TRANSFER LEARNING SIMULATION (100 TRIALS, K=1)")
    print("="*80)
    
    # Extract temporal mean embeddings per video for S&C videos
    emb_models = {
        "Transformer (Mix)": (models["Transformer (Mix)"][0], test_l_mix, "Transformer"),
        "AAGCN (Bone 3D)": (models["AAGCN (Bone 3D)"][0], test_l_bone, "AAGCN")
    }
    
    video_embs = {}
    for m_name, (m, ldr, m_type) in emb_models.items():
        print(f"Extracting embeddings for {m_name}...")
        win_embs = get_embeddings(m, ldr, device, m_type)
        v_embs = {}
        for v in sc_vids:
            mask = (test_vids == v)
            # Normalize embedding
            mean_emb = np.mean(win_embs[mask], axis=0)
            norm = np.linalg.norm(mean_emb)
            v_embs[v] = mean_emb / (norm + 1e-8)
        video_embs[m_name] = v_embs

    # Combine for SkelGym-Full (concatenating normalized embeddings)
    v_embs_full = {}
    for v in sc_vids:
        c_emb = np.concatenate([video_embs["Transformer (Mix)"][v], video_embs["AAGCN (Bone 3D)"][v]])
        c_emb = c_emb / (np.linalg.norm(c_emb) + 1e-8)
        v_embs_full[v] = c_emb
    video_embs["SkelGym-Full"] = v_embs_full

    # Group video ids by class
    vids_by_class = {c: [] for c in range(4)}
    for v in sc_vids:
        vids_by_class[vid_to_true[v]].append(v)

    np.random.seed(42)
    N_TRIALS = 100
    
    for m_name in ["Transformer (Mix)", "AAGCN (Bone 3D)", "SkelGym-Full"]:
        trial_accs = []
        embs = video_embs[m_name]
        for _ in range(N_TRIALS):
            # Sample 1 random support video per class
            support_vids = [np.random.choice(vids_by_class[c]) for c in range(4)]
            query_vids = [v for v in sc_vids if v not in support_vids]
            
            support_vecs = np.stack([embs[v] for v in support_vids])  # (4, D)
            query_vecs = np.stack([embs[v] for v in query_vids])      # (50, D)
            query_labels = np.array([vid_to_true[v] for v in query_vids])
            
            # Cosine similarity matrix (50, 4)
            sim = np.dot(query_vecs, support_vecs.T)
            preds = np.argmax(sim, axis=1)
            acc = np.mean(preds == query_labels) * 100.0
            trial_accs.append(acc)
            
        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)
        ci_low, ci_high = np.percentile(trial_accs, 2.5), np.percentile(trial_accs, 97.5)
        print(f"{m_name:<25}: Mean = {mean_acc:.2f}% ± {std_acc:.2f}% | 95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

    print("\nExternal benchmark verification complete!")

if __name__ == "__main__":
    main()

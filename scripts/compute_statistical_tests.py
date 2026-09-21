import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.models.ensemble import WeightedSoftVotingEnsemble, aggregate_video_level_predictions

def get_predictions(model_type, feat_type, ckpt_path, device, val_l, test_l):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    m = build_model(model_type, feat_type, num_classes=22)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    trainer = Trainer(model=m, device=device)
    y_vt, y_vp, y_vprob = trainer.predict(val_l)
    y_tt, y_tp, y_tprob = trainer.predict(test_l)
    return y_vprob, y_tprob, y_vt, y_tt, y_vp, y_tp

def mcnemar_test(y_true, y_pred1, y_pred2):
    c01 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    c10 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    b, c = c10, c01
    chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
    p_val = stats.chi2.sf(chi2, 1)
    return b, c, chi2, p_val

def bootstrap_window(y_true, y_pred, B=1000, seed=42):
    np.random.seed(seed)
    N = len(y_true)
    accs, f1s = [], []
    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(np.mean(yt == yp) * 100.0)
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return np.mean(accs), (np.percentile(accs, 2.5), np.percentile(accs, 97.5)), np.mean(f1s), (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))

def bootstrap_video(y_true_vid, y_pred_vid, B=1000, seed=42):
    np.random.seed(seed)
    N = len(y_true_vid)
    accs, f1s = [], []
    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        yt, yp = y_true_vid[idx], y_pred_vid[idx]
        accs.append(np.mean(yt == yp) * 100.0)
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return np.mean(accs), (np.percentile(accs, 2.5), np.percentile(accs, 97.5)), np.mean(f1s), (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running on device: {device}")
    
    metadata_path = "data/Final_dataset_metadata.csv" if os.path.exists("data/Final_dataset_metadata.csv") else "Final_dataset_metadata.csv"
    landmark_dir = "data/landmarks"
    
    loaders = {}
    for feat in ["mix", "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]:
        _, val_l, test_l = get_dataloaders(
            metadata_path=metadata_path,
            feature_method=feat,
            batch_size=32,
            seq_len=32,
            stride=32,
            val_test_stride=32,
            landmark_dir=landmark_dir,
            num_workers=0,
            in_memory=True
        )
        loaders[feat] = (val_l, test_l)
    
    val_l_mix, test_l_mix = loaders["mix"]
    y_test_true = np.array(test_l_mix.dataset.labels)
    test_vids = test_l_mix.dataset.video_ids
    val_vids = val_l_mix.dataset.video_ids
    y_val_true = np.array(val_l_mix.dataset.labels)

    # 1. Sequence Baselines (LSTM, BiLSTM)
    print("Loading Baseline LSTM & BiLSTM...")
    vprob_lstm, tprob_lstm, _, _, _, tp_lstm = get_predictions(
        "LSTM", "mix", "checkpoints/best_LSTM_T1.9_mix.pt", device, loaders["mix"][0], loaders["mix"][1]
    )
    vprob_bilstm, tprob_bilstm, _, _, _, tp_bilstm = get_predictions(
        "BiLSTM", "mix", "checkpoints/best_BiLSTM_T1.18_mix.pt", device, loaders["mix"][0], loaders["mix"][1]
    )

    # 2. Clean & Aug Transformer Mix
    print("Loading Clean & Aug Transformer...")
    vprob_clean_trans, tprob_clean_trans, _, _, _, tp_clean_trans = get_predictions(
        "Transformer", "mix", "checkpoints/best_Transformer_T1.27_mix.pt", device, loaders["mix"][0], loaders["mix"][1]
    )
    vprob_aug_trans, tprob_aug_trans, _, _, _, tp_aug_trans = get_predictions(
        "Transformer", "mix", "checkpoints/best_Transformer_T2.2_mix.pt", device, loaders["mix"][0], loaders["mix"][1]
    )

    # 3. ST-GCN Baseline Rel 3D
    print("Loading ST-GCN Baseline...")
    vprob_stgcn, tprob_stgcn, _, _, _, tp_stgcn = get_predictions(
        "STGCN", "rel_3d", "checkpoints/best_STGCN_T3.2_rel_3d.pt", device, loaders["rel_3d"][0], loaders["rel_3d"][1]
    )

    # 4. AAGCN Streams
    print("Loading AAGCN streams...")
    vprob_bone, tprob_bone, _, _, _, tp_bone = get_predictions(
        "AAGCN", "bone_3d", "checkpoints/best_AAGCN_T4.2_bone_3d.pt", device, loaders["bone_3d"][0], loaders["bone_3d"][1]
    )
    vprob_joint, tprob_joint, _, _, _, tp_joint = get_predictions(
        "AAGCN", "rel_3d", "checkpoints/best_AAGCN_T4.3_rel_3d.pt", device, loaders["rel_3d"][0], loaders["rel_3d"][1]
    )
    vprob_jmot, tprob_jmot, _, _, _, tp_jmot = get_predictions(
        "AAGCN", "joint_motion_3d", "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt", device, loaders["joint_motion_3d"][0], loaders["joint_motion_3d"][1]
    )
    vprob_bmot, tprob_bmot, _, _, _, tp_bmot = get_predictions(
        "AAGCN", "bone_motion_3d", "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt", device, loaders["bone_motion_3d"][0], loaders["bone_motion_3d"][1]
    )

    # Four-Stream AAGCN
    four_stream_val_probs = [vprob_joint, vprob_bone, vprob_jmot, vprob_bmot]
    four_stream_test_probs = [tprob_joint, tprob_bone, tprob_jmot, tprob_bmot]
    four_stream_ens = WeightedSoftVotingEnsemble()
    four_stream_ens.fit_window(four_stream_val_probs, y_val_true)
    tp_4stream_win = four_stream_ens.predict_window(four_stream_test_probs)
    tprob_4stream = np.mean(four_stream_test_probs, axis=0)

    # SkelGym-Lite (Transformer Mix + AAGCN Bone)
    lite_val_probs = [vprob_aug_trans, vprob_bone]
    lite_test_probs = [tprob_aug_trans, tprob_bone]
    skelgym_lite = WeightedSoftVotingEnsemble()
    skelgym_lite.fit_window(lite_val_probs, y_val_true)
    skelgym_lite.fit_video(lite_val_probs, y_val_true, val_vids)
    tp_skelgym_lite_win = skelgym_lite.predict_window(lite_test_probs)
    _, tp_skelgym_lite_vid, _, _ = skelgym_lite.predict_video(lite_test_probs, y_test_true, test_vids)

    # SkelGym-Full 5-Stream Ensemble
    five_stream_val = [vprob_aug_trans, vprob_joint, vprob_bone, vprob_jmot, vprob_bmot]
    five_stream_test = [tprob_aug_trans, tprob_joint, tprob_bone, tprob_jmot, tprob_bmot]
    skelgym_full = WeightedSoftVotingEnsemble()
    skelgym_full.fit_window(five_stream_val, y_val_true)
    skelgym_full.fit_video(five_stream_val, y_val_true, val_vids)
    tp_skelgym_win = skelgym_full.predict_window(five_stream_test)
    y_test_vid_t, tp_skelgym_vid, tprob_skelgym_vid, skelgym_vid_metrics = skelgym_full.predict_video(five_stream_test, y_test_true, test_vids)

    # Video level predictions for individual models
    _, vp_lstm, _, _ = aggregate_video_level_predictions(tprob_lstm, y_test_true, test_vids)
    _, vp_bilstm, _, _ = aggregate_video_level_predictions(tprob_bilstm, y_test_true, test_vids)
    _, vp_clean_trans, _, _ = aggregate_video_level_predictions(tprob_clean_trans, y_test_true, test_vids)
    _, vp_aug_trans, _, _ = aggregate_video_level_predictions(tprob_aug_trans, y_test_true, test_vids)
    _, vp_stgcn, _, _ = aggregate_video_level_predictions(tprob_stgcn, y_test_true, test_vids)
    _, vp_bone, _, _ = aggregate_video_level_predictions(tprob_bone, y_test_true, test_vids)
    _, vp_4stream, _, _ = aggregate_video_level_predictions(tprob_4stream, y_test_true, test_vids)

    print("\n" + "="*80)
    print("MCNEMAR TEST (WINDOW LEVEL, N=2,743)")
    print("="*80)
    
    comparisons_win = [
        ("Clean Transformer vs Aug Transformer", tp_clean_trans, tp_aug_trans),
        ("ST-GCN vs Four-Stream AAGCN", tp_stgcn, tp_4stream_win),
        ("Transformer Mix (Aug) vs SkelGym-Full", tp_aug_trans, tp_skelgym_win),
        ("AAGCN Bone (Aug) vs SkelGym-Full", tp_bone, tp_skelgym_win),
        ("Four-Stream AAGCN vs SkelGym-Full", tp_4stream_win, tp_skelgym_win)
    ]
    
    for name, p1, p2 in comparisons_win:
        b, c, chi2, p_val = mcnemar_test(y_test_true, p1, p2)
        acc1 = np.mean(p1 == y_test_true) * 100
        acc2 = np.mean(p2 == y_test_true) * 100
        diff = acc2 - acc1
        odds_ratio = c / b if b > 0 else np.nan
        print(f"[{name}]")
        print(f"  Acc1: {acc1:.2f}% -> Acc2: {acc2:.2f}% (Delta = {diff:+.2f}%)")
        print(f"  Discordant: b={b} (1 right, 2 wrong), c={c} (1 wrong, 2 right)")
        print(f"  McNemar Chi2 = {chi2:.3f}, p-value = {p_val:.4e}, Odds Ratio = {odds_ratio:.2f}")

    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TEST & PAIRED T-TEST (VIDEO LEVEL, N=233)")
    print("="*80)

    vid_acc_clean = (vp_clean_trans == y_test_vid_t).astype(float)
    vid_acc_aug = (vp_aug_trans == y_test_vid_t).astype(float)
    vid_acc_stgcn = (vp_stgcn == y_test_vid_t).astype(float)
    vid_acc_bone = (vp_bone == y_test_vid_t).astype(float)
    vid_acc_4stream = (vp_4stream == y_test_vid_t).astype(float)
    vid_acc_full = (tp_skelgym_vid == y_test_vid_t).astype(float)

    comparisons_vid = [
        ("Clean Transformer vs Aug Transformer", vid_acc_clean, vid_acc_aug),
        ("ST-GCN vs Four-Stream AAGCN", vid_acc_stgcn, vid_acc_4stream),
        ("Transformer Mix (Aug) vs SkelGym-Full", vid_acc_aug, vid_acc_full),
        ("AAGCN Bone (Aug) vs SkelGym-Full", vid_acc_bone, vid_acc_full),
        ("Four-Stream AAGCN vs SkelGym-Full", vid_acc_4stream, vid_acc_full)
    ]

    for name, v1, v2 in comparisons_vid:
        diff = v2 - v1
        mean_diff = np.mean(diff) * 100
        n_diff = np.sum(diff != 0)
        w_res = stats.wilcoxon(v2, v1, zero_method="pratt")
        t_res = stats.ttest_rel(v2, v1)
        s_d = np.std(diff, ddof=1)
        cohens_d = np.mean(diff) / s_d if s_d > 0 else 0.0
        z_stat = (w_res.statistic - n_diff * (n_diff + 1) / 4) / np.sqrt(n_diff * (n_diff + 1) * (2 * n_diff + 1) / 24) if n_diff > 0 else 0
        r_biserial = abs(z_stat) / np.sqrt(len(v1)) if len(v1) > 0 else 0

        print(f"[{name}]")
        print(f"  Vid Acc 1: {np.mean(v1)*100:.2f}% -> Vid Acc 2: {np.mean(v2)*100:.2f}% (Delta = {mean_diff:+.2f}%)")
        print(f"  Non-zero paired differences: {n_diff}/{len(v1)} videos")
        print(f"  Wilcoxon W = {w_res.statistic:.1f}, p-value = {w_res.pvalue:.4e}")
        print(f"  Paired t-statistic = {t_res.statistic:.3f}, p-value = {t_res.pvalue:.4e}")
        print(f"  Cohen's d = {cohens_d:.3f}, Effect Size r = {r_biserial:.3f}")

    print("\n" + "="*80)
    print("NON-PARAMETRIC BOOTSTRAP RESAMPLING (B=1,000 RESAMPLES)")
    print("="*80)
    
    models_bootstrap = [
        ("LSTM (Mix 117-d)", tp_lstm, vp_lstm),
        ("BiLSTM (Mix 117-d)", tp_bilstm, vp_bilstm),
        ("ST-GCN (Rel 3D)", tp_stgcn, vp_stgcn),
        ("Transformer (Mix 117-d)", tp_aug_trans, vp_aug_trans),
        ("AAGCN (Bone 3D)", tp_bone, vp_bone),
        ("SkelGym-Lite", tp_skelgym_lite_win, tp_skelgym_lite_vid),
        ("SkelGym-Full", tp_skelgym_win, tp_skelgym_vid)
    ]
    
    bootstrap_rows = []
    print(f"{'Architecture':<26} | {'Window Acc (95% CI)':<26} | {'Window F1 (95% CI)':<24} | {'Video Acc (95% CI)':<26} | {'Video F1 (95% CI)':<24}")
    print("-" * 130)
    
    for name, p_win, p_vid in models_bootstrap:
        w_acc_mean, w_acc_ci, w_f1_mean, w_f1_ci = bootstrap_window(y_test_true, p_win, B=1000, seed=42)
        v_acc_mean, v_acc_ci, v_f1_mean, v_f1_ci = bootstrap_video(y_test_vid_t, p_vid, B=1000, seed=42)
        
        w_acc_str = f"{w_acc_mean:.2f}% [{w_acc_ci[0]:.2f}%, {w_acc_ci[1]:.2f}%]"
        w_f1_str = f"{w_f1_mean:.4f} [{w_f1_ci[0]:.4f}, {w_f1_ci[1]:.4f}]"
        v_acc_str = f"{v_acc_mean:.2f}% [{v_acc_ci[0]:.2f}%, {v_acc_ci[1]:.2f}%]"
        v_f1_str = f"{v_f1_mean:.4f} [{v_f1_ci[0]:.4f}, {v_f1_ci[1]:.4f}]"
        
        print(f"{name:<26} | {w_acc_str:<26} | {w_f1_str:<24} | {v_acc_str:<26} | {v_f1_str:<24}")
        bootstrap_rows.append({
            "name": name,
            "w_acc_str": w_acc_str,
            "w_f1_str": w_f1_str,
            "v_acc_str": v_acc_str,
            "v_f1_str": v_f1_str
        })

    # Save to outputs/bootstrap_confidence_intervals.md
    out_md_path = PROJECT_ROOT / "outputs" / "bootstrap_confidence_intervals.md"
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# Bootstrap 95% Confidence Intervals (B=1,000 resamples)\n\n")
        f.write("Evaluated on held-out test partitions ($N=2,743$ temporal windows, $N=233$ action videos) under strict video-level partitioning.\n\n")
        f.write("| Architecture | Window Accuracy (95% CI) | Window Macro F1 (95% CI) | Video Accuracy (95% CI) | Video Macro F1 (95% CI) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for row in bootstrap_rows:
            bold = "**" if "SkelGym" in row["name"] else ""
            f.write(f"| {bold}{row['name']}{bold} | {row['w_acc_str']} | {row['w_f1_str']} | {row['v_acc_str']} | {row['v_f1_str']} |\n")
    print(f"\nSaved updated bootstrap confidence intervals to: {out_md_path}")

if __name__ == "__main__":
    main()

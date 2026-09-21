import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats

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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running on device: {device}")
    
    metadata_path = "data/Final_dataset_metadata.csv"
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
    y_test_true = test_l_mix.dataset.labels
    test_vids = test_l_mix.dataset.video_ids
    val_vids = val_l_mix.dataset.video_ids
    y_val_true = val_l_mix.dataset.labels

    # 1. Clean Transformer Mix (T1.27)
    print("Loading Clean Transformer...")
    vprob_clean_trans, tprob_clean_trans, _, _, _, tp_clean_trans = get_predictions(
        "Transformer", "mix", "checkpoints/best_Transformer_T1.27_mix.pt", device, loaders["mix"][0], loaders["mix"][1]
    )

    # 2. SkelGym-Aug Transformer Mix (T2.2)
    print("Loading Aug Transformer...")
    vprob_aug_trans, tprob_aug_trans, _, _, _, tp_aug_trans = get_predictions(
        "Transformer", "mix", "checkpoints/best_Transformer_T2.2_mix.pt", device, loaders["mix"][0], loaders["mix"][1]
    )

    # 3. ST-GCN Baseline Rel 3D (T3.2)
    print("Loading ST-GCN...")
    vprob_stgcn, tprob_stgcn, _, _, _, tp_stgcn = get_predictions(
        "STGCN", "rel_3d", "checkpoints/best_STGCN_T3.2_rel_3d.pt", device, loaders["rel_3d"][0], loaders["rel_3d"][1]
    )

    # 4. AAGCN Bone 3D (T4.2)
    print("Loading AAGCN Bone...")
    vprob_bone, tprob_bone, _, _, _, tp_bone = get_predictions(
        "AAGCN", "bone_3d", "checkpoints/best_AAGCN_T4.2_bone_3d.pt", device, loaders["bone_3d"][0], loaders["bone_3d"][1]
    )

    # 5. Other AAGCN streams
    print("Loading remaining AAGCN streams...")
    vprob_joint, tprob_joint, _, _, _, _ = get_predictions(
        "AAGCN", "rel_3d", "checkpoints/best_AAGCN_T4.3_rel_3d.pt", device, loaders["rel_3d"][0], loaders["rel_3d"][1]
    )
    vprob_jmot, tprob_jmot, _, _, _, _ = get_predictions(
        "AAGCN", "joint_motion_3d", "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt", device, loaders["joint_motion_3d"][0], loaders["joint_motion_3d"][1]
    )
    vprob_bmot, tprob_bmot, _, _, _, _ = get_predictions(
        "AAGCN", "bone_motion_3d", "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt", device, loaders["bone_motion_3d"][0], loaders["bone_motion_3d"][1]
    )

    # Four-Stream AAGCN
    four_stream_val_probs = [vprob_joint, vprob_bone, vprob_jmot, vprob_bmot]
    four_stream_test_probs = [tprob_joint, tprob_bone, tprob_jmot, tprob_bmot]
    four_stream_ens = WeightedSoftVotingEnsemble()
    four_stream_ens.fit_window(four_stream_val_probs, y_val_true)
    tp_4stream_win = four_stream_ens.predict_window(four_stream_test_probs)
    tprob_4stream = np.mean(four_stream_test_probs, axis=0)

    # SkelGym-Full 5-Stream Ensemble
    five_stream_val = [vprob_aug_trans, vprob_joint, vprob_bone, vprob_jmot, vprob_bmot]
    five_stream_test = [tprob_aug_trans, tprob_joint, tprob_bone, tprob_jmot, tprob_bmot]
    skelgym_full = WeightedSoftVotingEnsemble()
    skelgym_full.fit_window(five_stream_val, y_val_true)
    skelgym_full.fit_video(five_stream_val, y_val_true, val_vids)
    tp_skelgym_win = skelgym_full.predict_window(five_stream_test)
    y_test_vid_t, tp_skelgym_vid, tprob_skelgym_vid, skelgym_vid_metrics = skelgym_full.predict_video(five_stream_test, y_test_true, test_vids)

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

    # Compute video level predictions using aggregate_video_level_predictions
    _, vp_clean, _, _ = aggregate_video_level_predictions(tprob_clean_trans, y_test_true, test_vids)
    _, vp_aug, _, _ = aggregate_video_level_predictions(tprob_aug_trans, y_test_true, test_vids)
    _, vp_stgcn, _, _ = aggregate_video_level_predictions(tprob_stgcn, y_test_true, test_vids)
    _, vp_bone, _, _ = aggregate_video_level_predictions(tprob_bone, y_test_true, test_vids)
    _, vp_4stream, _, _ = aggregate_video_level_predictions(tprob_4stream, y_test_true, test_vids)

    vid_acc_clean = (vp_clean == y_test_vid_t).astype(float)
    vid_acc_aug = (vp_aug == y_test_vid_t).astype(float)
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
    print("BIOMECHANICAL ERROR TAXONOMY BREAKDOWN (SkelGym-Full on Test Set)")
    print("="*80)
    
    import pandas as pd
    df_meta = pd.read_csv(metadata_path)
    class_names = sorted(df_meta['class'].unique())

    errors_win = np.where(tp_skelgym_win != y_test_true)[0]
    total_win_errors = len(errors_win)
    print(f"Total Window Errors: {total_win_errors} / {len(y_test_true)} ({total_win_errors/len(y_test_true)*100:.2f}% error rate)")

    # Category 1: Upper-Limb Radioulnar Pronation/Supination Optical Ambiguity (Hammer Curl <-> Barbell Biceps Curl)
    idx_cat1 = [i for i in errors_win if (
        (class_names[y_test_true[i]] == "hammer curl" and class_names[tp_skelgym_win[i]] == "barbell biceps curl") or
        (class_names[y_test_true[i]] == "barbell biceps curl" and class_names[tp_skelgym_win[i]] == "hammer curl")
    )]

    # Category 2: Posterior-Chain Pelvic/Knee Flexion Execution Overlap (Deadlift <-> Romanian Deadlift)
    idx_cat2 = [i for i in errors_win if (
        (class_names[y_test_true[i]] == "deadlift" and class_names[tp_skelgym_win[i]] == "romanian deadlift") or
        (class_names[y_test_true[i]] == "romanian deadlift" and class_names[tp_skelgym_win[i]] == "deadlift")
    )]

    # Category 3: Bench Press Torso Inclination Foreshortening (Flat vs Incline vs Decline)
    bench_classes = {"bench press", "incline bench press", "decline bench press"}
    idx_cat3 = [i for i in errors_win if (
        class_names[y_test_true[i]] in bench_classes and class_names[tp_skelgym_win[i]] in bench_classes and
        y_test_true[i] != tp_skelgym_win[i]
    )]

    # Category 4: Overhead & Pull Kinetic Couplings
    overhead_classes = {"lat pulldown", "pull Up", "shoulder press", "t bar row"}
    idx_cat4 = [i for i in errors_win if (
        class_names[y_test_true[i]] in overhead_classes and class_names[tp_skelgym_win[i]] in overhead_classes and
        y_test_true[i] != tp_skelgym_win[i]
    )]

    cat1_cnt = len(idx_cat1)
    cat2_cnt = len(idx_cat2)
    cat3_cnt = len(idx_cat3)
    cat4_cnt = len(idx_cat4)
    cat_other_cnt = total_win_errors - (cat1_cnt + cat2_cnt + cat3_cnt + cat4_cnt)

    print(f"Category 1 (Radioulnar Pronation/Supination Optical Ambiguity): {cat1_cnt} errors ({cat1_cnt/total_win_errors*100:.2f}%)")
    print(f"Category 2 (Posterior-Chain Pelvic/Knee Flexion Form Overlap): {cat2_cnt} errors ({cat2_cnt/total_win_errors*100:.2f}%)")
    print(f"Category 3 (Pressing Plane Inclination Foreshortening): {cat3_cnt} errors ({cat3_cnt/total_win_errors*100:.2f}%)")
    print(f"Category 4 (Overhead & Back Kinetic Chain Couplings): {cat4_cnt} errors ({cat4_cnt/total_win_errors*100:.2f}%)")
    print(f"Category 5 (Residual Minor Errors across other classes): {cat_other_cnt} errors ({cat_other_cnt/total_win_errors*100:.2f}%)")

    # Analyze Video-Level Errors
    errors_vid = np.where(tp_skelgym_vid != y_test_vid_t)[0]
    total_vid_errors = len(errors_vid)
    print(f"\nTotal Video Errors: {total_vid_errors} / {len(y_test_vid_t)} ({total_vid_errors/len(y_test_vid_t)*100:.2f}% error rate)")

    vid_cat1 = 0
    vid_cat2 = 0
    vid_cat3 = 0
    vid_cat4 = 0
    vid_other = 0

    for i in errors_vid:
        true_c = class_names[y_test_vid_t[i]]
        pred_c = class_names[tp_skelgym_vid[i]]
        if (true_c == "hammer curl" and pred_c == "barbell biceps curl") or (true_c == "barbell biceps curl" and pred_c == "hammer curl"):
            vid_cat1 += 1
        elif (true_c == "deadlift" and pred_c == "romanian deadlift") or (true_c == "romanian deadlift" and pred_c == "deadlift"):
            vid_cat2 += 1
        elif true_c in bench_classes and pred_c in bench_classes:
            vid_cat3 += 1
        elif true_c in overhead_classes and pred_c in overhead_classes:
            vid_cat4 += 1
        else:
            vid_other += 1

    print(f"Video Cat 1 (Radioulnar Ambiguity): {vid_cat1} errors ({vid_cat1/total_vid_errors*100:.2f}%)")
    print(f"Video Cat 2 (Posterior-Chain Form Overlap): {vid_cat2} errors ({vid_cat2/total_vid_errors*100:.2f}%)")
    print(f"Video Cat 3 (Pressing Plane Inclination): {vid_cat3} errors ({vid_cat3/total_vid_errors*100:.2f}%)")
    print(f"Video Cat 4 (Overhead / Back Couplings): {vid_cat4} errors ({vid_cat4/total_vid_errors*100:.2f}%)")
    print(f"Video Other (Residual Minor Dispersed): {vid_other} errors ({vid_other/total_vid_errors*100:.2f}%)")

if __name__ == "__main__":
    main()

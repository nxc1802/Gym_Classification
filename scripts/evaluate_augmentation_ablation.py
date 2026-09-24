import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.models.ensemble import aggregate_video_level_predictions

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    metadata_path = "data/Final_dataset_metadata.csv"
    landmark_dir = "data/landmarks"

    df_meta = pd.read_csv(metadata_path)
    label_map = sorted(df_meta["class"].unique())
    print(f"Classes ({len(label_map)}): {label_map}")

    # Models to compare
    models = {
        "Trans_NoAug": {"ckpt": "checkpoints/best_Transformer_T1.27_mix.pt", "model": "Transformer", "feat": "mix"},
        "Trans_WithAug": {"ckpt": "checkpoints/best_Transformer_T2.2_mix.pt", "model": "Transformer", "feat": "mix"},
        "Bone_NoAug": {"ckpt": "checkpoints/best_AAGCN_T4.1_bone_3d.pt", "model": "AAGCN", "feat": "bone_3d"},
        "Bone_WithAug": {"ckpt": "checkpoints/best_AAGCN_T4.2_bone_3d.pt", "model": "AAGCN", "feat": "bone_3d"},
    }

    results = {}

    for name, cfg in models.items():
        print(f"\n--- Evaluating {name} ---")
        ckpt_path = Path(cfg["ckpt"])
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        m = build_model(cfg["model"], cfg["feat"], num_classes=22)
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()

        _, _, test_l = get_dataloaders(
            metadata_path=metadata_path,
            feature_method=cfg["feat"],
            batch_size=32,
            seq_len=32,
            stride=32,
            val_test_stride=32,
            landmark_dir=landmark_dir,
            num_workers=0,
            in_memory=True
        )

        test_video_ids = test_l.dataset.video_ids
        trainer = Trainer(model=m, device=device)
        y_true, y_pred, y_prob = trainer.predict(test_l)

        # Window metrics
        rep_win = classification_report(y_true, y_pred, target_names=label_map, output_dict=True, zero_division=0)
        
        # Video metrics
        y_vid_t, y_vid_p, y_vid_prob, vid_metrics = aggregate_video_level_predictions(y_prob, y_true, test_video_ids)
        rep_vid = classification_report(y_vid_t, y_vid_p, target_names=label_map, output_dict=True, zero_division=0)

        results[name] = {
            "win_acc": np.mean(y_true == y_pred),
            "vid_acc": vid_metrics["accuracy"],
            "rep_win": rep_win,
            "rep_vid": rep_vid
        }

    # Summary table per class
    print("\n" + "=" * 120)
    print("DETAILED PER-CLASS ANALYSIS: NO AUGMENTATION vs WITH SKELGYM-AUG")
    print("=" * 120)

    rows = []
    for cls in label_map:
        t_no_r = results["Trans_NoAug"]["rep_win"][cls]["recall"]
        t_aug_r = results["Trans_WithAug"]["rep_win"][cls]["recall"]
        t_diff_r = t_aug_r - t_no_r

        t_no_f1 = results["Trans_NoAug"]["rep_win"][cls]["f1-score"]
        t_aug_f1 = results["Trans_WithAug"]["rep_win"][cls]["f1-score"]
        t_diff_f1 = t_aug_f1 - t_no_f1

        # Video recall
        t_no_vid_r = results["Trans_NoAug"]["rep_vid"][cls]["recall"]
        t_aug_vid_r = results["Trans_WithAug"]["rep_vid"][cls]["recall"]
        t_vid_diff_r = t_aug_vid_r - t_no_vid_r

        # AAGCN Bone comparison
        b_no_r = results["Bone_NoAug"]["rep_win"][cls]["recall"]
        b_aug_r = results["Bone_WithAug"]["rep_win"][cls]["recall"]
        b_diff_r = b_aug_r - b_no_r

        supp = results["Trans_NoAug"]["rep_win"][cls]["support"]
        vid_supp = results["Trans_NoAug"]["rep_vid"][cls]["support"]

        rows.append({
            "Class": cls,
            "Supp(Win/Vid)": f"{supp}/{vid_supp}",
            "Trans_NoAug_Rec": t_no_r * 100,
            "Trans_Aug_Rec": t_aug_r * 100,
            "Trans_Delta_Rec": t_diff_r * 100,
            "Trans_NoAug_F1": t_no_f1,
            "Trans_Aug_F1": t_aug_f1,
            "Trans_Delta_F1": t_diff_f1,
            "Trans_Vid_NoAug_Rec": t_no_vid_r * 100,
            "Trans_Vid_Aug_Rec": t_aug_vid_r * 100,
            "Trans_Vid_Delta_Rec": t_vid_diff_r * 100,
            "Bone_NoAug_Rec": b_no_r * 100,
            "Bone_Aug_Rec": b_aug_r * 100,
            "Bone_Delta_Rec": b_diff_r * 100,
        })

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))

    # Save to CSV for inspection
    df_res.to_csv("outputs/per_class_augmentation_ablation.csv", index=False)
    print("\nSaved detailed table to outputs/per_class_augmentation_ablation.csv")

if __name__ == "__main__":
    main()

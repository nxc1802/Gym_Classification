"""
Data Description and Landmark Extraction Reporting Module.
Validates dataset statistics against paper Table 1, performs MediaPipe pose extraction,
and packages landmark CSVs for cloud/server deployment without heavy videos.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

from src.constants import ACTIONS
from src.data.extractor import extract_landmarks_from_video

def generate_dataset_report(
    metadata_path: str = "Final_dataset_metadata.csv",
    output_report_dir: str = "outputs/dataset_report"
) -> Dict[str, Any]:
    """
    Analyzes metadata CSV to verify Table 1 counts, resolutions, and frame distributions.
    Saves Markdown report and LaTeX code for Table 1.
    """
    out_dir = Path(output_report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)

    # 1. Video counts per class by split
    video_counts = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    for col in ["train", "val", "test"]:
        if col not in video_counts.columns:
            video_counts[col] = 0

    # Ensure all 22 classes present in order
    video_counts = video_counts.reindex(ACTIONS, fill_value=0)

    # 2. Resolution breakdown
    res_counts = df["resolution"].value_counts().to_dict()

    # 3. Frame statistics
    num_frames = df["num_frames"]
    frame_stats = {
        "total_videos": len(df),
        "total_frames": int(num_frames.sum()),
        "min_frames": int(num_frames.min()),
        "max_frames": int(num_frames.max()),
        "mean_frames": float(num_frames.mean()),
        "median_frames": float(num_frames.median())
    }

    # Generate Markdown Report
    md_lines = [
        "# Dataset Description & Paper Verification Report",
        "",
        "## 1. Summary Statistics",
        f"- **Total Videos**: {frame_stats['total_videos']}",
        f"- **Total Frame Count**: {frame_stats['total_frames']:,}",
        f"- **Frame Range**: {frame_stats['min_frames']} to {frame_stats['max_frames']} (Mean: {frame_stats['mean_frames']:.1f}, Median: {frame_stats['median_frames']:.1f})",
        "",
        "## 2. Video Resolutions Distribution",
    ]
    for res, count in res_counts.items():
        md_lines.append(f"- `{res}`: {count} videos ({count / len(df) * 100:.1f}%)")

    md_lines.extend([
        "",
        "## 3. Video Counts per Exercise Class by Split (Table 1 Verification)",
        "| Exercise Class | Train Videos | Val Videos | Test Videos | Total Videos |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])

    for act in ACTIONS:
        tr = video_counts.loc[act, "train"]
        va = video_counts.loc[act, "val"]
        te = video_counts.loc[act, "test"]
        tot = tr + va + te
        md_lines.append(f"| {act} | {tr} | {va} | {te} | {tot} |")

    tr_tot = video_counts["train"].sum()
    va_tot = video_counts["val"].sum()
    te_tot = video_counts["test"].sum()
    md_lines.append(f"| **TOTAL** | **{tr_tot}** | **{va_tot}** | **{te_tot}** | **{tr_tot + va_tot + te_tot}** |")

    # Generate LaTeX Table 1 code
    tex_lines = [
        "\\begin{table}[!h]",
        "\\centering",
        "\\caption{Video and segment counts per exercise class by split}",
        "\\label{tab:counts-per-class}",
        "\\begin{tabular}{l r r r | r r r}",
        "\\hline",
        "\\multirow{2}{*}{\\textbf{Class}}   & \\multicolumn{3}{c|}{\\textbf{Videos}} & \\multicolumn{3}{c}{\\textbf{Segments}} \\\\",
        "  & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} & \\textbf{Train} & \\textbf{Val} & \\textbf{Test} \\\\",
        "\\hline"
    ]
    for act in ACTIONS:
        tr = video_counts.loc[act, "train"]
        va = video_counts.loc[act, "val"]
        te = video_counts.loc[act, "test"]
        tex_lines.append(f"{act:<22} & {tr:<3} & {va:<3} & {te:<3} & {tr:<3} & {va:<3} & {te:<3} \\\\")
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    # Save to files
    md_path = out_dir / "dataset_report.md"
    tex_path = out_dir / "table1_counts.tex"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))

    print(f"Dataset report saved to: {md_path}")
    print(f"LaTeX Table 1 saved to: {tex_path}")

    return {
        "video_counts": video_counts,
        "frame_stats": frame_stats,
        "res_counts": res_counts,
        "md_path": str(md_path),
        "tex_path": str(tex_path)
    }

def _extract_single_video_worker(task: Tuple[str, str]) -> Tuple[bool, str, Optional[str]]:
    """
    Worker function to process one video and write out its landmark CSV.
    """
    vid_file_str, out_file_str = task
    try:
        out_p = Path(out_file_str)
        if not out_p.exists() or out_p.stat().st_size == 0:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            extract_landmarks_from_video(vid_file_str, out_file_str)
        return True, Path(vid_file_str).name, None
    except Exception as e:
        return False, Path(vid_file_str).name, str(e)

def run_mediapipe_extraction_pipeline(
    raw_dataset_dir: str,
    output_landmark_dir: str = "data/landmarks",
    metadata_path: str = "Final_dataset_metadata.csv",
    smoke_test: bool = False,
    smoke_class: str = "barbell biceps curl",
    zip_output: bool = True,
    num_workers: int = 8
) -> str:
    """
    Extracts MediaPipe landmarks from video files in raw_dataset_dir.
    Supports multiprocessing with num_workers to utilize server CPU cores.
    If smoke_test is True, processes only smoke_class with minimal video count.
    Saves landmark CSV files and optionally creates a ZIP archive for server upload.
    """
    raw_dir = Path(raw_dataset_dir)
    out_dir = Path(output_landmark_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(metadata_path)
    if smoke_test:
        print(f"\n[SMOKE TEST MODE] Limiting extraction to class: '{smoke_class}'")
        meta_df = meta_df[meta_df["class"] == smoke_class].copy()
        # Take at most 2 train, 1 val, 1 test
        train_samples = meta_df[meta_df["split"] == "train"].head(2)
        val_samples = meta_df[meta_df["split"] == "val"].head(1)
        test_samples = meta_df[meta_df["split"] == "test"].head(1)
        meta_df = pd.concat([train_samples, val_samples, test_samples]).reset_index(drop=True)
        print(f"Smoke test video count: {len(meta_df)} (Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)})")

    tasks = []
    missing_count = 0

    for idx, row in meta_df.iterrows():
        rel_path = row["filepath"]
        split = row["split"]
        act = row["class"]

        candidates = [
            raw_dir / rel_path,
            raw_dir / Path(rel_path).name,
            raw_dir / split / act / Path(rel_path).name,
            raw_dir / act / Path(rel_path).name
        ]
        vid_file = None
        for c in candidates:
            if c.exists():
                vid_file = c
                break

        if vid_file is None:
            matches = list(raw_dir.glob(f"**/{Path(rel_path).name}"))
            if matches:
                vid_file = matches[0]

        if vid_file and vid_file.exists():
            out_file = out_dir / split / act / f"{vid_file.stem}.csv"
            tasks.append((str(vid_file), str(out_file)))
        else:
            missing_count += 1

    total_tasks = len(tasks)
    print(f"Found {total_tasks} valid videos to process ({missing_count} missing from metadata).")
    extracted_count = 0

    if num_workers > 1 and total_tasks > 1:
        print(f"Starting parallel extraction using {num_workers} worker processes...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {executor.submit(_extract_single_video_worker, t): t[0] for t in tasks}
            for future in as_completed(future_to_video):
                success, vname, err = future.result()
                extracted_count += 1
                if success:
                    if extracted_count % 25 == 0 or extracted_count == total_tasks:
                        print(f"[{extracted_count}/{total_tasks}] Extracted landmarks: {vname}")
                else:
                    print(f"[{extracted_count}/{total_tasks}] ERROR extracting {vname}: {err}")
    else:
        for t in tasks:
            success, vname, err = _extract_single_video_worker(t)
            extracted_count += 1
            if success:
                print(f"[{extracted_count}/{total_tasks}] Extracted landmarks: {vname}")
            else:
                print(f"[{extracted_count}/{total_tasks}] ERROR extracting {vname}: {err}")

    print(f"\nMediaPipe Extraction finished: {extracted_count} processed, {missing_count} missing.")

    # Zip output for server transfer
    if zip_output and extracted_count > 0:
        zip_name = "landmarks_smoketest.zip" if smoke_test else "landmarks_dataset.zip"
        zip_path = Path("outputs") / zip_name
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Packaging extracted landmark CSVs into {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for f in files:
                    if f.endswith(".csv"):
                        full_p = Path(root) / f
                        arcname = full_p.relative_to(out_dir)
                        zf.write(full_p, arcname)
        print(f"ZIP package created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
        return str(zip_path)

    return str(out_dir)

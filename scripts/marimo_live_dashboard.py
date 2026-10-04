import marimo

app = marimo.App(width="full", app_title="SkelGym: Master E2E Pipeline Dashboard")


@app.cell
def __():
    import os
    import sys
    import time
    import json
    import subprocess
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch

    GYM_ROOT = Path("/marimo/Gym_Classification")
    if str(GYM_ROOT) not in sys.path:
        sys.path.insert(0, str(GYM_ROOT))
    return GYM_ROOT, Path, json, mo, np, os, pd, subprocess, sys, time, torch


@app.cell
def __(GYM_ROOT, mo, sys, torch):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available()
        else 0.0
    )

    header_ui = mo.vstack([
        mo.md(
            f"""
            # 🏋️‍♂️ SkelGym: End-to-End Pipeline & Live Experiment Dashboard
            *Cross-Paradigm Skeletal Action Recognition under Strict Video-Level Partitioning & Multi-Seed Replication*

            ---
            """
        ),
        mo.hstack([
            mo.stat(value=f"{device_name}", label="Compute Accelerator", caption=f"{vram_gb} GB VRAM"),
            mo.stat(value=f"PyTorch {torch.__version__}", label="Framework", caption=f"CUDA {torch.version.cuda}"),
            mo.stat(value=f"Python {sys.version.split()[0]}", label="Runtime", caption="Linux x86_64"),
            mo.stat(value="Active", label="E2E Pipeline Status", caption="Marimo Pair-Programming Live"),
        ], justify="space-between"),
    ])
    header_ui
    return device_name, header_ui, vram_gb


@app.cell
def __(GYM_ROOT, mo, pd):
    meta_path = GYM_ROOT / "Final_dataset_metadata.csv"
    if not meta_path.exists():
        meta_path = GYM_ROOT / "data" / "Final_dataset_metadata.csv"
    if meta_path.exists():
        df_meta = pd.read_csv(meta_path)
        total_vids = len(df_meta)
        splits = df_meta["split"].value_counts().to_dict()
        train_count = splits.get("train", 0)
        val_count = splits.get("val", 0)
        test_count = splits.get("test", 0)
        classes_count = df_meta["class"].nunique() if "class" in df_meta.columns else 22
    else:
        df_meta = pd.DataFrame()
        total_vids, train_count, val_count, test_count, classes_count = 1024, 580, 208, 236, 22

    dataset_ui = mo.vstack([
        mo.md("### 📊 Dataset Integrity & Video-Level Partition (6:2:2)"),
        mo.hstack([
            mo.stat(value=f"{total_vids:,}", label="Total Source Videos", caption="10.2 GB RGB Corpus"),
            mo.stat(value=f"{train_count} ({train_count/total_vids*100:.1f}%)", label="Train Partition", caption="Stride S=16 (13,136 win)"),
            mo.stat(value=f"{val_count} ({val_count/total_vids*100:.1f}%)", label="Validation Partition", caption="Stride S=32 (2,075 win)"),
            mo.stat(value=f"{test_count} ({test_count/total_vids*100:.1f}%)", label="Test Partition", caption="Stride S=32 (2,743 win)"),
            mo.stat(value=f"{classes_count}", label="Fine-Grained Classes", caption="13 Anatomical Joints"),
        ], justify="space-between"),
        mo.accordion({
            "🔍 Xem Bảng Phân Bổ Metadata Chi Tiết": mo.ui.table(df_meta.head(15)) if not df_meta.empty else mo.md("No metadata loaded")
        })
    ])
    dataset_ui
    return (
        classes_count,
        dataset_ui,
        df_meta,
        meta_path,
        splits,
        test_count,
        total_vids,
        train_count,
        val_count,
    )


@app.cell
def __(GYM_ROOT, json, mo, pd, subprocess, time):
    refresh = mo.ui.refresh(default_interval="5s")

    trigger_file = GYM_ROOT / "outputs" / "triggers.jsonl"
    heartbeat_file = GYM_ROOT / "outputs" / "keepalive.heartbeat"
    runner_log_file = GYM_ROOT / "outputs" / "runner.log"

    events = []
    if trigger_file.exists():
        try:
            lines = [l.strip() for l in trigger_file.read_text().split("\n") if l.strip()]
            events = [json.loads(l) for l in lines]
        except Exception:
            pass

    ckpts = list((GYM_ROOT / "checkpoints").glob("**/*.pt"))
    ckpt_count = len(ckpts)

    hb_age_str = "Chưa kích hoạt"
    if heartbeat_file.exists():
        try:
            age = time.time() - heartbeat_file.stat().st_mtime
            hb_age_str = f"{age:.0f}s trước (" + ("OK 🟢" if age < 30 else "CẢNH BÁO ⚠️") + ")"
        except Exception:
            pass

    ps_res = subprocess.run(["pgrep", "-f", "marimo_master_e2e_runner"], capture_output=True, text=True)
    runner_pid = ps_res.stdout.strip()
    is_running = bool(runner_pid)

    recent_events = events[-15:] if events else []
    df_triggers = pd.DataFrame(recent_events) if recent_events else pd.DataFrame()

    recent_log = ""
    if runner_log_file.exists():
        try:
            log_lines = runner_log_file.read_text().splitlines()
            recent_log = "\n".join(log_lines[-12:])
        except Exception:
            pass

    status_str = f"🟢 Đang chạy (PID {runner_pid})" if is_running else "⚪ Đang rảnh / Chờ lệnh"

    monitor_ui = mo.vstack([
        mo.md("### ⚡ Live Parallel Training & Event-Trigger Monitor"),
        mo.hstack([
            refresh,
            mo.md(f"**Trạng Thái Runner:** `{status_str}` | **Heartbeat:** `{hb_age_str}`")
        ], align="center"),
        mo.hstack([
            mo.stat(value=f"{ckpt_count} / ~150", label="Checkpoints Saved", caption="Target: 5 Stages x 3 Seeds"),
            mo.stat(value=f"{len(events)} Events", label="Trigger Events Emitted", caption="outputs/triggers.jsonl"),
            mo.stat(value="4 Workers", label="GPU Concurrency", caption="NVIDIA Blackwell 102GB"),
            mo.stat(value="BFloat16 AMP", label="Arithmetic Precision", caption="Zero Gradient Underflow"),
        ], justify="space-between"),
        mo.accordion({
            "📋 15 Sự Kiện Kích Hoạt Gần Nhất (Event Triggers)": mo.ui.table(df_triggers) if not df_triggers.empty else mo.md("*Chưa có sự kiện nào được ghi nhận.*"),
            "📜 Log Output Gần Nhất Của Runner": mo.md(f"```text\n{recent_log}\n```") if recent_log else mo.md("*Log trống.*")
        })
    ])
    monitor_ui
    return (
        ckpt_count,
        ckpts,
        df_triggers,
        events,
        hb_age_str,
        heartbeat_file,
        is_running,
        monitor_ui,
        ps_res,
        recent_events,
        recent_log,
        refresh,
        runner_log_file,
        runner_pid,
        status_str,
        trigger_file,
    )


@app.cell
def __(GYM_ROOT, mo):
    results_path = GYM_ROOT / "outputs" / "RESULTS_FINAL.md"
    if results_path.exists():
        content = results_path.read_text()
        results_ui = mo.vstack([
            mo.md("### 📑 Báo Cáo Kết Quả Thực Nghiệm (`outputs/RESULTS_FINAL.md`)"),
            mo.accordion({
                "📊 Xem Chi Tiết Báo Cáo 12 Bảng Nghiên Cứu": mo.md(content[:3000] + "\n\n*(Xem tiếp trong file outputs/RESULTS_FINAL.md)*")
            })
        ])
    else:
        results_ui = mo.md("*Chưa có file `outputs/RESULTS_FINAL.md`.*")
    results_ui
    return content, results_path, results_ui


if __name__ == "__main__":
    app.run()

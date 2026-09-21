#!/usr/bin/env python3
"""
Reproducible Hardware Inference Latency Benchmark for SkelGym.
Measures per-window inference latency (ms) and throughput (FPS)
for all constituent model backbones and ensemble configurations.

Usage:
    python scripts/benchmark_hardware_latency.py --device auto
    python scripts/benchmark_hardware_latency.py --device cpu
    python scripts/benchmark_hardware_latency.py --device mps
    python scripts/benchmark_hardware_latency.py --device cuda
"""

import sys
import time
import argparse
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model

def get_device(dev_arg: str) -> torch.device:
    if dev_arg != "auto":
        return torch.device(dev_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def benchmark_model(model: torch.nn.Module, dummy_input: torch.Tensor, device: torch.device, warmup: int = 50, iterations: int = 500) -> float:
    model.eval()
    x = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        sync_device(device)

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
            sync_device(device)
        t1 = time.perf_counter()

    lat_ms = (t1 - t0) / iterations * 1000.0
    return lat_ms

def main():
    parser = argparse.ArgumentParser(description="SkelGym Hardware Latency Benchmark")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Inference device")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=500, help="Number of timed benchmark iterations")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"==================================================================")
    print(f"  SkelGym Inference Latency Benchmark")
    print(f"  Active Device: {device} | Warmup: {args.warmup} | Iterations: {args.iterations}")
    print(f"==================================================================")

    # Model specifications (Type, Feature Space, Input Shape for T=32)
    models_spec = [
        ("Transformer Mix (117-d)", "Transformer", "mix", torch.randn(1, 32, 117), "399K"),
        ("AAGCN Joint Stream", "AAGCN", "rel_3d", torch.randn(1, 32, 39), "378K"),
        ("AAGCN Bone Stream", "AAGCN", "bone_3d", torch.randn(1, 32, 39), "378K"),
        ("AAGCN Joint-Motion Stream", "AAGCN", "joint_motion_3d", torch.randn(1, 32, 39), "378K"),
        ("AAGCN Bone-Motion Stream", "AAGCN", "bone_motion_3d", torch.randn(1, 32, 39), "378K"),
    ]

    latencies = {}
    print(f"\n{'Model Architecture':<30} | {'Parameters':<10} | {'Latency (ms)':<14} | {'Throughput (FPS)':<16}")
    print("-" * 76)

    for name, m_type, feat, dummy_in, params in models_spec:
        model = build_model(m_type, feat, num_classes=22).to(device)
        lat = benchmark_model(model, dummy_in, device, warmup=args.warmup, iterations=args.iterations)
        latencies[name] = lat
        fps = 1000.0 / lat
        print(f"{name:<30} | {params:<10} | {lat:>9.2f} ms   | {fps:>12.0f} FPS")

    # Ensembles
    lite_lat = latencies["Transformer Mix (117-d)"] + latencies["AAGCN Bone Stream"]
    lite_fps = 1000.0 / lite_lat

    full_lat = sum(latencies.values())
    full_fps = 1000.0 / full_lat

    print("-" * 76)
    print(f"{'SkelGym-Lite (Transformer + Bone)':<30} | {'777K':<10} | {lite_lat:>9.2f} ms   | {lite_fps:>12.0f} FPS")
    print(f"{'SkelGym-Full (5 Streams)':<30} | {'1.91M':<10} | {full_lat:>9.2f} ms   | {full_fps:>12.0f} FPS")
    print("=" * 76)
    print("Benchmark complete.\n")

if __name__ == "__main__":
    main()

"""
Script to generate CLEAN, INDIVIDUAL, and MINIMALIST architecture diagrams
for ALL models and fusion setups in SkelGym.

Outputs high-resolution individual PNG (300 DPI) and vector PDF files in paper/images/:
  1. model_lstm (Unidirectional LSTM Baseline)
  2. model_bilstm (Bidirectional LSTM Baseline)
  3. model_transformer (Skeletal Transformer Encoder)
  4. model_stgcn (Baseline ST-GCN)
  5. model_aagcn (Adaptive Spatial-Temporal GCN & Adaptive Block)
  6. model_dual_branch (Dual-Branch Early Feature Fusion: Coordinates + Angles)
  7. model_fusion_2stream (Two-Stream AAGCN: Joint + Bone)
  8. model_fusion_3stream (Three-Stream AAGCN: Joint + Bone + Motion)
  9. model_fusion_4stream (Four-Stream AAGCN: Unified Graph Paradigm)
  10. model_fusion_skelgym_lite (SkelGym-Lite: Dual-Stream Cross-Paradigm)
  11. model_fusion_skelgym_full (SkelGym-Full: 5-Stream Cross-Paradigm SLSQP Ensemble)
  12. model_ensemble_stacking (Stacking Meta-Learner Architecture)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = "/Volumes/WorkSpace/Project/Gym_Classification/paper/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Clean Academic Color Palette (Minimalist, High Contrast)
# -------------------------------------------------------------
C_BG_BOX    = "#FFFFFF"
C_LINE_DARK = "#2D3748"
C_TEXT_DARK = "#1A202C"
C_TEXT_MUTED= "#4A5568"

# Theme colors for components
C_INPUT   = "#EBF8FF"   # Light Blue
C_BORDER_IN = "#3182CE"

C_TRANS   = "#F0F9FF"   # Soft Sky Blue
C_BORDER_TR = "#2B6CB0"

C_GCN     = "#E6FFFA"   # Soft Mint / Teal
C_BORDER_GC = "#319795"

C_RECURR  = "#FAF5FF"   # Soft Lavender
C_BORDER_RC = "#805AD5"

C_HEAD    = "#FFF5F5"   # Soft Coral / Pink
C_BORDER_HD = "#E53E3E"

C_FUSION  = "#FEFCBF"   # Soft Light Amber
C_BORDER_FS = "#D69E2E"

C_OUTPUT  = "#EDFDFD"
C_BORDER_OUT= "#234E52"


def create_box(ax, x, y, w, h, text_top, text_bot=None, bg=C_BG_BOX, border=C_LINE_DARK,
               lw=1.3, radius=0.03, fs_top=9.0, fs_bot=7.5, zorder=2):
    """Draws a clean, rounded rectangular block with 1 or 2 lines of text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg, edgecolor=border, linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(patch)
    if text_bot is None or text_bot == "":
        ax.text(x + w / 2, y + h / 2, text_top, ha='center', va='center',
                fontsize=fs_top, fontweight='bold', color=C_TEXT_DARK, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h * 0.62, text_top, ha='center', va='center',
                fontsize=fs_top, fontweight='bold', color=C_TEXT_DARK, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.28, text_bot, ha='center', va='center',
                fontsize=fs_bot, color=C_TEXT_MUTED, zorder=zorder + 1)
    return patch


def create_arrow(ax, x1, y1, x2, y2, color=C_LINE_DARK, lw=1.3, rad=0.0, zorder=3):
    """Draws a clean directional arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=4.5,head_width=3.2",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow


# =============================================================================
# 1. UNIDIRECTIONAL LSTM BASELINE
# =============================================================================
def draw_lstm_diagram():
    fig, ax = plt.subplots(figsize=(6.2, 7.8), dpi=300)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    ax.text(3.1, 7.45, "Unidirectional LSTM Sequence Baseline", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Input
    create_box(ax, 1.1, 6.5, 4.0, 0.55, "Input Sequence Tensor", "Shape: [Batch, Time=32, Dim=117]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.1, 6.5, 3.1, 6.0)

    # LayerNorm
    create_box(ax, 1.1, 5.45, 4.0, 0.55, "LayerNorm(117)", "Standardizes skeletal landmark vector",
               bg=C_BG_BOX, border=C_BORDER_RC)
    create_arrow(ax, 3.1, 5.45, 3.1, 4.95)

    # 2-Layer LSTM
    create_box(ax, 1.1, 3.75, 4.0, 1.2, "2-Layer Unidirectional LSTM",
               "hidden_dim = 160 (Forward time modeling)\nnum_layers = 2 | Dropout = 0.3\nbatch_first = True | ~345K Parameters\nOutput: [Batch, Time=32, Dim=160]",
               bg=C_RECURR, border=C_BORDER_RC, fs_bot=7.2)
    create_arrow(ax, 3.1, 3.75, 3.1, 3.25)

    # GAP
    create_box(ax, 1.1, 2.3, 4.0, 0.55, "Temporal Global Average Pooling (GAP)", "pooled = out.mean(dim=1) → [Batch, 160]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.1, 2.3, 3.1, 1.8)

    # MLP Head
    create_box(ax, 1.1, 0.5, 4.0, 0.90, "Two-Stage MLP Classifier Head",
               "Dropout(0.3) → Linear(160 → 64) → ReLU\n→ Dropout(0.3) → Linear(64 → 22)\nOutput: 22 Exercise Class Probabilities",
               bg=C_HEAD, border=C_BORDER_HD, fs_bot=7.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_lstm.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_lstm.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_lstm.png & .pdf")


# =============================================================================
# 2. BIDIRECTIONAL LSTM BASELINE
# =============================================================================
def draw_bilstm_diagram():
    fig, ax = plt.subplots(figsize=(6.2, 7.8), dpi=300)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    ax.text(3.1, 7.45, "Bidirectional LSTM Sequence Baseline", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Input
    create_box(ax, 1.1, 6.5, 4.0, 0.55, "Input Sequence Tensor", "Shape: [Batch, Time=32, Dim=117]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.1, 6.5, 3.1, 6.0)

    # LayerNorm
    create_box(ax, 1.1, 5.45, 4.0, 0.55, "LayerNorm(117)", "Feature standardization across joints",
               bg=C_BG_BOX, border=C_BORDER_RC)
    create_arrow(ax, 3.1, 5.45, 3.1, 4.95)

    # 2-Layer BiLSTM
    create_box(ax, 1.1, 3.75, 4.0, 1.2, "2-Layer Bidirectional LSTM",
               "hidden_dim = 96 per direction\nForward (96) + Backward (96) = 192 total\nDropout = 0.3 | batch_first = True\nOutput: [Batch, Time=32, Dim=192]",
               bg=C_RECURR, border=C_BORDER_RC, fs_bot=7.2)
    create_arrow(ax, 3.1, 3.75, 3.1, 3.25)

    # GAP
    create_box(ax, 1.1, 2.3, 4.0, 0.55, "Temporal Global Average Pooling (GAP)", "Output: [Batch, 192]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.1, 2.3, 3.1, 1.8)

    # MLP Head
    create_box(ax, 1.1, 0.5, 4.0, 0.90, "Two-Stage MLP Classifier Head",
               "Dropout(0.3) → Linear(192 → 64) → ReLU\n→ Dropout(0.3) → Linear(64 → 22)\nOutput: 22 Exercise Class Probabilities",
               bg=C_HEAD, border=C_BORDER_HD, fs_bot=7.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_bilstm.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_bilstm.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_bilstm.png & .pdf")


# =============================================================================
# 3. SKELETAL TRANSFORMER
# =============================================================================
def draw_transformer_diagram():
    fig, ax = plt.subplots(figsize=(6.5, 9.2), dpi=300)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(3.25, 8.85, "Skeletal Transformer Encoder", ha='center', va='center',
            fontsize=13, fontweight='bold', color=C_TEXT_DARK)

    # 1. Input
    create_box(ax, 1.25, 7.9, 4.0, 0.60, "Input Skeleton Sequence", "Shape: [Batch, Time=32, Dim=117]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.25, 7.9, 3.25, 7.35)

    # 2. Linear Projection + Pos Enc
    create_box(ax, 1.25, 6.75, 4.0, 0.60, "Linear Projection + Sinusoidal PE",
               "Linear(117 → 128) + PE(Time=32, d=128)", bg=C_BG_BOX, border=C_BORDER_TR)
    create_arrow(ax, 3.25, 6.75, 3.25, 6.2)

    # 3. 4x Transformer Layer Container
    cont = FancyBboxPatch((0.9, 2.5), 4.7, 3.7, boxstyle="round,pad=0,rounding_size=0.04",
                          facecolor="#F7FAFC", edgecolor=C_BORDER_TR, linewidth=1.5, linestyle="--")
    ax.add_patch(cont)
    ax.text(1.1, 5.95, "4× Pre-LN Transformer Encoder Layers", fontsize=9.5, fontweight='bold', color=C_BORDER_TR)

    # Inside container:
    create_box(ax, 1.25, 5.0, 4.0, 0.65, "Multi-Head Self-Attention (MHSA)",
               "Pre-LN → 8 Heads, d_k=16 (d=128)\n+ Residual Add", bg=C_TRANS, border=C_BORDER_TR)
    create_arrow(ax, 3.25, 5.0, 3.25, 4.35)

    create_box(ax, 1.25, 3.5, 4.0, 0.70, "Feed-Forward Network (FFN)",
               "Pre-LN → Linear(128 → 256) → GELU\n→ Linear(256 → 128) + Residual Add", bg=C_TRANS, border=C_BORDER_TR)

    create_arrow(ax, 3.25, 2.5, 3.25, 1.95)

    # 4. GAP
    create_box(ax, 1.25, 1.45, 4.0, 0.50, "Temporal Global Average Pooling (GAP)", "Output: [Batch, 128]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.25, 1.45, 3.25, 0.95)

    # 5. Classifier Head
    create_box(ax, 1.25, 0.35, 4.0, 0.60, "Two-Stage MLP Classifier Head",
               "Linear(128 → 64) → ReLU → Linear(64 → 22 Classes)", bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_transformer.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_transformer.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_transformer.png & .pdf")


# =============================================================================
# 4. ST-GCN BASELINE
# =============================================================================
def draw_stgcn_diagram():
    fig, ax = plt.subplots(figsize=(6.2, 8.5), dpi=300)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(3.1, 8.15, "ST-GCN Baseline (Yan et al. 2018)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Input
    create_box(ax, 1.1, 7.2, 4.0, 0.55, "Input Kinematic Stream", "Shape: [B, C_in=3, T=32, V=13]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.1, 7.2, 3.1, 6.7)

    # Data BN
    create_box(ax, 1.1, 6.15, 4.0, 0.55, "Data BatchNorm2d(3)", "Standardizes coordinates",
               bg=C_BG_BOX, border=C_TEXT_MUTED)
    create_arrow(ax, 3.1, 6.15, 3.1, 5.65)

    # 3x ST-GCN Block Container
    cont = FancyBboxPatch((0.8, 2.6), 4.6, 2.9, boxstyle="round,pad=0,rounding_size=0.04",
                          facecolor="#F7FAFC", edgecolor=C_TEXT_MUTED, linewidth=1.4, linestyle="--")
    ax.add_patch(cont)
    ax.text(1.0, 5.25, "3× ST-GCN Blocks (Channels: 48 → 96 → 150)", fontsize=9.0, fontweight='bold', color=C_TEXT_DARK)

    create_box(ax, 1.1, 4.4, 4.0, 0.65, "Spatial Graph Conv (Rigid A_phys)",
               "A = A_norm ⊙ M (Fixed physical limbs)\nConv2d(1×1) → BatchNorm2d → ReLU",
               bg=C_BG_BOX, border=C_TEXT_MUTED)
    create_arrow(ax, 3.1, 4.4, 3.1, 3.9)

    create_box(ax, 1.1, 2.85, 4.0, 0.65, "Temporal Conv & Residual Shortcut",
               "Conv2d(9×1, stride s) → BN → Dropout\nResidual: Conv2d(1×1, stride s) + BN → ReLU",
               bg=C_BG_BOX, border=C_TEXT_MUTED)

    create_arrow(ax, 3.1, 2.6, 3.1, 2.1)

    # GAP
    create_box(ax, 1.1, 1.5, 4.0, 0.55, "Global Average Pooling (GAP)", "Output: [Batch, 150]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 3.1, 1.5, 3.1, 1.0)

    # Classifier
    create_box(ax, 1.1, 0.35, 4.0, 0.65, "Linear Classifier Head",
               "Linear(150 → 22 Classes) → Softmax\nOutput: 22 Exercise Class Probabilities",
               bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_stgcn.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_stgcn.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_stgcn.png & .pdf")


# =============================================================================
# 5. ADAPTIVE ST-GCN (AAGCN)
# =============================================================================
def draw_aagcn_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(5.25, 8.2, "Adaptive Spatial-Temporal Graph Convolutional Network (AAGCN)", ha='center', va='center',
            fontsize=13, fontweight='bold', color=C_TEXT_DARK)

    # Left Column: Macro Backbone (x: 0.5 to 4.8)
    left_cont = FancyBboxPatch((0.5, 0.35), 4.3, 7.8, boxstyle="round,pad=0,rounding_size=0.04",
                               facecolor="#FAFAFA", edgecolor=C_LINE_DARK, linewidth=1.2)
    ax.add_patch(left_cont)
    ax.text(2.65, 7.85, "AAGCN Macro Architecture (Per-Stream)", ha='center',
            fontsize=10.5, fontweight='bold', color=C_TEXT_DARK)

    create_box(ax, 0.8, 6.9, 3.7, 0.55, "Input Kinematic Stream", "Shape: [Batch, C=3, T=32, V=13]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 2.65, 6.9, 2.65, 6.4)

    create_box(ax, 0.8, 5.85, 3.7, 0.55, "Data BatchNorm2d(3)", "Joint-level normalisation",
               bg=C_BG_BOX, border=C_TEXT_MUTED)
    create_arrow(ax, 2.65, 5.85, 2.65, 5.35)

    stages = [
        ("AAGCN Stage 1", "Channels: 3 → 48, Stride: 1", 4.8, C_GCN, C_BORDER_GC),
        ("AAGCN Stage 2", "Channels: 48 → 96, Stride: 2 (T: 32→16)", 3.9, C_GCN, C_BORDER_GC),
        ("AAGCN Stage 3", "Channels: 96 → 150, Stride: 2 (T: 16→8)", 3.0, C_GCN, C_BORDER_GC),
    ]
    for name, desc, y_pos, bg, border in stages:
        create_box(ax, 0.8, y_pos, 3.7, 0.55, name, desc, bg=bg, border=border)
        if y_pos > 3.0:
            create_arrow(ax, 2.65, y_pos, 2.65, y_pos - 0.35)

    create_arrow(ax, 2.65, 3.0, 2.65, 2.45)
    create_box(ax, 0.8, 1.85, 3.7, 0.60, "Global Average Pooling (GAP)", "Spatial (V) + Temporal (T) Pooling\nOutput: [Batch, 150]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 2.65, 1.85, 2.65, 1.35)

    create_box(ax, 0.8, 0.65, 3.7, 0.70, "Stream Linear Classifier Head",
               "Linear(150 → 22 Classes) → Softmax\nYields Soft Probability Vector P_m ∈ Δ²¹",
               bg=C_HEAD, border=C_BORDER_HD)

    # Right Column: Zoom into Adaptive Graph Convolution Block (x: 5.3 to 10.0)
    right_cont = FancyBboxPatch((5.3, 0.35), 4.7, 7.8, boxstyle="round,pad=0,rounding_size=0.04",
                               facecolor="#FFFFFF", edgecolor=C_BORDER_GC, linewidth=1.5)
    ax.add_patch(right_cont)
    ax.text(7.65, 7.85, "Adaptive Graph Conv (Spatial-Temporal Block)", ha='center',
            fontsize=10.5, fontweight='bold', color=C_BORDER_GC)

    # 3 Adjacency Branches
    create_box(ax, 5.5, 6.75, 1.3, 0.75, "Branch 1", "A_topo ⊙ M\n(Anatomy)", bg=C_BG_BOX, border=C_TEXT_MUTED, fs_top=8.0, fs_bot=7.0)
    create_box(ax, 7.0, 6.75, 1.3, 0.75, "Branch 2", "Matrix B\n(Global Free)", bg=C_INPUT, border=C_BORDER_IN, fs_top=8.0, fs_bot=7.0)
    create_box(ax, 8.5, 6.75, 1.3, 0.75, "Branch 3", "C(X) Attention\n(Dynamic)", bg=C_FUSION, border=C_BORDER_FS, fs_top=8.0, fs_bot=7.0)

    create_arrow(ax, 6.15, 6.75, 7.65, 6.2)
    create_arrow(ax, 7.65, 6.75, 7.65, 6.2)
    create_arrow(ax, 9.15, 6.75, 7.65, 6.2)

    # Summation
    create_box(ax, 5.6, 5.65, 4.1, 0.55, "Unified Adjacency Matrix",
               "A_total = (A_topo ⊙ M) + B + C(X)", bg=C_GCN, border=C_BORDER_GC)
    create_arrow(ax, 7.65, 5.65, 7.65, 5.05)

    # Graph Message Passing
    create_box(ax, 5.6, 4.4, 4.1, 0.65, "Spatial Graph Message Passing",
               "X_g = A_total · X  →  Conv2d(1×1) → BN → GELU", bg=C_BG_BOX, border=C_BORDER_GC)
    create_arrow(ax, 7.65, 4.4, 7.65, 3.8)

    # Temporal Convolution
    create_box(ax, 5.6, 3.15, 4.1, 0.65, "Temporal Convolution (MS-TCN)",
               "Conv2d(kernel=9×1, stride=s) → BN → Dropout(0.2)", bg=C_BG_BOX, border=C_BORDER_TR)
    create_arrow(ax, 7.65, 3.15, 7.65, 2.55)

    # Residual Shortcut
    create_box(ax, 5.6, 1.85, 4.1, 0.70, "Residual Skip Connection (+)",
               "Shortcut: Conv2d(1×1, stride=s) + BN (if stride/channels change)\nElement-wise Sum (+) → GELU",
               bg=C_FUSION, border=C_BORDER_FS, fs_bot=7.0)

    # Output of Block
    create_box(ax, 5.6, 0.65, 4.1, 0.55, "Block Output Feature Map",
               "Passed to next stage or GAP", bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 7.65, 1.85, 7.65, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_aagcn.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_aagcn.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_aagcn.png & .pdf")


# =============================================================================
# 6. DUAL-BRANCH EARLY FEATURE FUSION (BranchConcat Architecture)
# =============================================================================
def draw_dual_branch_diagram():
    fig, ax = plt.subplots(figsize=(7.8, 6.8), dpi=300)
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    ax.text(3.9, 6.45, "Dual-Branch Feature Fusion (BranchConcat Architecture)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Branch 1: Coordinates
    create_box(ax, 0.6, 5.2, 3.0, 0.75, "Branch 1: 3D Coordinates", "Shape: [B, T=32, Dim₁=39/49]\nEncodes Euclidean metric position",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 2.1, 5.2, 2.1, 4.5)
    create_box(ax, 0.6, 3.4, 3.0, 1.1, "Branch 1 Temporal Encoder",
               "Linear(Dim₁ → d_model)\nLSTM / Transformer Encoder (L=2)\nGlobal Pooling / Last Step\nOutput: r₁ ∈ ℝ^d_model (d=64)",
               bg=C_TRANS, border=C_BORDER_TR, fs_bot=7.0)

    # Branch 2: Angles
    create_box(ax, 4.2, 5.2, 3.0, 0.75, "Branch 2: Joint Angles", "Shape: [B, T=32, Dim₂=78/286]\nEncodes scale-invariant rotation",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 5.7, 5.2, 5.7, 4.5)
    create_box(ax, 4.2, 3.4, 3.0, 1.1, "Branch 2 Temporal Encoder",
               "Linear(Dim₂ → d_model)\nLSTM / Transformer Encoder (L=2)\nGlobal Pooling / Last Step\nOutput: r₂ ∈ ℝ^d_model (d=64)",
               bg=C_TRANS, border=C_BORDER_TR, fs_bot=7.0)

    # Concatenation
    create_arrow(ax, 2.1, 3.4, 3.3, 2.35)
    create_arrow(ax, 5.7, 3.4, 4.5, 2.35)
    create_box(ax, 2.0, 1.6, 3.8, 0.75, "Feature Concatenation [r₁, r₂]",
               "Merged Latent Vector: r = [r₁ || r₂]\nShape: [Batch, 2 × d_model = 128]",
               bg=C_FUSION, border=C_BORDER_FS)

    # Classifier Head
    create_arrow(ax, 3.9, 1.6, 3.9, 1.1)
    create_box(ax, 1.9, 0.35, 4.0, 0.75, "Shared MLP Classifier Head",
               "Linear(128 → 64) → ReLU → Dropout(0.3)\n→ Linear(64 → 22 Classes) → Softmax",
               bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_dual_branch.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_dual_branch.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_dual_branch.png & .pdf")


# =============================================================================
# 7. TWO-STREAM AAGCN
# =============================================================================
def draw_fusion_2stream():
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(3.75, 5.85, "Two-Stream AAGCN (Static Kinematics Late Fusion)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Stream 1: Joint
    create_box(ax, 0.6, 4.6, 2.9, 0.7, "Joint Stream (Rel 3D)", "T=32, V=13 | 3D Coordinates", bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 2.05, 4.6, 2.05, 3.9)
    create_box(ax, 0.6, 3.2, 2.9, 0.7, "AAGCN Joint Backbone", "3 Stages [48, 96, 150]\nOutput: P_joint ∈ Δ²¹", bg=C_GCN, border=C_BORDER_GC)

    # Stream 2: Bone
    create_box(ax, 4.0, 4.6, 2.9, 0.7, "Bone Stream (Bone 3D)", "T=32, V=13 | Directed Vectors", bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 5.45, 4.6, 5.45, 3.9)
    create_box(ax, 4.0, 3.2, 2.9, 0.7, "AAGCN Bone Backbone", "3 Stages [48, 96, 150]\nOutput: P_bone ∈ Δ²¹", bg=C_GCN, border=C_BORDER_GC)

    # Late Fusion Node
    create_arrow(ax, 2.05, 3.2, 3.4, 2.2)
    create_arrow(ax, 5.45, 3.2, 4.1, 2.2)

    create_box(ax, 1.8, 1.4, 3.9, 0.8, "Weighted Soft Late Fusion",
               "P_final = w₁ · P_joint + w₂ · P_bone\n(w₁ + w₂ = 1,  w ≥ 0)", bg=C_FUSION, border=C_BORDER_FS)
    create_arrow(ax, 3.75, 1.4, 3.75, 0.95)

    create_box(ax, 2.0, 0.35, 3.5, 0.6, "Final Exercise Prediction", "argmax(P_final) ∈ {1, ..., 22}",
               bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_2stream.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_2stream.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_fusion_2stream.png & .pdf")


# =============================================================================
# 8. THREE-STREAM AAGCN (Joint + Bone + Motion)
# =============================================================================
def draw_fusion_3stream():
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=300)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(4.6, 5.85, "Three-Stream AAGCN (Joint + Bone + Motion Fusion)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Stream 1: Joint Position
    create_box(ax, 0.5, 4.6, 2.5, 0.7, "Joint Stream (Pos 3D)", "T=32, V=13 | 3D Coordinates", bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 1.75, 4.6, 1.75, 3.9)
    create_box(ax, 0.5, 3.2, 2.5, 0.7, "AAGCN Joint Backbone", "3 Stages [48, 96, 150]\nOutput: P_joint ∈ Δ²¹", bg=C_GCN, border=C_BORDER_GC)

    # Stream 2: Bone Vector
    create_box(ax, 3.35, 4.6, 2.5, 0.7, "Bone Stream (Bone 3D)", "T=32, V=13 | Limb Vectors", bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 4.6, 4.6, 4.6, 3.9)
    create_box(ax, 3.35, 3.2, 2.5, 0.7, "AAGCN Bone Backbone", "3 Stages [48, 96, 150]\nOutput: P_bone ∈ Δ²¹", bg=C_GCN, border=C_BORDER_GC)

    # Stream 3: Motion Vector
    create_box(ax, 6.2, 4.6, 2.5, 0.7, "Motion Stream (Vel 3D)", "T=31, V=13 | Joint Velocities", bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 7.45, 4.6, 7.45, 3.9)
    create_box(ax, 6.2, 3.2, 2.5, 0.7, "AAGCN Motion Backbone", "3 Stages [48, 96, 150]\nOutput: P_motion ∈ Δ²¹", bg=C_GCN, border=C_BORDER_GC)

    # Late Fusion
    create_arrow(ax, 1.75, 3.2, 4.0, 2.2)
    create_arrow(ax, 4.6, 3.2, 4.6, 2.2)
    create_arrow(ax, 7.45, 3.2, 5.2, 2.2)

    create_box(ax, 2.6, 1.4, 4.0, 0.8, "Weighted Soft Late Fusion",
               "P_final = w₁ · P_joint + w₂ · P_bone + w₃ · P_motion\n(∑ w_i = 1,  w_i ≥ 0)", bg=C_FUSION, border=C_BORDER_FS)
    create_arrow(ax, 4.6, 1.4, 4.6, 0.95)

    create_box(ax, 2.85, 0.35, 3.5, 0.6, "Final Exercise Prediction", "argmax(P_final) ∈ {1, ..., 22}",
               bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_3stream.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_3stream.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_fusion_3stream.png & .pdf")


# =============================================================================
# 9. FOUR-STREAM AAGCN
# =============================================================================
def draw_fusion_4stream():
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(5.25, 5.85, "Four-Stream AAGCN (Unified Graph Paradigm)", ha='center', va='center',
            fontsize=13, fontweight='bold', color=C_TEXT_DARK)

    streams = [
        ("Joint Position", "Rel 3D [B,3,32,13]", "P_joint", 0.4),
        ("Bone Vectors", "Limb 3D [B,3,32,13]", "P_bone", 2.9),
        ("Joint Motion", "Vel 3D [B,3,31,13]", "P_jmot", 5.4),
        ("Bone Motion", "B-Vel 3D [B,3,31,13]", "P_bmot", 7.9)
    ]

    for name, shape_txt, prob_txt, x_pos in streams:
        create_box(ax, x_pos, 4.6, 2.2, 0.7, name, shape_txt, bg=C_INPUT, border=C_BORDER_IN, fs_top=8.5, fs_bot=7.0)
        create_arrow(ax, x_pos + 1.1, 4.6, x_pos + 1.1, 3.9)
        create_box(ax, x_pos, 3.2, 2.2, 0.7, "AAGCN Backbone", f"3 Stages [48,96,150]\n{prob_txt} ∈ Δ²¹",
                   bg=C_GCN, border=C_BORDER_GC, fs_top=8.5, fs_bot=7.0)
        create_arrow(ax, x_pos + 1.1, 3.2, 5.25, 2.2, color=C_TEXT_MUTED)

    create_box(ax, 3.0, 1.4, 4.5, 0.8, "Four-Stream Softmax Late Fusion",
               "P_fused = (P_joint + P_bone + P_jmot + P_bmot) / 4\n(Graph Unified Consensus)",
               bg=C_FUSION, border=C_BORDER_FS)
    create_arrow(ax, 5.25, 1.4, 5.25, 0.95)

    create_box(ax, 3.5, 0.35, 3.5, 0.6, "Final Exercise Prediction", "argmax(P_fused) ∈ {1, ..., 22}",
               bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_4stream.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_4stream.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_fusion_4stream.png & .pdf")


# =============================================================================
# 10. SKELGYM-LITE (Dual-Stream Cross-Paradigm)
# =============================================================================
def draw_fusion_skelgym_lite():
    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=300)
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(4.0, 6.05, "SkelGym-Lite (Dual-Stream Cross-Paradigm Architecture)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=C_TEXT_DARK)

    # Branch 1: Transformer
    create_box(ax, 0.7, 4.5, 3.1, 0.7, "Sequence Mix Stream", "117-d Biomechanical Mix\nShape: [Batch, 32, 117]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 2.25, 4.5, 2.25, 3.8)
    create_box(ax, 0.7, 3.0, 3.1, 0.8, "Transformer Backbone",
               "4L Pre-LN MHSA (d=128)\nCaptures Global Temporal Phase\nOutput: P_seq ∈ Δ²¹",
               bg=C_TRANS, border=C_BORDER_TR)

    # Branch 2: AAGCN Bone
    create_box(ax, 4.2, 4.5, 3.1, 0.7, "Bone Kinematic Stream", "Directed Limb Vectors\nShape: [Batch, 3, 32, 13]",
               bg=C_INPUT, border=C_BORDER_IN)
    create_arrow(ax, 5.75, 4.5, 5.75, 3.8)
    create_box(ax, 4.2, 3.0, 3.1, 0.8, "AAGCN Bone Backbone",
               "Adaptive Graph Conv (V=13)\nCaptures Anatomical Segment Flow\nOutput: P_bone ∈ Δ²¹",
               bg=C_GCN, border=C_BORDER_GC)

    # Late Fusion
    create_arrow(ax, 2.25, 3.0, 3.6, 2.1)
    create_arrow(ax, 5.75, 3.0, 4.4, 2.1)

    create_box(ax, 2.0, 1.35, 4.0, 0.75, "SLSQP Soft Weighted Fusion",
               "P_final = w_seq · P_seq + w_bone · P_bone\nWeights calibrated on Validation NLL",
               bg=C_FUSION, border=C_BORDER_FS)
    create_arrow(ax, 4.0, 1.35, 4.0, 0.9)

    create_box(ax, 2.25, 0.35, 3.5, 0.55, "Ultra-Low Latency Prediction",
               "argmax(P_final) | 1.44 ms on Host CPU", bg=C_HEAD, border=C_BORDER_HD)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_skelgym_lite.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_skelgym_lite.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_fusion_skelgym_lite.png & .pdf")


# =============================================================================
# 11. SKELGYM-FULL (5-Stream Cross-Paradigm Ensemble & Consensus Decision)
# =============================================================================
def draw_fusion_skelgym_full():
    fig, ax = plt.subplots(figsize=(10.0, 7.6), dpi=300)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    ax.text(5.0, 7.25, "SkelGym-Full Cross-Paradigm Ensemble & Decision Engine", ha='center', va='center',
            fontsize=13.5, fontweight='bold', color=C_TEXT_DARK)

    # Upper Tier: 5 Constituent Expert Backbones
    experts = [
        ("Transformer", "Sequence Mix (117-d)", "Output: P_trans", 0.5, C_TRANS, C_BORDER_TR),
        ("AAGCN Joint", "Joint Rel 3D", "Output: P_joint", 2.4, C_GCN, C_BORDER_GC),
        ("AAGCN Bone", "Bone Vectors 3D", "Output: P_bone", 4.3, C_GCN, C_BORDER_GC),
        ("AAGCN J-Mot", "Joint Velocity", "Output: P_jmot", 6.2, C_GCN, C_BORDER_GC),
        ("AAGCN B-Mot", "Bone Velocity", "Output: P_bmot", 8.1, C_GCN, C_BORDER_GC)
    ]

    for name, desc, p_lbl, x_pos, bg, border in experts:
        create_box(ax, x_pos, 5.75, 1.6, 0.95, name, f"{desc}\n{p_lbl}",
                   bg=bg, border=border, fs_top=8.5, fs_bot=7.0)
        create_arrow(ax, x_pos + 0.8, 5.75, 5.0, 4.75, color=C_TEXT_MUTED)

    # Stage 1: Window-Level Weighted Soft Voting
    create_box(ax, 2.0, 3.8, 6.0, 0.95, "Stage 1: Window-Level Weighted Soft Voting",
               "P_win(w) = w₁·P_trans + w₂·P_joint + w₃·P_bone + w₄·P_jmot + w₅·P_bmot\nWeights w* calibrated via SLSQP simplex optimization on validation NLL",
               bg=C_FUSION, border=C_BORDER_FS, fs_top=9.5, fs_bot=7.8)

    create_arrow(ax, 5.0, 3.8, 5.0, 3.1)

    # Stage 2: Temporal Video Consensus Aggregation
    create_box(ax, 1.5, 1.8, 7.0, 1.3, "Stage 2: Temporal Video Consensus Aggregation",
               "Soft-pooling across K sliding windows over full video recording:\n"
               "P_vid(c) = ∑ₘ w_{vid, m} · [ (1 / K) ∑ₖ P_{m, c}(w_k) ]\n"
               "Acts as an expectation-based low-pass probability filter\n"
               "Eliminates tracking occlusions and static boundary pauses",
               bg=C_INPUT, border=C_BORDER_IN, fs_top=10.0, fs_bot=8.0)

    create_arrow(ax, 5.0, 1.8, 5.0, 1.15)

    # Final Output
    create_box(ax, 2.8, 0.45, 4.4, 0.70, "Final Video Exercise Classification",
               "ŷ_vid = argmax P̂_vid(c)  across 22 Classes\nVideo Consensus Benchmark: 79.11% ± 0.25%",
               bg=C_HEAD, border=C_BORDER_HD, fs_top=9.5, fs_bot=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_skelgym_full.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_fusion_skelgym_full.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_fusion_skelgym_full.png & .pdf")


# =============================================================================
# 12. STACKING META-LEARNER ARCHITECTURE
# =============================================================================
def draw_ensemble_stacking_diagram():
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=300)
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(4.75, 6.65, "Stacking Meta-Classifier Architecture (Level-0 & Level-1)", ha='center', va='center',
            fontsize=13, fontweight='bold', color=C_TEXT_DARK)

    # Level-0 Base Models (5 models)
    models = [
        ("Transformer", "Seq Mix 117-d", 0.5, C_TRANS, C_BORDER_TR),
        ("BiLSTM", "Seq Mix 117-d", 2.3, C_RECURR, C_BORDER_RC),
        ("AAGCN Joint", "Graph Coords", 4.1, C_GCN, C_BORDER_GC),
        ("AAGCN Bone", "Graph Limb", 5.9, C_GCN, C_BORDER_GC),
        ("AAGCN Motion", "Graph Velocity", 7.7, C_GCN, C_BORDER_GC)
    ]

    for name, desc, x_pos, bg, border in models:
        create_box(ax, x_pos, 5.3, 1.45, 0.9, name, f"{desc}\nOutput: P_m ∈ Δ²¹",
                   bg=bg, border=border, fs_top=8.5, fs_bot=6.8)
        create_arrow(ax, x_pos + 0.725, 5.3, 4.75, 4.3, color=C_TEXT_MUTED)

    # Meta-Features Formation
    create_box(ax, 1.8, 3.4, 5.9, 0.9, "Level-0 Meta-Feature Construction",
               "Concatenate base model probability vectors:\n"
               "X_meta = [P₁, P₂, P₃, P₄, P₅] ∈ ℝ^(Batch × 110)\n"
               "(5 models × 22 class probabilities = 110 features)",
               bg=C_FUSION, border=C_BORDER_FS, fs_top=9.5, fs_bot=7.8)

    create_arrow(ax, 4.75, 3.4, 4.75, 2.7)

    # Level-1 Meta-Learner
    create_box(ax, 1.8, 1.7, 5.9, 1.0, "Level-1 Meta-Classifier: Logistic Regression",
               "Multinomial Logistic Regression (L-BFGS solver, C=1.0)\n"
               "Learns optimal inter-model correlation & cross-entropy weights\n"
               "Fitted strictly on out-of-fold validation set predictions",
               bg=C_INPUT, border=C_BORDER_IN, fs_top=9.5, fs_bot=7.8)

    create_arrow(ax, 4.75, 1.7, 4.75, 1.1)

    # Final Output
    create_box(ax, 2.5, 0.35, 4.5, 0.75, "Final Meta-Ensemble Class Decision",
               "ŷ = argmax P_meta(c) ∈ {1, ..., 22}\nCaptures non-linear cross-paradigm consensus",
               bg=C_HEAD, border=C_BORDER_HD, fs_top=9.5, fs_bot=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_ensemble_stacking.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "model_ensemble_stacking.pdf"), bbox_inches='tight', facecolor='white')
    plt.close()
    print(" -> Saved: model_ensemble_stacking.png & .pdf")


# =============================================================================
# MAIN RUNNER
# =============================================================================
if __name__ == "__main__":
    print("Rendering individual clean architecture diagrams...")
    draw_lstm_diagram()
    draw_bilstm_diagram()
    draw_transformer_diagram()
    draw_stgcn_diagram()
    draw_aagcn_diagram()
    draw_dual_branch_diagram()
    draw_fusion_2stream()
    draw_fusion_3stream()
    draw_fusion_4stream()
    draw_fusion_skelgym_lite()
    draw_fusion_skelgym_full()
    draw_ensemble_stacking_diagram()
    print("\nAll 12 individual diagrams successfully generated and exported to paper/images!")

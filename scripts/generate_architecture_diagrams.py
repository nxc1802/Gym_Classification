"""
Script to generate ultra-high-resolution, publication-grade architectural diagrams
for SkelGym deep learning models and multi-stream fusion hierarchy.
Outputs vector PDF and 300+ DPI PNG files into paper/images/.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUTPUT_DIR = "/Volumes/WorkSpace/Project/Gym_Classification/paper/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Color Palette Definition (Elsevier / Nature / IEEE Academic Style)
# -------------------------------------------------------------
C_NAVY = "#1B365D"       # Deep primary blue
C_STEEL = "#2B6CB0"      # Mid blue for transformers
C_TEAL = "#2A9D8F"       # Teal for adaptive graph convolved blocks
C_CORAL = "#E76F51"      # Warm coral for classification heads
C_GOLD = "#D97706"       # Dark amber for attention & weights
C_PURPLE = "#6D597A"     # Purple for temporal / recurrent
C_SLATE = "#4A5568"      # Slate for inputs / data
C_LIGHT_BLUE = "#EBF8FF" # Background tint blue
C_LIGHT_TEAL = "#E6FFFA" # Background tint teal
C_LIGHT_CORAL = "#FFF5F5"# Background tint coral
C_LIGHT_GOLD = "#FEF3C7" # Background tint gold
C_LIGHT_GREY = "#F8FAFC" # General container background
C_BORDER = "#CBD5E0"     # Clean subtle border
C_DARK = "#2D3748"       # Main text color


def draw_box(ax, x, y, w, h, title, subtitle=None, bg=C_LIGHT_GREY, border=C_STEEL,
             lw=1.5, radius=0.015, title_color=C_DARK, subtitle_color="#4A5568",
             fontsize=9, sub_fontsize=7.5, fontweight='bold', zorder=2):
    """Draws a clean rounded rectangle box with title and subtitle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg, edgecolor=border, linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(box)
    
    if subtitle is None or subtitle == "":
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=title_color, zorder=zorder+1)
    else:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=title_color, zorder=zorder+1)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                fontsize=sub_fontsize, color=subtitle_color, zorder=zorder+1)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=C_SLATE, lw=1.5, style="-|>", rad=0.0, zorder=3):
    """Draws a crisp directional arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"{style},head_length=4.5,head_width=3.5",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow


# =============================================================================
# DIAGRAM 1: SINGLE MODEL MICRO-ARCHITECTURES
# =============================================================================
def generate_diagram_single_models():
    print("[1/2] Generating Single Model Architectures Diagram...")
    fig = plt.figure(figsize=(18, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Header / Super Title
    ax.text(9.0, 10.6, "SkelGym Lightweight Single-Model Micro-Architectures (~350K–400K Parameters)",
            ha='center', va='center', fontsize=16, fontweight='bold', color=C_NAVY)
    ax.text(9.0, 10.3, "Calibrated Custom Backbones Built from Scratch for Privacy-Preserving Real-Time Gym Exercise Classification",
            ha='center', va='center', fontsize=10.5, fontstyle='italic', color=C_SLATE)

    # -------------------------------------------------------------------------
    # PANEL A: SKELETAL TRANSFORMER (Left Column, x: 0.45 to 5.75)
    # -------------------------------------------------------------------------
    p_a = FancyBboxPatch((0.45, 0.4), 5.4, 9.6, boxstyle="round,pad=0,rounding_size=0.08",
                         facecolor="#FAFDFE", edgecolor="#BEE3F8", linewidth=1.5, zorder=1)
    ax.add_patch(p_a)
    ax.text(0.65, 9.75, "Panel A: Skeletal Transformer (Sequence Paradigm)",
            fontsize=12, fontweight='bold', color=C_NAVY, zorder=2)
    ax.text(0.65, 9.5, "Input: Dense Sequence Mix (117-d) | 399K Params | 12.50 MFLOPs | Latency: 0.42 ms",
            fontsize=8.5, color=C_STEEL, fontweight='bold', zorder=2)

    # Tensor Input
    draw_box(ax, 1.25, 8.75, 4.0, 0.52, "Sequence Input Tensor", "Shape: [Batch B, Time T=32, Feat D=117]",
             bg=C_LIGHT_BLUE, border=C_STEEL, fontsize=9.5, sub_fontsize=8)
    draw_arrow(ax, 3.25, 8.75, 3.25, 8.35)

    # Input LayerNorm & Linear Projection
    draw_box(ax, 1.25, 7.85, 4.0, 0.50, "Input LayerNorm(117) + Linear Proj(117 → 128)",
             "Feature standardization & d_model projection + Dropout(0.2)",
             bg="#FFFFFF", border=C_STEEL, fontsize=8.5, sub_fontsize=7.5)
    draw_arrow(ax, 3.25, 7.85, 3.25, 7.45)

    # Learnable Positional Encoding
    draw_box(ax, 1.25, 6.95, 4.0, 0.50, "+ Learnable Positional Encoding",
             "Param: [1, 128, 128] captures cyclic repetition phases",
             bg=C_LIGHT_GOLD, border=C_GOLD, fontsize=8.5, sub_fontsize=7.5)
    draw_arrow(ax, 3.25, 6.95, 3.25, 6.55)

    # 4x Transformer Block Container
    t_box = FancyBboxPatch((1.1, 3.0), 4.4, 3.5, boxstyle="round,pad=0,rounding_size=0.06",
                           facecolor="#F0F9FF", edgecolor=C_STEEL, linewidth=1.8, linestyle="--", zorder=2)
    ax.add_patch(t_box)
    ax.text(1.3, 6.25, "Stacked Pre-LN Transformer Layer (× 4 Layers)",
            fontsize=9.5, fontweight='bold', color=C_NAVY, zorder=3)
    ax.text(1.3, 6.05, "Pre-LN for smooth gradients; d_model=128, nhead=8, d_ff=192",
            fontsize=7.5, fontstyle='italic', color=C_SLATE, zorder=3)

    # Sub-block: LayerNorm & MHSA
    draw_box(ax, 1.5, 5.25, 3.8, 0.65, "LayerNorm(128) → Multi-Head Attention",
             "8 Heads (d_k = 16) | Q, K, V = Z·W | Residual (+)",
             bg="#FFFFFF", border=C_STEEL, fontsize=8.5, sub_fontsize=7.5)
    # Residual skip arrow for MHSA (neatly inside the container)
    draw_arrow(ax, 1.35, 5.85, 1.35, 5.05, color=C_STEEL, lw=1.2, rad=-0.3)
    ax.text(0.95, 5.45, "Skip\n(+)", fontsize=7, color=C_STEEL, fontweight='bold', ha='center', va='center')

    draw_arrow(ax, 3.4, 5.25, 3.4, 4.85)

    # Sub-block: LayerNorm & FFN
    draw_box(ax, 1.5, 4.15, 3.8, 0.65, "LayerNorm(128) → Feed-Forward Network",
             "Linear(128 → 192) → GELU → Linear(192 → 128) | Residual (+)",
             bg="#FFFFFF", border=C_STEEL, fontsize=8.5, sub_fontsize=7.5)
    # Residual skip arrow for FFN
    draw_arrow(ax, 1.35, 4.75, 1.35, 3.95, color=C_STEEL, lw=1.2, rad=-0.3)
    ax.text(0.95, 4.35, "Skip\n(+)", fontsize=7, color=C_STEEL, fontweight='bold', ha='center', va='center')

    # Post Transformer Norm
    draw_box(ax, 1.65, 3.2, 3.4, 0.45, "Final LayerNorm(128)", "Tensor: [Batch B, Time T=32, d_model=128]",
             bg="#FFFFFF", border=C_STEEL, fontsize=8.2, sub_fontsize=7.2)

    draw_arrow(ax, 3.25, 3.0, 3.25, 2.55)

    # Temporal GAP
    draw_box(ax, 1.25, 2.05, 4.0, 0.50, "Temporal Global Average Pooling (GAP)",
             "Collapses Time: ẑ = (1/T) ∑ Z_t  →  Shape: [B, 128]",
             bg=C_LIGHT_BLUE, border=C_STEEL, fontsize=8.5, sub_fontsize=7.5)
    draw_arrow(ax, 3.25, 2.05, 3.25, 1.65)

    # Two-Stage Classifier Head
    draw_box(ax, 1.25, 0.65, 4.0, 1.0, "Two-Stage Classifier Head (MLP)",
             "Linear(128 → 128) → GELU → LayerNorm(128) → Dropout(0.2)\n→ Linear(128 → 22 Classes) → Softmax (Δ²¹)",
             bg=C_LIGHT_CORAL, border=C_CORAL, fontsize=8.5, sub_fontsize=7.5)

    # -------------------------------------------------------------------------
    # PANEL B: ADAPTIVE SPATIAL-TEMPORAL GCN (Middle Column, x: 6.2 to 12.3)
    # -------------------------------------------------------------------------
    p_b = FancyBboxPatch((6.2, 0.4), 6.1, 9.6, boxstyle="round,pad=0,rounding_size=0.08",
                         facecolor="#F6FFFE", edgecolor="#81E6D9", linewidth=1.5, zorder=1)
    ax.add_patch(p_b)
    ax.text(6.4, 9.75, "Panel B: Attention-Enhanced AAGCN Micro-Architecture",
            fontsize=12, fontweight='bold', color="#134E4A", zorder=2)
    ax.text(6.4, 9.5, "Input: Graph Stream (Joint/Bone/Motion) | 378K Params | 101.43 MFLOPs | Latency: 0.96 ms",
            fontsize=8.5, color=C_TEAL, fontweight='bold', zorder=2)

    # Input Tensor
    draw_box(ax, 6.6, 8.75, 5.3, 0.52, "Kinematic Graph Input Tensor",
             "Shape: [B, C_in=3, T=32, V=13 Joints]  |  Normalized by Data BatchNorm2d",
             bg=C_LIGHT_TEAL, border=C_TEAL, fontsize=9.5, sub_fontsize=8)
    draw_arrow(ax, 9.25, 8.75, 9.25, 8.35)

    # Micro-Breakdown of Adaptive Graph Convolution
    ag_box = FancyBboxPatch((6.4, 4.95), 5.7, 3.35, boxstyle="round,pad=0,rounding_size=0.06",
                            facecolor="#FFFFFF", edgecolor=C_TEAL, linewidth=1.8, zorder=2)
    ax.add_patch(ag_box)
    ax.text(6.6, 8.05, "Adaptive Graph Convolution Module (Spatial GCN)",
            fontsize=10, fontweight='bold', color="#134E4A", zorder=3)
    ax.text(6.6, 7.85, "Unified Adjacency:  A_total = (A_topo ⊙ M) + B + C(X)",
            fontsize=8.5, fontweight='bold', color=C_CORAL, zorder=3)

    # 3 Parallel Branches of Adjacency
    draw_box(ax, 6.55, 6.35, 1.7, 1.35, "Branch 1:\nAnatomical",
             "A_topo ∈ ℝ¹³ˣ¹³\n(Biomechanic)\n⊙ Mask M\n(Learnable)",
             bg=C_LIGHT_GREY, border=C_SLATE, fontsize=8, sub_fontsize=7.2)

    draw_box(ax, 8.4, 6.35, 1.7, 1.35, "Branch 2:\nGlobal Graph",
             "Matrix B ∈ ℝ¹³ˣ¹³\n(Learnable free\npairwise synergy\ne.g., wrist-ankle)",
             bg=C_LIGHT_BLUE, border=C_STEEL, fontsize=8, sub_fontsize=7.2)

    draw_box(ax, 10.25, 6.35, 1.7, 1.35, "Branch 3:\nDynamic Attn",
             "C(X) = Softmax(\nθ(X)ᵀ·φ(X) / √d)\nInstance-specific\nself-attention",
             bg=C_LIGHT_GOLD, border=C_GOLD, fontsize=8, sub_fontsize=7.0)

    # Summing Junction
    ax.text(9.25, 6.0, "Summation ⨁ : A_total ∈ ℝᴮ ˣ ¹³ ˣ ¹³",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_NAVY, zorder=3)

    # Feature Message Passing & Projection
    draw_box(ax, 6.6, 5.1, 5.3, 0.65, "Spatial Message Passing & Channel Projection",
             "Einstein Sum: X_g = A_total · X  →  Conv2d(1×1, C_in → C_out) → BatchNorm2d → GELU",
             bg=C_LIGHT_TEAL, border=C_TEAL, fontsize=8.5, sub_fontsize=7.5)

    draw_arrow(ax, 9.25, 4.95, 9.25, 4.55)

    # 3-Stage Backbone Flow
    backbone_box = FancyBboxPatch((6.4, 2.05), 5.7, 2.45, boxstyle="round,pad=0,rounding_size=0.06",
                                  facecolor="#E6FFFA", edgecolor=C_TEAL, linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(backbone_box)
    ax.text(6.6, 4.25, "Calibrated 3-Stage Backbone:  Channels [48, 96, 150]  (~355K Params)",
            fontsize=9.2, fontweight='bold', color="#134E4A", zorder=3)

    # Stage 1
    draw_box(ax, 6.55, 3.4, 5.4, 0.65, "Stage 1 Block (Channels: 3 → 48, Stride: 1)",
             "AdaptiveGraphConv(3→48) → MS-TCN(9×1, s=1, drop=0.2) + Residual(1×1) → GELU",
             bg="#FFFFFF", border=C_TEAL, fontsize=8, sub_fontsize=7.2)
    # Stage 2
    draw_box(ax, 6.55, 2.7, 5.4, 0.60, "Stage 2 Block (Channels: 48 → 96, Stride: 1)",
             "AdaptiveGraphConv(48→96) → MS-TCN(9×1, s=1, drop=0.2) + Residual(1×1) → GELU",
             bg="#FFFFFF", border=C_TEAL, fontsize=8, sub_fontsize=7.2)
    # Stage 3
    draw_box(ax, 6.55, 2.15, 5.4, 0.50, "Stage 3 Block (Channels: 96 → 150, Stride: 2)",
             "Downsamples Time T: 32 → 16 | Latent Channels = 150 | Output: [B, 150, 16, 13]",
             bg="#FFFFFF", border=C_TEAL, fontsize=8, sub_fontsize=7.2)

    draw_arrow(ax, 9.25, 2.05, 9.25, 1.65)

    # Spatial-Temporal GAP & Classifier Head
    draw_box(ax, 6.6, 0.65, 5.3, 1.0, "Spatial-Temporal GAP & Two-Stage Classifier Head",
             "AdaptiveAvgPool2d(1×1) collapses Time & Joints: [B, 150, 16, 13] → [B, 150]\nLinear(150 → 150) → GELU → LayerNorm(150) → Dropout(0.2) → Linear(150 → 22 Classes)",
             bg=C_LIGHT_CORAL, border=C_CORAL, fontsize=8.5, sub_fontsize=7.5)

    # -------------------------------------------------------------------------
    # PANEL C: COMPARATIVE BASELINES (Right Column, x: 12.8 to 17.55)
    # -------------------------------------------------------------------------
    p_c = FancyBboxPatch((12.8, 0.4), 4.75, 9.6, boxstyle="round,pad=0,rounding_size=0.08",
                         facecolor="#FDFBFC", edgecolor="#CBD5E0", linewidth=1.5, zorder=1)
    ax.add_patch(p_c)
    ax.text(13.0, 9.75, "Panel C: Comparative Baselines",
            fontsize=12, fontweight='bold', color=C_NAVY, zorder=2)
    ax.text(13.0, 9.5, "Parameter-Calibrated (~350K) Reference Architectures",
            fontsize=8.5, color=C_SLATE, fontstyle='italic', zorder=2)

    # Top Subpanel: ST-GCN (Yan et al. 2018)
    st_box = FancyBboxPatch((13.0, 5.25), 4.35, 4.15, boxstyle="round,pad=0,rounding_size=0.06",
                            facecolor="#FFFFFF", edgecolor=C_SLATE, linewidth=1.2, zorder=2)
    ax.add_patch(st_box)
    ax.text(13.2, 9.15, "ST-GCN Baseline (Yan et al. 2018)",
            fontsize=10, fontweight='bold', color=C_NAVY, zorder=3)
    ax.text(13.2, 8.95, "Rigid Anatomical Adjacency | 365K Params",
            fontsize=7.8, color=C_SLATE, zorder=3)

    draw_box(ax, 13.2, 7.95, 3.95, 0.85, "Spatial Graph Conv (Rigid A_phys)",
             "Fixed adjacency A_norm = D⁻¹/²(A+I)D⁻¹/²\n⊙ Mask M (Only adjacent limbs communicate!)\nConv2d(1×1) → BatchNorm2d → ReLU",
             bg=C_LIGHT_GREY, border=C_SLATE, fontsize=8, sub_fontsize=7.0)
    draw_arrow(ax, 15.17, 7.95, 15.17, 7.55)

    draw_box(ax, 13.2, 6.75, 3.95, 0.75, "Temporal Conv & Residual Shortcut",
             "Conv2d(9×1, stride s) → BatchNorm2d → Dropout\nResidual: Conv2d(1×1, stride s) + BN → ReLU",
             bg=C_LIGHT_GREY, border=C_SLATE, fontsize=8, sub_fontsize=7.0)
    draw_arrow(ax, 15.17, 6.75, 15.17, 6.35)

    draw_box(ax, 13.2, 5.45, 3.95, 0.85, "Deficiency in Heavy Lifting",
             "Critical kinetic synergies (e.g., foot drive to\nbarbell) lack physical edges. Window Acc: 43.57%\n(Surpassed by AAGCN by +17.26% on baseline)",
             bg=C_LIGHT_CORAL, border=C_CORAL, fontsize=7.8, sub_fontsize=6.8)

    # Bottom Subpanel: BiLSTM Sequence Baseline
    lstm_box = FancyBboxPatch((13.0, 0.65), 4.35, 4.35, boxstyle="round,pad=0,rounding_size=0.06",
                             facecolor="#FFFFFF", edgecolor=C_PURPLE, linewidth=1.2, zorder=2)
    ax.add_patch(lstm_box)
    ax.text(13.2, 4.75, "BiLSTM Baseline (Recurrent Paradigm)",
            fontsize=10, fontweight='bold', color=C_PURPLE, zorder=3)
    ax.text(13.2, 4.55, "Bidirectional Sequence Encoder | 367K Params",
            fontsize=7.8, color=C_SLATE, zorder=3)

    draw_box(ax, 13.2, 3.75, 3.95, 0.65, "Input Projection & Scaling",
             "Input: [B, T=32, D=117] → LayerNorm",
             bg=C_LIGHT_GREY, border=C_PURPLE, fontsize=8, sub_fontsize=7.2)
    draw_arrow(ax, 15.17, 3.75, 15.17, 3.35)

    draw_box(ax, 13.2, 2.35, 3.95, 0.95, "2-Layer Bidirectional LSTM",
             "hidden_dim = 96 (Forward: 96 + Backward: 96)\nTotal Hidden = 192 | Dropout = 0.3\nCaptures bidirectional temporal trajectory\nOutput: [B, T=32, 192]",
             bg="#FAF5FF", border=C_PURPLE, fontsize=8, sub_fontsize=7.0)
    draw_arrow(ax, 15.17, 2.35, 15.17, 1.95)

    draw_box(ax, 13.2, 0.85, 3.95, 1.05, "Temporal Pooling & MLP Head",
             "Global Average Pooling (GAP) over Time\n→ [B, 192] → Dropout(0.3) → Linear(192 → 64)\n→ ReLU → Dropout(0.3) → Linear(64 → 22)\nWindow Acc: 57.61% | Video Acc: 66.09%",
             bg=C_LIGHT_CORAL, border=C_CORAL, fontsize=8, sub_fontsize=7.0)

    # Footnote / Design Philosophy
    ax.text(9.0, 0.15, "Note: All backbones are constrained to ~350K–400K parameters to ensure rigorous, fair comparative benchmarking on edge devices.",
            ha='center', va='center', fontsize=8.5, color=C_SLATE, fontstyle='italic')

    # Save outputs
    png_path = os.path.join(OUTPUT_DIR, "figure_model_architectures_single.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_model_architectures_single.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f" -> Successfully saved: {png_path}")
    print(f" -> Successfully saved: {pdf_path}")


# =============================================================================
# DIAGRAM 2: MULTI-STREAM FUSION HIERARCHY & ENSEMBLE ARCHITECTURE
# =============================================================================
def generate_diagram_fusion_hierarchy():
    print("[2/2] Generating Multi-Stream Fusion Hierarchy Diagram...")
    fig = plt.figure(figsize=(19, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Header / Super Title
    ax.text(9.5, 11.6, "SkelGym Multi-Stream Kinematic Fusion Hierarchy & Dual-Stage Decision Engine",
            ha='center', va='center', fontsize=16, fontweight='bold', color=C_NAVY)
    ax.text(9.5, 11.3, "From Orthogonal Skeleton Streams to Cross-Paradigm SLSQP Ensembles and Video-Consensus Soft Pooling",
            ha='center', va='center', fontsize=10.5, fontstyle='italic', color=C_SLATE)

    # -------------------------------------------------------------------------
    # LEVEL 1: INPUT SKELETON REPRESENTATION & 5 STREAMS (y: 9.2 to 10.9)
    # -------------------------------------------------------------------------
    lvl1_box = FancyBboxPatch((0.5, 9.15), 18.0, 1.85, boxstyle="round,pad=0,rounding_size=0.06",
                             facecolor="#FAFDFE", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(lvl1_box)
    ax.text(0.7, 10.75, "LEVEL 1: Orthogonal Stream Generation from MediaPipe Pose (T=32 Frames, V=13 Load-Bearing Joints)",
            fontsize=11.5, fontweight='bold', color=C_NAVY, zorder=2)

    # Stream 1: Joint Position
    draw_box(ax, 0.7, 9.35, 3.3, 1.15, "Stream 1: Joint Position (Rel 3D)",
             "X_joint(c,t,v) = p̃(t, v) ∈ ℝ³ ˣ ³² ˣ ¹³\nPelvis-centered relative 3D coords\nEncodes absolute spatial posture",
             bg=C_LIGHT_BLUE, border=C_STEEL, fontsize=8.5, sub_fontsize=7.2)

    # Stream 2: Bone Vectors
    draw_box(ax, 4.3, 9.35, 3.3, 1.15, "Stream 2: Bone Vector (Bone 3D)",
             "X_bone(c,t,v) = p̃(target) - p̃(source)\nDirected limb segment vectors\nSpatial variance & length invariant",
             bg=C_LIGHT_TEAL, border=C_TEAL, fontsize=8.5, sub_fontsize=7.2)

    # Stream 3: Joint Motion
    draw_box(ax, 7.9, 9.35, 3.3, 1.15, "Stream 3: Joint Motion (J-Mot)",
             "X_jmot(c,t,v) = X_joint(t+1) - X_joint(t)\nFirst-order joint velocity (Δp)\nCaptures instantaneous limb speed",
             bg="#FAF5FF", border=C_PURPLE, fontsize=8.5, sub_fontsize=7.2)

    # Stream 4: Bone Motion
    draw_box(ax, 11.5, 9.35, 3.3, 1.15, "Stream 4: Bone Motion (B-Mot)",
             "X_bmot(c,t,v) = X_bone(t+1) - X_bone(t)\nRate of change of bone vectors (Δb)\nEncodes rotational joint velocity",
             bg=C_LIGHT_GOLD, border=C_GOLD, fontsize=8.5, sub_fontsize=7.2)

    # Stream 5: Dense Sequence Mix
    draw_box(ax, 15.1, 9.35, 3.2, 1.15, "Stream 5: Sequence Mix (117-d)",
             "X_mix ∈ ℝ³² ˣ ¹¹⁷\n39 Joints × 3 Coordinates\nGlobal spatio-temporal flattened stream",
             bg=C_LIGHT_CORAL, border=C_CORAL, fontsize=8.5, sub_fontsize=7.2)

    # Connect Level 1 to Level 2
    for x_c in [2.35, 5.95, 9.55, 13.15, 16.7]:
        draw_arrow(ax, x_c, 9.35, x_c, 8.85)

    # -------------------------------------------------------------------------
    # LEVEL 2: CONSTITUENT BACKBONE EXPERTS (y: 7.4 to 8.85)
    # -------------------------------------------------------------------------
    lvl2_box = FancyBboxPatch((0.5, 7.35), 18.0, 1.5, boxstyle="round,pad=0,rounding_size=0.06",
                             facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(lvl2_box)
    ax.text(0.7, 8.65, "LEVEL 2: Single-Stream Expert Models (Trained from Scratch, ~350K–400K Parameters Each)",
            fontsize=11.5, fontweight='bold', color=C_NAVY, zorder=2)

    # Expert 1
    draw_box(ax, 0.7, 7.5, 3.3, 0.95, "AAGCN Joint Expert",
             "Adaptive Graph Conv (V=13)\nWindow Acc: 61.72% | Video: 69.96%\nParams: 378K | FLOPs: 101.4M",
             bg="#FFFFFF", border=C_STEEL, fontsize=8.5, sub_fontsize=7.2)

    # Expert 2
    draw_box(ax, 4.3, 7.5, 3.3, 0.95, "AAGCN Bone Expert (Strongest Graph)",
             "Adaptive Graph Conv (V=13)\nWindow Acc: 65.84% | Video: 72.96%\nParams: 378K | FLOPs: 101.4M",
             bg="#E6FFFA", border=C_TEAL, fontsize=8.5, sub_fontsize=7.2)

    # Expert 3
    draw_box(ax, 7.9, 7.5, 3.3, 0.95, "AAGCN Joint-Motion Expert",
             "Adaptive Graph Conv (V=13)\nWindow Acc: 46.81% | Video: 64.38%\nParams: 378K | FLOPs: 101.4M",
             bg="#FFFFFF", border=C_PURPLE, fontsize=8.5, sub_fontsize=7.2)

    # Expert 4
    draw_box(ax, 11.5, 7.5, 3.3, 0.95, "AAGCN Bone-Motion Expert",
             "Adaptive Graph Conv (V=13)\nWindow Acc: 57.49% | Video: 79.83%\nParams: 378K | FLOPs: 101.4M",
             bg="#FFFFFF", border=C_GOLD, fontsize=8.5, sub_fontsize=7.2)

    # Expert 5
    draw_box(ax, 15.1, 7.5, 3.2, 0.95, "Transformer Mix Expert",
             "Pre-LN MHSA (4L, 8H, d=128)\nWindow Acc: 64.53% | Video: 72.39%\nParams: 399K | FLOPs: 12.5M",
             bg="#FFF5F5", border=C_CORAL, fontsize=8.5, sub_fontsize=7.2)

    # -------------------------------------------------------------------------
    # LEVEL 3: FUSION ARCHITECTURES COMPARISON (y: 4.1 to 6.9)
    # -------------------------------------------------------------------------
    lvl3_box = FancyBboxPatch((0.5, 4.05), 18.0, 3.0, boxstyle="round,pad=0,rounding_size=0.06",
                             facecolor="#FFFFFF", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(lvl3_box)
    ax.text(0.7, 6.8, "LEVEL 3: Hierarchical Multi-Stream Fusion Paradigms & Benchmark Configurations",
            fontsize=11.5, fontweight='bold', color=C_NAVY, zorder=2)

    # Fusion Paradigm 1: 2-Stream AAGCN
    p1 = FancyBboxPatch((0.7, 4.25), 3.9, 2.35, boxstyle="round,pad=0,rounding_size=0.05",
                        facecolor="#F7FAFC", edgecolor=C_STEEL, linewidth=1.4, zorder=2)
    ax.add_patch(p1)
    ax.text(2.65, 6.35, "Config 1: Two-Stream AAGCN", ha='center', fontsize=9.5, fontweight='bold', color=C_NAVY)
    ax.text(2.65, 6.10, "Static Kinematics Late Fusion", ha='center', fontsize=8, fontstyle='italic', color=C_SLATE)
    ax.text(2.65, 5.45, "Streams: Joint (3D) + Bone (3D)\n\n• Combines position & segment angles\n• Window Acc: 66.61% | Video: 73.82%\n• Params: 756K | FLOPs: 202.8 MFLOPs\n• Host CPU Latency: 1.97 ms",
            ha='center', va='center', fontsize=7.8, color=C_DARK)

    # Fusion Paradigm 2: 4-Stream AAGCN
    p2 = FancyBboxPatch((4.9, 4.25), 4.1, 2.35, boxstyle="round,pad=0,rounding_size=0.05",
                        facecolor="#F0FFF4", edgecolor=C_TEAL, linewidth=1.4, zorder=2)
    ax.add_patch(p2)
    ax.text(6.95, 6.35, "Config 2: Four-Stream AAGCN", ha='center', fontsize=9.5, fontweight='bold', color="#134E4A")
    ax.text(6.95, 6.10, "Complete Unified Graph Paradigm", ha='center', fontsize=8, fontstyle='italic', color=C_SLATE)
    ax.text(6.95, 5.45, "Streams: Joint + Bone + J-Mot + B-Mot\n\n• Spatial posture + dynamic velocities\n• Window Acc: 68.78% ± 1.35%\n• Video Acc: 77.25% ± 1.55% (F1: 0.7678)\n• Params: 1.51M | FLOPs: 405.7 MFLOPs\n• Host CPU Latency: 3.84 ms",
            ha='center', va='center', fontsize=7.8, color=C_DARK)

    # Fusion Paradigm 3: SkelGym-Lite (Edge Pareto Hero)
    p3 = FancyBboxPatch((9.3, 4.25), 4.3, 2.35, boxstyle="round,pad=0,rounding_size=0.05",
                        facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=2.0, zorder=2)
    ax.add_patch(p3)
    ax.text(11.45, 6.35, "★ Config 3: SkelGym-Lite (Edge-Optimal)", ha='center', fontsize=10, fontweight='bold', color="#744210")
    ax.text(11.45, 6.10, "Cross-Paradigm Minimal Hybrid Ensemble", ha='center', fontsize=8, fontstyle='italic', color="#975A16")
    ax.text(11.45, 5.45, "Streams: Transformer Mix + AAGCN Bone\n\n• Global Self-Attention + Local Bone Topology\n• Window Acc: 69.66% ± 0.71%\n• Video Acc: 76.68% ± 0.25% (F1: 0.7582)\n• Params: 777K | FLOPs: 113.93 MFLOPs\n• Host CPU Latency: 1.44 ms (CUDA: 0.19 ms)\n• Optimal Pareto efficiency for mobile/edge!",
            ha='center', va='center', fontsize=7.8, color="#5F370E")

    # Fusion Paradigm 4: SkelGym-Full (Peak SOTA Benchmark)
    p4 = FancyBboxPatch((13.9, 4.25), 4.4, 2.35, boxstyle="round,pad=0,rounding_size=0.05",
                        facecolor="#FFF5F5", edgecolor=C_CORAL, linewidth=2.0, zorder=2)
    ax.add_patch(p4)
    ax.text(16.1, 6.35, "★ Config 4: SkelGym-Full (SOTA Ensemble)", ha='center', fontsize=10, fontweight='bold', color="#9B2C2C")
    ax.text(16.1, 6.10, "5-Stream Cross-Paradigm Late Fusion", ha='center', fontsize=8, fontstyle='italic', color="#C53030")
    ax.text(16.1, 5.45, "Streams: Transformer Mix + 4-Stream AAGCN\n\n• Unified Sequence & 4-Stream Graph Experts\n• Window Acc: 69.74% ± 1.04% (F1: 0.6882)\n• Video Acc: 79.11% ± 0.25% (F1: 0.7834)\n• Baseline Seed Video Acc: 78.97% (F1: 0.7765)\n• Params: 1.91M | FLOPs: 418.21 MFLOPs\n• Host CPU Latency: 4.33 ms (CUDA: 0.54 ms)",
            ha='center', va='center', fontsize=7.8, color="#742A2A")

    # Connect Level 2 to Level 3 neatly
    draw_arrow(ax, 2.35, 7.5, 2.65, 6.6, color=C_STEEL)
    draw_arrow(ax, 5.95, 7.5, 6.95, 6.6, color=C_TEAL)
    draw_arrow(ax, 9.55, 7.5, 6.95, 6.6, color=C_PURPLE)
    draw_arrow(ax, 13.15, 7.5, 6.95, 6.6, color=C_GOLD)
    draw_arrow(ax, 16.7, 7.5, 11.45, 6.6, color=C_CORAL)
    draw_arrow(ax, 5.95, 7.5, 11.45, 6.6, color=C_TEAL)
    draw_arrow(ax, 16.7, 7.5, 16.1, 6.6, color=C_CORAL)
    draw_arrow(ax, 6.95, 7.5, 16.1, 6.6, color=C_TEAL)

    # -------------------------------------------------------------------------
    # LEVEL 4: SLSQP OPTIMIZATION & TEMPORAL DECISION ENGINE (y: 0.5 to 3.7)
    # -------------------------------------------------------------------------
    lvl4_box = FancyBboxPatch((0.5, 0.45), 18.0, 3.25, boxstyle="round,pad=0,rounding_size=0.06",
                             facecolor="#FAFDFE", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(lvl4_box)
    ax.text(0.7, 3.45, "LEVEL 4: Validation-Calibrated SLSQP Simplex Optimization & Dual-Stage Decision Engine",
            fontsize=11.5, fontweight='bold', color=C_NAVY, zorder=2)

    # SLSQP Formulation Box
    draw_box(ax, 0.7, 0.75, 4.8, 2.45, "SLSQP Constrained Optimization",
             "Calibrated strictly on Validation Partition (Zero Leakage)\n\n"
             "Minimizes Negative Log-Likelihood (NLL):\n"
             "  w* = argmin_w - (1/N_val) ∑ log( ∑ w_m · P_{m, y_i} )\n"
             "  subject to:  ∑ w_m = 1,  w_m ≥ 0  (Probability Simplex Δ⁴)\n\n"
             "Empirical SLSQP Stream Allocation:\n"
             "• AAGCN Bone: 62.38% ± 11.58% (Primary Driver)\n"
             "• Transformer Mix: 26.58% ± 6.51% (Sequence Complement)\n"
             "• Bone-Motion: 14.54% | Joint-Motion: 2.86% | Joint: ~0%",
             bg="#FFFFFF", border=C_NAVY, fontsize=9, sub_fontsize=7.5)

    draw_arrow(ax, 5.5, 1.95, 6.2, 1.95, color=C_NAVY, lw=2.0)

    # Dual Decision Pipeline: Window-Level vs Video-Level
    # Stage 1: Window Voting
    draw_box(ax, 6.2, 1.35, 4.8, 1.85, "Stage 1: Window-Level Weighted Soft Voting",
             "P_win(w) = ∑_{m=1}⁵ w_m · P_m(w)\n\n"
             "• Evaluated per 32-frame sliding window (1.07s)\n"
             "• Captures fine-grained instantaneous phase kinetics\n"
             "• Resolves rapid intra-repetition motion transitions\n"
             "• Window Accuracy: 69.74% ± 1.04% (F1: 0.6882)",
             bg=C_LIGHT_BLUE, border=C_STEEL, fontsize=9, sub_fontsize=7.5)

    draw_arrow(ax, 11.0, 2.25, 11.7, 2.25, color=C_STEEL, lw=2.0)

    # Stage 2: Video Consensus Aggregation
    draw_box(ax, 11.7, 0.75, 6.5, 2.45, "Stage 2: Temporal Video Consensus Aggregation",
             "Temporal Soft-Pooling over K_v sliding windows:\n"
             "  P̂_vid(c) = ∑_{m=1}⁵ w_{vid, m}* [ (1/K_v) ∑_{k=1}^{K_v} P_{m,c}(w_{v,k}) ]\n"
             "  ŷ_vid = argmax_{c} P̂_vid(c)   over 22 Resistance Exercises\n\n"
             "Biomechanical & Statistical Benefits:\n"
             "• Low-pass expectation filter attenuates tracking noise & occlusion spikes\n"
             "• Cancels out non-cyclic boundary states (un-racking weights, pauses)\n"
             "• Yields consistent +9.37% surge over window accuracy!\n"
             "• Video Consensus Accuracy: 79.11% ± 0.25% (F1: 0.7834)",
             bg=C_LIGHT_TEAL, border=C_TEAL, fontsize=9.2, sub_fontsize=7.6)

    # Save outputs
    png_path = os.path.join(OUTPUT_DIR, "figure_model_fusion_hierarchy.png")
    pdf_path = os.path.join(OUTPUT_DIR, "figure_model_fusion_hierarchy.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f" -> Successfully saved: {png_path}")
    print(f" -> Successfully saved: {pdf_path}")


if __name__ == "__main__":
    generate_diagram_single_models()
    generate_diagram_fusion_hierarchy()
    print("\nAll architecture diagrams have been successfully rendered and exported to paper/images!")

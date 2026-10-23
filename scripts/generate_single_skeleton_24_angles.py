#!/usr/bin/env python3
r"""
SkelGym Kinematic Topology Generator: 24 Biomechanical Angles on Single Skeleton
================================================================================
Generates a clean, publication-quality, standalone figure of the human skeleton
with 24 domain-engineered kinematic angles (MediaPipe 33-landmark pose graph):
  - 14 Limb & Axial Segment Elevation Angles (\theta_1 ... \theta_14, Cyan/Blue)
  - 10 Joint Articulation & Posture Angles (\psi_15 ... \psi_24, Orange)

Outputs:
  - paper/images/24_angles_single_skeleton.png (and .pdf): Pure human skeleton
  - paper/images/24_angles_single_skeleton_legend.png (and .pdf): With clean 1-line legend
All text/mathematical definitions are provided separately as LaTeX/Markdown text.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# 1. CANONICAL MEDIAPIPE 33 SKELETAL 2D/3D LANDMARKS
# ============================================================================
LM = {
    0:  np.array([422.9, 146.8]), # Nose
    1:  np.array([452.2, 100.5]),
    2:  np.array([472.1, 100.6]),
    3:  np.array([491.5, 101.0]),
    4:  np.array([394.7,  99.5]),
    5:  np.array([375.1,  99.0]),
    6:  np.array([355.2,  99.0]),
    7:  np.array([522.2, 124.7]),
    8:  np.array([330.6, 122.1]),
    9:  np.array([451.2, 185.8]),
    10: np.array([396.7, 185.7]),
    11: np.array([558.0, 279.0]), # Left Shoulder
    12: np.array([288.4, 279.1]), # Right Shoulder
    13: np.array([645.9, 369.8]), # Left Elbow
    14: np.array([196.4, 364.4]), # Right Elbow
    15: np.array([732.5, 285.8]), # Left Wrist
    16: np.array([116.5, 284.5]), # Right Wrist
    17: np.array([805.6, 323.6]),
    18: np.array([ 38.0, 329.0]),
    19: np.array([781.2, 264.1]),
    20: np.array([ 63.7, 265.6]),
    21: np.array([733.9, 329.1]),
    22: np.array([113.8, 327.9]),
    23: np.array([495.2, 513.7]), # Left Hip
    24: np.array([347.0, 514.2]), # Right Hip
    25: np.array([518.7, 715.3]), # Left Knee
    26: np.array([323.5, 715.6]), # Right Knee
    27: np.array([538.3, 893.1]), # Left Ankle
    28: np.array([303.9, 894.2]), # Right Ankle
    29: np.array([552.0, 928.0]), # Left Heel
    30: np.array([290.0, 930.0]), # Right Heel
    31: np.array([580.0, 955.0]), # Left Toe
    32: np.array([262.0, 958.0]), # Right Toe
}

mid_shoulder = (LM[11] + LM[12]) / 2.0
mid_hip = (LM[23] + LM[24]) / 2.0

# ============================================================================
# 2. COLOR PALETTE FOR ACADEMIC PUBLICATION
# ============================================================================
COLOR_ELEV = "#0284C7"       # Vibrant Sky Blue (14 Elevation Angles)
COLOR_ELEV_FILL = "#38BDF8"
COLOR_ELEV_BG = "#F0F9FF"
COLOR_ELEV_BORDER = "#0284C7"

COLOR_JOINT = "#EA580C"      # Vibrant Deep Orange (10 Joint Articulations)
COLOR_JOINT_FILL = "#FB923C"
COLOR_JOINT_BG = "#FFF7ED"
COLOR_JOINT_BORDER = "#EA580C"

COLOR_BONE = "#94A3B8"
COLOR_ACTIVE_BONE = "#2563EB"
COLOR_REF_LINE = "#64748B"

# ============================================================================
# 3. DRAWING PRIMITIVES
# ============================================================================
def draw_arc(ax, center, radius, theta_start, theta_end, edge_color, fill_color, lw=2.2, alpha=0.35, zorder=10):
    """Draws a smooth circular sector representing the measured kinematic angle."""
    diff = (theta_end - theta_start) % (2 * np.pi)
    if diff > np.pi:
        diff = diff - 2 * np.pi
    n_pts = 35
    angles = np.linspace(theta_start, theta_start + diff, n_pts)
    arc_x = center[0] + radius * np.cos(angles)
    arc_y = center[1] + radius * np.sin(angles)
    ax.plot(arc_x, arc_y, color=edge_color, linewidth=lw, zorder=zorder)
    poly_x = np.concatenate([[center[0]], arc_x, [center[0]]])
    poly_y = np.concatenate([[center[1]], arc_y, [center[1]]])
    ax.fill(poly_x, poly_y, color=fill_color, alpha=alpha, zorder=zorder-2)

def draw_badge(ax, target_pt, badge_pos, text, color, bg_color, border_color, 
               fontsize=13.5, leader=True, zorder=15):
    """Draws a high-visibility badge with a dotted leader line pointing to the target angle."""
    if leader and target_pt is not None:
        ax.plot([target_pt[0], badge_pos[0]], [target_pt[1], badge_pos[1]],
                color=border_color, linewidth=1.3, linestyle=":", alpha=0.75, zorder=zorder-1)
        ax.plot(target_pt[0], target_pt[1], marker="o", markersize=3.8, color=border_color, zorder=zorder)
    
    ax.text(
        badge_pos[0], badge_pos[1], text,
        fontsize=fontsize, fontweight="bold", color=color,
        ha="center", va="center",
        bbox=dict(
            boxstyle="round,pad=0.38,rounding_size=0.35",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.7,
            alpha=0.99
        ),
        zorder=zorder
    )

def render_skeleton_anatomy(ax, badge_fs=13.5):
    """Renders the anatomical silhouette, bones, and all 24 kinematic badges."""
    p11, p12, p13, p14 = LM[11], LM[12], LM[13], LM[14]
    p15, p16 = LM[15], LM[16]
    p23, p24, p25, p26 = LM[23], LM[24], LM[25], LM[26]
    p27, p28 = LM[27], LM[28]

    # Anatomical silhouette
    head = patches.Ellipse((LM[0][0], LM[0][1] - 18), width=108, height=138, 
                           facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.6, zorder=1)
    ax.add_patch(head)

    torso_pts = [
        p11 + np.array([28, -15]), p11 + np.array([28, 25]),
        p23 + np.array([28, 10]), p24 + np.array([-28, 10]),
        p12 + np.array([-28, 25]), p12 + np.array([-28, -15])
    ]
    ax.add_patch(patches.Polygon(torso_pts, closed=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.4, zorder=1))

    for seg in [(p11, p13), (p13, p15), (p12, p14), (p14, p16),
                (p23, p25), (p25, p27), (p24, p26), (p26, p28)]:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                color="#F1F5F9", linewidth=30, solid_capstyle="round", zorder=1)

    # Base skeleton bones
    bones = [
        (11, 12), (12, 24), (24, 23), (23, 11),
        (27, 29), (29, 31), (27, 31),
        (28, 30), (30, 32), (28, 32),
        (15, 17), (15, 19), (15, 21), (17, 19),
        (16, 18), (16, 20), (16, 22), (18, 20),
    ]
    for b1, b2 in bones:
        ax.plot([LM[b1][0], LM[b2][0]], [LM[b1][1], LM[b2][1]],
                color=COLOR_BONE, linewidth=3.8, solid_capstyle="round", zorder=3)

    # Kinematic axes (orange dashed)
    ax.plot([LM[0][0], mid_shoulder[0]], [LM[0][1], mid_shoulder[1]], 
            color="#EA580C", linewidth=3.0, linestyle="--", zorder=4)
    ax.plot([mid_shoulder[0], mid_hip[0]], [mid_shoulder[1], mid_hip[1]], 
            color="#EA580C", linewidth=3.0, linestyle="--", zorder=4)
    ax.plot([mid_shoulder[0], p13[0]], [mid_shoulder[1], p13[1]], 
            color="#EA580C", linewidth=2.6, linestyle="--", zorder=4)
    ax.plot([mid_shoulder[0], p14[0]], [mid_shoulder[1], p14[1]], 
            color="#EA580C", linewidth=2.6, linestyle="--", zorder=4)

    # Active limb bones (royal blue)
    for b1, b2 in [(p11, p13), (p13, p15), (p12, p14), (p14, p16),
                   (p23, p25), (p25, p27), (p24, p26), (p26, p28)]:
        ax.plot([b1[0], b2[0]], [b1[1], b2[1]], color=COLOR_ACTIVE_BONE, linewidth=5.2, solid_capstyle="round", zorder=5)

    # Joint nodes
    for idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        pt = LM[idx]
        ax.plot(pt[0], pt[1], marker="o", markersize=10, color="#1E293B", zorder=7)
        ax.plot(pt[0], pt[1], marker="o", markersize=5.5, color="#FFFFFF", zorder=8)

    ax.plot(mid_shoulder[0], mid_shoulder[1], marker="o", markersize=9.5, color="#EA580C", zorder=7)
    ax.plot(mid_shoulder[0], mid_shoulder[1], marker="o", markersize=5.0, color="#FEF3C7", zorder=8)
    ax.plot(mid_hip[0], mid_hip[1], marker="o", markersize=9.5, color="#EA580C", zorder=7)
    ax.plot(mid_hip[0], mid_hip[1], marker="o", markersize=5.0, color="#FEF3C7", zorder=8)

    # Reference horizontal lines
    ax.plot([mid_shoulder[0] - 50, mid_shoulder[0] + 50], [mid_shoulder[1], mid_shoulder[1]], 
            color=COLOR_REF_LINE, linewidth=1.4, linestyle=":", zorder=6)
    ax.plot([mid_hip[0] - 50, mid_hip[0] + 50], [mid_hip[1], mid_hip[1]], 
            color=COLOR_REF_LINE, linewidth=1.4, linestyle=":", zorder=6)

    # ------------------------------------------------------------------------
    # 14 ELEVATION ANGLES (\theta_1 .. \theta_{14}) - CYAN/BLUE
    # ------------------------------------------------------------------------
    # \theta_1: Left Upper Arm (11 -> 13)
    ref_1 = p11 + np.array([75, 0])
    ax.plot([p11[0], ref_1[0]], [p11[1], ref_1[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v1 = p13 - p11
    draw_arc(ax, p11, 46, 0.0, np.arctan2(v1[1], v1[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p11 + np.array([40, 18]), np.array([680.0, 220.0]), r"$\theta_1$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_2: Right Upper Arm (12 -> 14)
    ref_2 = p12 + np.array([-75, 0])
    ax.plot([p12[0], ref_2[0]], [p12[1], ref_2[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v2 = p14 - p12
    draw_arc(ax, p12, 46, np.pi, np.arctan2(v2[1], v2[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p12 + np.array([-40, 18]), np.array([165.0, 220.0]), r"$\theta_2$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_3: Left Forearm (13 -> 15)
    ref_3 = p13 + np.array([75, 0])
    ax.plot([p13[0], ref_3[0]], [p13[1], ref_3[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v3 = p15 - p13
    draw_arc(ax, p13, 40, 0.0, np.arctan2(v3[1], v3[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p13 + np.array([35, -12]), np.array([760.0, 415.0]), r"$\theta_3$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_4: Right Forearm (14 -> 16)
    ref_4 = p14 + np.array([-75, 0])
    ax.plot([p14[0], ref_4[0]], [p14[1], ref_4[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v4 = p16 - p14
    draw_arc(ax, p14, 40, np.pi, np.arctan2(v4[1], v4[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p14 + np.array([-35, -12]), np.array([85.0, 415.0]), r"$\theta_4$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_5: Left Thigh (23 -> 25)
    ref_5 = p23 + np.array([75, 0])
    ax.plot([p23[0], ref_5[0]], [p23[1], ref_5[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v5 = p25 - p23
    draw_arc(ax, p23, 45, 0.0, np.arctan2(v5[1], v5[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p23 + np.array([32, 32]), np.array([650.0, 620.0]), r"$\theta_5$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_6: Right Thigh (24 -> 26)
    ref_6 = p24 + np.array([-75, 0])
    ax.plot([p24[0], ref_6[0]], [p24[1], ref_6[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v6 = p26 - p24
    draw_arc(ax, p24, 45, np.pi, np.arctan2(v6[1], v6[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p24 + np.array([-32, 32]), np.array([195.0, 620.0]), r"$\theta_6$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_7: Left Shank (25 -> 27)
    ref_7 = p25 + np.array([75, 0])
    ax.plot([p25[0], ref_7[0]], [p25[1], ref_7[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v7 = p27 - p25
    draw_arc(ax, p25, 42, 0.0, np.arctan2(v7[1], v7[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p25 + np.array([28, 30]), np.array([660.0, 795.0]), r"$\theta_7$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_8: Right Shank (26 -> 28)
    ref_8 = p26 + np.array([-75, 0])
    ax.plot([p26[0], ref_8[0]], [p26[1], ref_8[1]], color=COLOR_REF_LINE, linewidth=1.5, linestyle=":", zorder=6)
    v8 = p28 - p26
    draw_arc(ax, p26, 42, np.pi, np.arctan2(v8[1], v8[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, p26 + np.array([-28, 30]), np.array([185.0, 795.0]), r"$\theta_8$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_9: Shoulder Girdle Clavicular Tilt (12 -> 11) - placed cleanly above clavicle
    draw_badge(ax, np.array([365.0, 279.1]), np.array([365.0, 252.0]), r"$\theta_9$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_{10}: Pelvic Girdle Tilt (24 -> 23)
    draw_badge(ax, mid_hip + np.array([0, 6]), np.array([421.1, 565.0]), r"$\theta_{10}$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_{11}: Left Torso Flank Tilt (11 -> 23)
    mid_flank_L = (p11 + p23) / 2.0
    draw_badge(ax, mid_flank_L, np.array([580.0, 435.0]), r"$\theta_{11}$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_{12}: Right Torso Flank Tilt (12 -> 24)
    mid_flank_R = (p12 + p24) / 2.0
    draw_badge(ax, mid_flank_R, np.array([265.0, 435.0]), r"$\theta_{12}$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_{13}: Spine Axis Tilt (mid_hip -> mid_shoulder)
    ref_spine = mid_hip + np.array([55, 0])
    ax.plot([mid_hip[0], ref_spine[0]], [mid_hip[1], ref_spine[1]], color=COLOR_REF_LINE, linewidth=1.4, linestyle=":", zorder=6)
    v_sp = mid_shoulder - mid_hip
    draw_arc(ax, mid_hip, 35, 0.0, np.arctan2(v_sp[1], v_sp[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, mid_hip + np.array([20, -25]), np.array([480.0, 470.0]), r"$\theta_{13}$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # \theta_{14}: Neck Axis Tilt (mid_shoulder -> 0)
    ref_neck = mid_shoulder + np.array([55, 0])
    ax.plot([mid_shoulder[0], ref_neck[0]], [mid_shoulder[1], ref_neck[1]], color=COLOR_REF_LINE, linewidth=1.4, linestyle=":", zorder=6)
    v_nk = LM[0] - mid_shoulder
    draw_arc(ax, mid_shoulder, 35, 0.0, np.arctan2(v_nk[1], v_nk[0]), COLOR_ELEV, COLOR_ELEV_FILL)
    draw_badge(ax, mid_shoulder + np.array([22, -30]), np.array([525.0, 180.0]), r"$\theta_{14}$", 
               COLOR_ELEV, COLOR_ELEV_BG, COLOR_ELEV_BORDER, fontsize=badge_fs)

    # ------------------------------------------------------------------------
    # 10 JOINT ARTICULATION ANGLES (\psi_{15} .. \psi_{24}) - ORANGE
    # ------------------------------------------------------------------------
    # \psi_{15}: Left Elbow Flexion (11 - 13 - 15)
    v_11_13 = p11 - p13
    v_15_13 = p15 - p13
    draw_arc(ax, p13, 38, np.arctan2(v_11_13[1], v_11_13[0]), np.arctan2(v_15_13[1], v_15_13[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p13 + np.array([12, -22]), np.array([710.0, 305.0]), r"$\psi_{15}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{16}: Right Elbow Flexion (12 - 14 - 16)
    v_12_14 = p12 - p14
    v_16_14 = p16 - p14
    draw_arc(ax, p14, 38, np.arctan2(v_12_14[1], v_12_14[0]), np.arctan2(v_16_14[1], v_16_14[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p14 + np.array([-12, -22]), np.array([135.0, 305.0]), r"$\psi_{16}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{17}: Left Shoulder Angle (23 - 11 - 13)
    v_23_11 = p23 - p11
    v_13_11 = p13 - p11
    draw_arc(ax, p11, 40, np.arctan2(v_23_11[1], v_23_11[0]), np.arctan2(v_13_11[1], v_13_11[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p11 + np.array([18, 30]), np.array([620.0, 335.0]), r"$\psi_{17}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{18}: Right Shoulder Angle (24 - 12 - 14)
    v_24_12 = p24 - p12
    v_14_12 = p14 - p12
    draw_arc(ax, p12, 40, np.arctan2(v_24_12[1], v_24_12[0]), np.arctan2(v_14_12[1], v_14_12[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p12 + np.array([-18, 30]), np.array([225.0, 335.0]), r"$\psi_{18}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{19}: Left Hip Hinge Angle (11 - 23 - 25)
    v_11_23 = p11 - p23
    v_25_23 = p25 - p23
    draw_arc(ax, p23, 38, np.arctan2(v_11_23[1], v_11_23[0]), np.arctan2(v_25_23[1], v_25_23[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p23 + np.array([20, 20]), np.array([575.0, 535.0]), r"$\psi_{19}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{20}: Right Hip Hinge Angle (12 - 24 - 26)
    v_12_24 = p12 - p24
    v_26_24 = p26 - p24
    draw_arc(ax, p24, 38, np.arctan2(v_12_24[1], v_12_24[0]), np.arctan2(v_26_24[1], v_26_24[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p24 + np.array([-20, 20]), np.array([270.0, 535.0]), r"$\psi_{20}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{21}: Left Knee Flexion (23 - 25 - 27) - cleanly inset to clear knee joint
    v_23_25 = p23 - p25
    v_27_25 = p27 - p25
    draw_arc(ax, p25, 42, np.arctan2(v_23_25[1], v_23_25[0]), np.arctan2(v_27_25[1], v_27_25[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p25 + np.array([-22, 0]), np.array([465.0, 715.0]), r"$\psi_{21}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{22}: Right Knee Flexion (24 - 26 - 28) - cleanly inset to clear knee joint
    v_24_26 = p24 - p26
    v_28_26 = p28 - p26
    draw_arc(ax, p26, 42, np.arctan2(v_24_26[1], v_24_26[0]), np.arctan2(v_28_26[1], v_28_26[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, p26 + np.array([22, 0]), np.array([378.0, 715.0]), r"$\psi_{22}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{23}: Torso Posture Angle (0 - mid_shoulder - mid_hip)
    v_0_ms = LM[0] - mid_shoulder
    v_mh_ms = mid_hip - mid_shoulder
    draw_arc(ax, mid_shoulder, 42, np.arctan2(v_0_ms[1], v_0_ms[0]), np.arctan2(v_mh_ms[1], v_mh_ms[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, np.array([410.0, 220.0]), np.array([320.0, 170.0]), r"$\psi_{23}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

    # \psi_{24}: Arm Splay Angle (13 - mid_shoulder - 14)
    v_13_ms = p13 - mid_shoulder
    v_14_ms = p14 - mid_shoulder
    draw_arc(ax, mid_shoulder, 55, np.arctan2(v_13_ms[1], v_13_ms[0]), np.arctan2(v_14_ms[1], v_14_ms[0]), COLOR_JOINT, COLOR_JOINT_FILL, lw=2.6)
    draw_badge(ax, mid_shoulder + np.array([0, 50]), np.array([423.2, 355.0]), r"$\psi_{24}$", 
               COLOR_JOINT, COLOR_JOINT_BG, COLOR_JOINT_BORDER, fontsize=badge_fs)

# ============================================================================
# 4. FIGURE BUILDERS
# ============================================================================
def build_pure_skeleton(out_png: Path, out_pdf: Path, dpi: int = 300):
    """Generates the pure human skeleton with ONLY badges and zero text clutter."""
    fig, ax = plt.subplots(figsize=(8.8, 10.6), dpi=dpi, facecolor="white")
    ax.axis("off")
    ax.set_aspect("equal")

    render_skeleton_anatomy(ax, badge_fs=13.5)

    ax.set_xlim(35, 810)
    ax.set_ylim(980, 55)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Generated Pure Skeleton -> {out_png} & {out_pdf}")

def build_legend_skeleton(out_png: Path, out_pdf: Path, dpi: int = 300):
    """Generates the human skeleton with an unobtrusive single-line minimal legend at bottom."""
    fig, ax = plt.subplots(figsize=(8.8, 11.2), dpi=dpi, facecolor="white")
    ax.axis("off")
    ax.set_aspect("equal")

    render_skeleton_anatomy(ax, badge_fs=13.5)

    legend_y = 1015
    ax.plot(145, legend_y, marker="s", markersize=9, color=COLOR_ELEV, zorder=12)
    ax.text(160, legend_y, r"$\mathbf{\theta_1 - \theta_{14}}$ : Segment Elevation Angles",
            fontsize=11.5, color="#0F172A", va="center", ha="left")
    
    ax.plot(475, legend_y, marker="s", markersize=9, color=COLOR_JOINT, zorder=12)
    ax.text(490, legend_y, r"$\mathbf{\psi_{15} - \psi_{24}}$ : Joint Articulation & Posture Angles",
            fontsize=11.5, color="#0F172A", va="center", ha="left")

    ax.set_xlim(35, 810)
    ax.set_ylim(1040, 55)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Generated Skeleton with Minimal Legend -> {out_png} & {out_pdf}")

# ============================================================================
# 5. MAIN CLI ENTRYPOINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready 24 Kinematic Angles diagrams.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "paper" / "images"),
        help="Target directory to save generated PNG and PDF figures."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster PNGs.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pure Skeleton (Canonical Default: 24_angles_single_skeleton)
    pure_png = out_dir / "24_angles_single_skeleton.png"
    pure_pdf = out_dir / "24_angles_single_skeleton.pdf"
    build_pure_skeleton(pure_png, pure_pdf, dpi=args.dpi)

    # 2. Minimal Legend Variant (24_angles_single_skeleton_legend)
    legend_png = out_dir / "24_angles_single_skeleton_legend.png"
    legend_pdf = out_dir / "24_angles_single_skeleton_legend.pdf"
    build_legend_skeleton(legend_png, legend_pdf, dpi=args.dpi)

    # Keep backup links for any legacy build scripts
    import shutil
    shutil.copy2(pure_png, out_dir / "24_angles_single_skeleton_compact.png")
    shutil.copy2(pure_pdf, out_dir / "24_angles_single_skeleton_compact.pdf")

    print("\n[SUCCESS] Pure skeleton figure generated without text clutter!")

if __name__ == "__main__":
    main()

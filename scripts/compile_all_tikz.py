"""
Script to generate all 12 architecture diagrams using pure LaTeX PGF/TikZ.
Compiles each .tex file with tectonic into vector PDF and converts to high-res PNG
using macOS native qlmanage.

Outputs:
  - Source files: paper/tikz/model_*.tex
  - Publication artifacts: paper/images/model_*.pdf & paper/images/model_*.png
"""

import os
import shutil
import subprocess

TIKZ_DIR = "/Volumes/WorkSpace/Project/Gym_Classification/paper/tikz"
IMG_DIR = "/Volumes/WorkSpace/Project/Gym_Classification/paper/images"
TECTONIC_BIN = "/opt/homebrew/bin/tectonic"

os.makedirs(TIKZ_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

PREAMBLE = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}

% Unified Academic Color Palette
\definecolor{cInput}{RGB}{235, 248, 255}
\definecolor{bInput}{RGB}{49, 130, 206}

\definecolor{cTrans}{RGB}{240, 249, 255}
\definecolor{bTrans}{RGB}{43, 108, 176}

\definecolor{cGCN}{RGB}{230, 255, 250}
\definecolor{bGCN}{RGB}{49, 151, 149}

\definecolor{cRecurr}{RGB}{250, 245, 255}
\definecolor{bRecurr}{RGB}{128, 90, 213}

\definecolor{cHead}{RGB}{255, 245, 245}
\definecolor{bHead}{RGB}{229, 62, 62}

\definecolor{cFusion}{RGB}{254, 252, 191}
\definecolor{bFusion}{RGB}{214, 158, 46}

\definecolor{cGray}{RGB}{247, 250, 252}
\definecolor{bGray}{RGB}{160, 174, 192}

\definecolor{tDark}{RGB}{26, 32, 44}
\definecolor{tMuted}{RGB}{74, 85, 104}
"""

DIAGRAMS = {}

# -----------------------------------------------------------------------------
# 1. Unidirectional LSTM
# -----------------------------------------------------------------------------
DIAGRAMS["model_lstm"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (0, 1.1) {Unidirectional LSTM Sequence Baseline};

\node[block=bInput, fill=cInput, text width=7.4cm] (input) at (0, 0) {
    \textbf{Input Sequence Tensor}\\
    {\footnotesize Shape: $[B, T=32, D=117]$}
};

\node[block=bGray, fill=white, text width=7.4cm, below=of input] (norm) {
    \textbf{LayerNorm(117)}\\
    {\footnotesize Standardizes skeletal landmark vector}
};

\node[block=bRecurr, fill=cRecurr, text width=7.4cm, below=of norm] (lstm) {
    \textbf{2-Layer Unidirectional LSTM}\\
    {\footnotesize $\text{hidden\_dim} = 160$ (Forward temporal flow)}\\
    {\footnotesize $\text{num\_layers} = 2$ \textbar{} $\text{Dropout} = 0.3$ \textbar{} $\approx 345\text{K}$ Params}\\
    {\footnotesize Output: $[B, T=32, 160]$}
};

\node[block=bInput, fill=cInput, text width=7.4cm, below=of lstm] (gap) {
    \textbf{Temporal Global Average Pooling (GAP)}\\
    {\footnotesize $\text{pooled} = \frac{1}{T}\sum_{t=1}^T h_t \longrightarrow [B, 160]$}
};

\node[block=bHead, fill=cHead, text width=7.4cm, below=of gap] (head) {
    \textbf{Two-Stage MLP Classifier Head}\\
    {\footnotesize $\text{Dropout}(0.3) \to \text{Linear}(160 \to 64) \to \text{ReLU}$}\\
    {\footnotesize $\to \text{Dropout}(0.3) \to \text{Linear}(64 \to 22) \to \text{Softmax}$}\\
    {\footnotesize Output: 22 Exercise Class Probabilities $P \in \Delta^{21}$}
};

\draw[arrow] (input) -- (norm);
\draw[arrow] (norm) -- (lstm);
\draw[arrow] (lstm) -- (gap);
\draw[arrow] (gap) -- (head);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 2. Bidirectional LSTM
# -----------------------------------------------------------------------------
DIAGRAMS["model_bilstm"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (0, 1.1) {Bidirectional LSTM Sequence Baseline};

\node[block=bInput, fill=cInput, text width=7.4cm] (input) at (0, 0) {
    \textbf{Input Sequence Tensor}\\
    {\footnotesize Shape: $[B, T=32, D=117]$}
};

\node[block=bGray, fill=white, text width=7.4cm, below=of input] (norm) {
    \textbf{LayerNorm(117)}\\
    {\footnotesize Feature standardization across joints}
};

\node[block=bRecurr, fill=cRecurr, text width=7.4cm, below=of norm] (bilstm) {
    \textbf{2-Layer Bidirectional LSTM}\\
    {\footnotesize $\text{hidden\_dim} = 96$ per direction ($\vec{h}_t \in \mathbb{R}^{96}, \overleftarrow{h}_t \in \mathbb{R}^{96}$)}\\
    {\footnotesize Forward + Backward $= 192$ \textbar{} $\text{Dropout} = 0.3$}\\
    {\footnotesize Output: $[B, T=32, 192]$}
};

\node[block=bInput, fill=cInput, text width=7.4cm, below=of bilstm] (gap) {
    \textbf{Temporal Global Average Pooling (GAP)}\\
    {\footnotesize $\text{pooled} = \frac{1}{T}\sum_{t=1}^T [\vec{h}_t \,\|\, \overleftarrow{h}_t] \longrightarrow [B, 192]$}
};

\node[block=bHead, fill=cHead, text width=7.4cm, below=of gap] (head) {
    \textbf{Two-Stage MLP Classifier Head}\\
    {\footnotesize $\text{Dropout}(0.3) \to \text{Linear}(192 \to 64) \to \text{ReLU}$}\\
    {\footnotesize $\to \text{Dropout}(0.3) \to \text{Linear}(64 \to 22) \to \text{Softmax}$}\\
    {\footnotesize Output: 22 Exercise Class Probabilities $P \in \Delta^{21}$}
};

\draw[arrow] (input) -- (norm);
\draw[arrow] (norm) -- (bilstm);
\draw[arrow] (bilstm) -- (gap);
\draw[arrow] (gap) -- (head);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 3. Skeletal Transformer
# -----------------------------------------------------------------------------
DIAGRAMS["model_transformer"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (0, 1.1) {Skeletal Transformer Encoder};

\node[block=bInput, fill=cInput, text width=7.6cm] (input) at (0, 0) {
    \textbf{Input Skeleton Sequence}\\
    {\footnotesize Shape: $[B, T=32, D=117]$}
};

\node[block=bTrans, fill=white, text width=7.6cm, below=of input] (proj) {
    \textbf{Linear Projection + Sinusoidal PE}\\
    {\footnotesize $\text{Linear}(117 \to 128) + \text{PE}(T=32, d=128) \longrightarrow [B, 32, 128]$}
};

% Container for 4x Encoder layers
\node[block=bTrans, fill=cTrans, text width=7.0cm, below=1.1cm of proj] (mhsa) {
    \textbf{Multi-Head Self-Attention (MHSA)}\\
    {\footnotesize $\text{Pre-LN} \to 8 \text{ Heads}, d_k=16 \ (d=128)$}\\
    {\footnotesize $\text{Softmax}(QK^T / \sqrt{d_k})V + \text{Residual Add}$}
};

\node[block=bTrans, fill=cTrans, text width=7.0cm, below=0.6cm of mhsa] (ffn) {
    \textbf{Feed-Forward Network (FFN)}\\
    {\footnotesize $\text{Pre-LN} \to \text{Linear}(128 \to 256) \to \text{GELU}$}\\
    {\footnotesize $\to \text{Linear}(256 \to 128) + \text{Residual Add}$}
};

\begin{scope}[on background layer]
    \node[draw=bTrans, line width=1.2pt, dashed, rounded corners=6pt, fill=cGray!60,
          fit=(mhsa)(ffn), inner sep=12pt] (cont) {};
    \node[anchor=south west, font=\bfseries\scriptsize, fill=bTrans!15, draw=bTrans, rounded corners=3pt, inner sep=3pt, xshift=2pt, yshift=2pt] at (cont.north west) {
        $4\times$ Pre-LN Layers
    };
\end{scope}

\node[block=bInput, fill=cInput, text width=7.6cm, below=0.8cm of cont] (gap) {
    \textbf{Temporal Global Average Pooling (GAP)}\\
    {\footnotesize $\text{pooled} = \frac{1}{T}\sum_{t=1}^T z_t \longrightarrow [B, 128]$}
};

\node[block=bHead, fill=cHead, text width=7.6cm, below=of gap] (head) {
    \textbf{Two-Stage MLP Classifier Head}\\
    {\footnotesize $\text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 22)$}\\
    {\footnotesize Output: $P \in \Delta^{21}$ across 22 Exercise Classes}
};

\draw[arrow] (input) -- (proj);
\draw[arrow] (proj) -- (cont);
\draw[arrow] (cont) -- (gap);
\draw[arrow] (gap) -- (head);
\draw[arrow] (mhsa) -- (ffn);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 4. ST-GCN Baseline
# -----------------------------------------------------------------------------
DIAGRAMS["model_stgcn"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.6cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (0, 1.1) {ST-GCN Baseline Architecture (Yan et al., 2018)};

\node[block=bInput, fill=cInput, text width=7.5cm] (input) at (0, 0) {
    \textbf{Input Kinematic Stream}\\
    {\footnotesize Shape: $[B, C=3, T=32, V=13]$}
};

\node[block=bGray, fill=white, text width=7.5cm, below=of input] (bn) {
    \textbf{Data BatchNorm2d(3)}\\
    {\footnotesize Standardizes coordinates across joints}
};

% 3 Stages Container
\node[block=bGray, fill=white, text width=6.8cm, below=1.1cm of bn] (sgc) {
    \textbf{Spatial Graph Conv (Rigid $A_{\text{phys}}$)}\\
    {\footnotesize $A = A_{\text{norm}} \odot M$ (Fixed physical anatomical limbs)}\\
    {\footnotesize $\text{Conv2d}(1\times 1) \to \text{BatchNorm2d} \to \text{ReLU}$}
};

\node[block=bGray, fill=white, text width=6.8cm, below=0.6cm of sgc] (tcn) {
    \textbf{Temporal Conv \& Residual Shortcut}\\
    {\footnotesize $\text{Conv2d}(9\times 1, \text{stride}=s) \to \text{BN} \to \text{Dropout}(0.2)$}\\
    {\footnotesize Residual: $\text{Conv2d}(1\times 1, \text{stride}=s) + \text{BN} \to \text{ReLU}$}
};

\begin{scope}[on background layer]
    \node[draw=bGray, line width=1.2pt, dashed, rounded corners=6pt, fill=cGray!60,
          fit=(sgc)(tcn), inner sep=12pt] (cont) {};
    \node[anchor=south west, font=\bfseries\scriptsize, fill=bGray!20, draw=bGray, rounded corners=3pt, inner sep=3pt, xshift=2pt, yshift=2pt] at (cont.north west) {
        $3\times$ ST-GCN Stages
    };
\end{scope}

\node[block=bInput, fill=cInput, text width=7.5cm, below=0.8cm of cont] (gap) {
    \textbf{Global Average Pooling (GAP)}\\
    {\footnotesize Spatial $(V)$ and Temporal $(T)$ Pooling $\longrightarrow [B, 150]$}
};

\node[block=bHead, fill=cHead, text width=7.5cm, below=of gap] (head) {
    \textbf{Linear Classifier Head}\\
    {\footnotesize $\text{Linear}(150 \to 22 \text{ Classes}) \to \text{Softmax}$}\\
    {\footnotesize Output: 22 Exercise Class Probabilities $P \in \Delta^{21}$}
};

\draw[arrow] (input) -- (bn);
\draw[arrow] (bn) -- (cont);
\draw[arrow] (cont) -- (gap);
\draw[arrow] (gap) -- (head);
\draw[arrow] (sgc) -- (tcn);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 5. AAGCN (Macro + Adaptive Block)
# -----------------------------------------------------------------------------
DIAGRAMS["model_aagcn"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.55cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=5pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (6.0, 1.5) {Adaptive Spatial-Temporal Graph Convolutional Network (AAGCN)};

% Sub-headers
\node[font=\bfseries\footnotesize, text=tDark] (sub1) at (0, 0.6) {AAGCN Macro Backbone};
\node[font=\bfseries\footnotesize, text=bGCN] (sub2) at (9.0, 0.6) {Adaptive Graph Conv (Spatial-Temporal Block)};

% Left Column: Macro Backbone
\node[block=bInput, fill=cInput, text width=5.0cm, below=0.35cm of sub1] (m_input) {
    \textbf{Input Kinematic Stream}\\
    {\footnotesize $[B, C=3, T=32, V=13]$}
};

\node[block=bGray, fill=white, text width=5.0cm, below=of m_input] (m_bn) {
    \textbf{Data BatchNorm2d(3)}\\
    {\footnotesize Joint-level normalisation}
};

\node[block=bGCN, fill=cGCN, text width=5.0cm, below=of m_bn] (m_s1) {
    \textbf{AAGCN Stage 1}\\
    {\footnotesize Channels: $3 \to 48$, Stride: 1}
};

\node[block=bGCN, fill=cGCN, text width=5.0cm, below=of m_s1] (m_s2) {
    \textbf{AAGCN Stage 2}\\
    {\footnotesize Channels: $48 \to 96$, Stride: 2 ($T: 32 \to 16$)}
};

\node[block=bGCN, fill=cGCN, text width=5.0cm, below=of m_s2] (m_s3) {
    \textbf{AAGCN Stage 3}\\
    {\footnotesize Channels: $96 \to 150$, Stride: 2 ($T: 16 \to 8$)}
};

\node[block=bInput, fill=cInput, text width=5.0cm, below=of m_s3] (m_gap) {
    \textbf{GAP over $(V, T)$}\\
    {\footnotesize Output: $[B, 150]$}
};

\node[block=bHead, fill=cHead, text width=5.0cm, below=of m_gap] (m_head) {
    \textbf{Linear Classifier Head}\\
    {\footnotesize $\text{Linear}(150 \to 22) \to \text{Softmax}$}\\
    {\footnotesize Yields $P_m \in \Delta^{21}$}
};

\draw[arrow] (m_input) -- (m_bn);
\draw[arrow] (m_bn) -- (m_s1);
\draw[arrow] (m_s1) -- (m_s2);
\draw[arrow] (m_s2) -- (m_s3);
\draw[arrow] (m_s3) -- (m_gap);
\draw[arrow] (m_gap) -- (m_head);

% Macro background container
\begin{scope}[on background layer]
    \node[draw=tDark!60, line width=1.1pt, rounded corners=6pt, fill=cGray!40,
          fit=(sub1)(m_input)(m_head), inner sep=8pt] (macro_cont) {};
\end{scope}

% Right Column: Zoom into Adaptive Block
\node[block=bGray, fill=white, text width=1.7cm] (b1) at (7.0, -0.6) {
    \textbf{Branch 1}\\
    {\scriptsize $A_{\text{topo}}\odot M$\\(Anatomy)}
};
\node[block=bInput, fill=cInput, text width=1.7cm] (b2) at (9.0, -0.6) {
    \textbf{Branch 2}\\
    {\scriptsize Matrix $B$\\(Global)}
};
\node[block=bFusion, fill=cFusion, text width=1.7cm] (b3) at (11.0, -0.6) {
    \textbf{Branch 3}\\
    {\scriptsize $C(X)$ Attention\\(Dynamic)}
};

\node[block=bGCN, fill=cGCN, text width=5.8cm, below=0.8cm of b2] (sumA) {
    \textbf{Unified Adaptive Adjacency Matrix}\\
    {\footnotesize $A_{\text{total}} = (A_{\text{topo}} \odot M) + B + C(X)$}
};

\node[block=bGCN, fill=white, text width=5.8cm, below=of sumA] (msg) {
    \textbf{Spatial Graph Message Passing}\\
    {\footnotesize $X_g = A_{\text{total}} \cdot X \longrightarrow \text{Conv2d}(1\times 1) \to \text{BN} \to \text{GELU}$}
};

\node[block=bTrans, fill=white, text width=5.8cm, below=of msg] (tcn_blk) {
    \textbf{Multi-Scale Temporal Conv (MS-TCN)}\\
    {\footnotesize $\text{Conv2d}(9\times 1, \text{stride}=s) \to \text{BN} \to \text{Dropout}(0.2)$}
};

\node[block=bFusion, fill=cFusion, text width=5.8cm, below=of tcn_blk] (res) {
    \textbf{Residual Shortcut Connection (+)}\\
    {\footnotesize Shortcut: $\text{Conv2d}(1\times 1, \text{stride}=s) + \text{BN} \to (+)$ Add $\to \text{GELU}$}
};

\node[block=bInput, fill=cInput, text width=5.8cm, below=of res] (blk_out) {
    \textbf{Block Output Feature Map}\\
    {\footnotesize Passed to next stage or GAP}
};

\draw[arrow] (b1.south) -- (sumA.north -| b1.south);
\draw[arrow] (b2.south) -- (sumA.north);
\draw[arrow] (b3.south) -- (sumA.north -| b3.south);
\draw[arrow] (sumA) -- (msg);
\draw[arrow] (msg) -- (tcn_blk);
\draw[arrow] (tcn_blk) -- (res);
\draw[arrow] (res) -- (blk_out);

% Block background container
\begin{scope}[on background layer]
    \node[draw=bGCN, line width=1.3pt, rounded corners=6pt, fill=white,
          fit=(sub2)(b1)(b3)(blk_out), inner sep=10pt] (block_cont) {};
\end{scope}

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 6. Dual-Branch Feature Fusion (BranchConcat)
# -----------------------------------------------------------------------------
DIAGRAMS["model_dual_branch"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (3.8, 1.25) {Dual-Branch Feature Fusion (BranchConcat Architecture)};

% Branch 1: Coordinates
\node[block=bInput, fill=cInput, text width=3.4cm] (in1) at (1.5, 0) {
    \textbf{Branch 1: Coordinates}\\
    {\footnotesize $[B, T=32, D_1=39/49]$}\\
    {\scriptsize Euclidean metric position}
};

\node[block=bTrans, fill=cTrans, text width=3.4cm, below=of in1] (enc1) {
    \textbf{Temporal Encoder 1}\\
    {\footnotesize $\text{Linear}(D_1 \to d_{\text{model}})$}\\
    {\footnotesize $\text{LSTM / Transformer} \ (L=2)$}\\
    {\footnotesize Output: $r_1 \in \mathbb{R}^{64}$}
};

% Branch 2: Angles
\node[block=bInput, fill=cInput, text width=3.4cm] (in2) at (6.1, 0) {
    \textbf{Branch 2: Angles}\\
    {\footnotesize $[B, T=32, D_2=78/286]$}\\
    {\scriptsize Scale-invariant rotation}
};

\node[block=bTrans, fill=cTrans, text width=3.4cm, below=of in2] (enc2) {
    \textbf{Temporal Encoder 2}\\
    {\footnotesize $\text{Linear}(D_2 \to d_{\text{model}})$}\\
    {\footnotesize $\text{LSTM / Transformer} \ (L=2)$}\\
    {\footnotesize Output: $r_2 \in \mathbb{R}^{64}$}
};

\draw[arrow] (in1) -- (enc1);
\draw[arrow] (in2) -- (enc2);

% Merge
\node[block=bFusion, fill=cFusion, text width=6.2cm, below=0.8cm of $(enc1.south)!0.5!(enc2.south)$] (concat) {
    \textbf{Feature Concatenation $[r_1 \,\|\, r_2]$}\\
    {\footnotesize Merged Latent Vector: $r = [r_1 \,\|\, r_2] \in \mathbb{R}^{128}$}
};

\node[block=bHead, fill=cHead, text width=6.2cm, below=of concat] (head) {
    \textbf{Shared MLP Classifier Head}\\
    {\footnotesize $\text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Dropout}(0.3)$}\\
    {\footnotesize $\to \text{Linear}(64 \to 22 \text{ Classes}) \to \text{Softmax}$}\\
    {\footnotesize Output: Predicted Class $\hat{y} \in \{1, \dots, 22\}$}
};

\draw[arrow] (enc1.south) -- (concat.north -| enc1.south);
\draw[arrow] (enc2.south) -- (concat.north -| enc2.south);
\draw[arrow] (concat) -- (head);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 7. Two-Stream AAGCN
# -----------------------------------------------------------------------------
DIAGRAMS["model_fusion_2stream"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (3.8, 1.25) {Two-Stream AAGCN (Static Kinematics Late Fusion)};

% Stream 1: Joint
\node[block=bInput, fill=cInput, text width=3.3cm] (in1) at (1.5, 0) {
    \textbf{Joint Stream (Rel 3D)}\\
    {\footnotesize $T=32, V=13$ \textbar{} Coords}
};

\node[block=bGCN, fill=cGCN, text width=3.3cm, below=of in1] (back1) {
    \textbf{AAGCN Joint Backbone}\\
    {\footnotesize 3 Stages $[48, 96, 150]$}\\
    {\footnotesize Output: $P_{\text{joint}} \in \Delta^{21}$}
};

% Stream 2: Bone
\node[block=bInput, fill=cInput, text width=3.3cm] (in2) at (6.1, 0) {
    \textbf{Bone Stream (Bone 3D)}\\
    {\footnotesize $T=32, V=13$ \textbar{} Vectors}
};

\node[block=bGCN, fill=cGCN, text width=3.3cm, below=of in2] (back2) {
    \textbf{AAGCN Bone Backbone}\\
    {\footnotesize 3 Stages $[48, 96, 150]$}\\
    {\footnotesize Output: $P_{\text{bone}} \in \Delta^{21}$}
};

\draw[arrow] (in1) -- (back1);
\draw[arrow] (in2) -- (back2);

% Late Fusion
\node[block=bFusion, fill=cFusion, text width=5.6cm, below=0.8cm of $(back1.south)!0.5!(back2.south)$] (fuse) {
    \textbf{Weighted Soft Late Fusion}\\
    {\footnotesize $P_{\text{final}} = w_1 \cdot P_{\text{joint}} + w_2 \cdot P_{\text{bone}}$}\\
    {\footnotesize $(w_1 + w_2 = 1, \ w_i \ge 0)$}
};

\node[block=bHead, fill=cHead, text width=5.0cm, below=of fuse] (pred) {
    \textbf{Final Exercise Prediction}\\
    {\footnotesize $\hat{y} = \arg\max(P_{\text{final}}) \in \{1, \dots, 22\}$}
};

\draw[arrow] (back1.south) -- (fuse.north -| back1.south);
\draw[arrow] (back2.south) -- (fuse.north -| back2.south);
\draw[arrow] (fuse) -- (pred);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 8. Three-Stream AAGCN
# -----------------------------------------------------------------------------
DIAGRAMS["model_fusion_3stream"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=5pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (4.8, 1.25) {Three-Stream AAGCN (Joint + Bone + Motion Fusion)};

% Stream 1: Joint Position
\node[block=bInput, fill=cInput, text width=2.8cm] (in1) at (1.0, 0) {
    \textbf{Joint Stream (3D)}\\
    {\footnotesize $T=32, V=13$}
};
\node[block=bGCN, fill=cGCN, text width=2.8cm, below=of in1] (back1) {
    \textbf{AAGCN Joint}\\
    {\footnotesize 3 Stages $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{joint}} \in \Delta^{21}$}
};

% Stream 2: Bone Vector
\node[block=bInput, fill=cInput, text width=2.8cm] (in2) at (4.8, 0) {
    \textbf{Bone Stream (3D)}\\
    {\footnotesize $T=32, V=13$}
};
\node[block=bGCN, fill=cGCN, text width=2.8cm, below=of in2] (back2) {
    \textbf{AAGCN Bone}\\
    {\footnotesize 3 Stages $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{bone}} \in \Delta^{21}$}
};

% Stream 3: Motion Vector
\node[block=bInput, fill=cInput, text width=2.8cm] (in3) at (8.6, 0) {
    \textbf{Motion Stream (3D)}\\
    {\footnotesize $T=31, V=13$}
};
\node[block=bGCN, fill=cGCN, text width=2.8cm, below=of in3] (back3) {
    \textbf{AAGCN Motion}\\
    {\footnotesize 3 Stages $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{motion}} \in \Delta^{21}$}
};

\draw[arrow] (in1) -- (back1);
\draw[arrow] (in2) -- (back2);
\draw[arrow] (in3) -- (back3);

% Late Fusion
\node[block=bFusion, fill=cFusion, text width=6.2cm, below=0.8cm of back2] (fuse) {
    \textbf{Weighted Soft Late Fusion}\\
    {\footnotesize $P_{\text{final}} = w_1 \cdot P_{\text{joint}} + w_2 \cdot P_{\text{bone}} + w_3 \cdot P_{\text{motion}}$}\\
    {\footnotesize $(\sum_{i=1}^3 w_i = 1, \ w_i \ge 0)$}
};

\node[block=bHead, fill=cHead, text width=5.0cm, below=of fuse] (pred) {
    \textbf{Final Exercise Prediction}\\
    {\footnotesize $\hat{y} = \arg\max(P_{\text{final}}) \in \{1, \dots, 22\}$}
};

\draw[arrow] (back1.south) -- (fuse.north -| back1.south);
\draw[arrow] (back2) -- (fuse);
\draw[arrow] (back3.south) -- (fuse.north -| back3.south);
\draw[arrow] (fuse) -- (pred);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 9. Four-Stream AAGCN
# -----------------------------------------------------------------------------
DIAGRAMS["model_fusion_4stream"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.6cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=5pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (5.7, 1.25) {Four-Stream AAGCN (Unified Graph Paradigm)};

\node[block=bInput, fill=cInput, text width=2.4cm] (in1) at (0.9, 0) {
    \textbf{Joint Stream}\\
    {\footnotesize Rel 3D $[B,3,32,13]$}
};
\node[block=bGCN, fill=cGCN, text width=2.4cm, below=of in1] (back1) {
    \textbf{AAGCN Joint}\\
    {\footnotesize $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{joint}} \in \Delta^{21}$}
};

\node[block=bInput, fill=cInput, text width=2.4cm] (in2) at (4.1, 0) {
    \textbf{Bone Stream}\\
    {\footnotesize Limb 3D $[B,3,32,13]$}
};
\node[block=bGCN, fill=cGCN, text width=2.4cm, below=of in2] (back2) {
    \textbf{AAGCN Bone}\\
    {\footnotesize $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{bone}} \in \Delta^{21}$}
};

\node[block=bInput, fill=cInput, text width=2.4cm] (in3) at (7.3, 0) {
    \textbf{Joint Motion}\\
    {\footnotesize Vel 3D $[B,3,31,13]$}
};
\node[block=bGCN, fill=cGCN, text width=2.4cm, below=of in3] (back3) {
    \textbf{AAGCN J-Mot}\\
    {\footnotesize $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{jmot}} \in \Delta^{21}$}
};

\node[block=bInput, fill=cInput, text width=2.4cm] (in4) at (10.5, 0) {
    \textbf{Bone Motion}\\
    {\footnotesize B-Vel 3D $[B,3,31,13]$}
};
\node[block=bGCN, fill=cGCN, text width=2.4cm, below=of in4] (back4) {
    \textbf{AAGCN B-Mot}\\
    {\footnotesize $[48, 96, 150]$}\\
    {\footnotesize $P_{\text{bmot}} \in \Delta^{21}$}
};

\draw[arrow] (in1) -- (back1);
\draw[arrow] (in2) -- (back2);
\draw[arrow] (in3) -- (back3);
\draw[arrow] (in4) -- (back4);

% Fusion
\node[block=bFusion, fill=cFusion, text width=6.8cm, below=0.8cm of $(back2.south)!0.5!(back3.south)$] (fuse) {
    \textbf{Four-Stream Softmax Late Fusion}\\
    {\footnotesize $P_{\text{fused}} = \frac{1}{4}(P_{\text{joint}} + P_{\text{bone}} + P_{\text{jmot}} + P_{\text{bmot}})$}\\
    {\footnotesize (Graph Unified Consensus)}
};

\node[block=bHead, fill=cHead, text width=5.0cm, below=of fuse] (pred) {
    \textbf{Final Exercise Prediction}\\
    {\footnotesize $\hat{y} = \arg\max(P_{\text{fused}}) \in \{1, \dots, 22\}$}
};

\draw[arrow] (back1.south) -- (fuse.north -| back1.south);
\draw[arrow] (back2.south) -- (fuse.north -| back2.south);
\draw[arrow] (back3.south) -- (fuse.north -| back3.south);
\draw[arrow] (back4.south) -- (fuse.north -| back4.south);
\draw[arrow] (fuse) -- (pred);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 10. SkelGym-Lite
# -----------------------------------------------------------------------------
DIAGRAMS["model_fusion_skelgym_lite"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=6pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (3.8, 1.55) {SkelGym-Lite (Dual-Stream Cross-Paradigm Architecture)};

% Stream 1: Transformer
\node[block=bInput, fill=cInput, text width=3.8cm] (in1) at (1.6, 0) {
    \textbf{Sequence Mix Stream}\\
    {\footnotesize 117-d Biomechanical Mix}\\
    {\footnotesize $[B, T=32, D=117]$}
};

\node[block=bTrans, fill=cTrans, text width=3.8cm, below=of in1] (back1) {
    \textbf{Transformer Backbone}\\
    {\footnotesize 4L Pre-LN MHSA ($d=128$)}\\
    {\footnotesize Captures Temporal Phase}\\
    {\footnotesize Output: $P_{\text{seq}} \in \Delta^{21}$}
};

% Stream 2: AAGCN Bone
\node[block=bInput, fill=cInput, text width=3.8cm] (in2) at (6.0, 0) {
    \textbf{Bone Stream (Kinematics)}\\
    {\footnotesize Directed Limb Vectors}\\
    {\footnotesize $[B, C=3, T=32, V=13]$}
};

\node[block=bGCN, fill=cGCN, text width=3.8cm, below=of in2] (back2) {
    \textbf{AAGCN Bone Backbone}\\
    {\footnotesize Adaptive Graph Conv ($V=13$)}\\
    {\footnotesize Anatomical Segment Flow}\\
    {\footnotesize Output: $P_{\text{bone}} \in \Delta^{21}$}
};

\draw[arrow] (in1) -- (back1);
\draw[arrow] (in2) -- (back2);

% Fusion
\node[block=bFusion, fill=cFusion, text width=5.6cm, below=0.8cm of $(back1.south)!0.5!(back2.south)$] (fuse) {
    \textbf{SLSQP Soft Weighted Fusion}\\
    {\footnotesize $P_{\text{final}} = w_{\text{seq}} \cdot P_{\text{seq}} + w_{\text{bone}} \cdot P_{\text{bone}}$}\\
    {\footnotesize Weights calibrated on Validation NLL}
};

\node[block=bHead, fill=cHead, text width=5.2cm, below=of fuse] (pred) {
    \textbf{Ultra-Low Latency Prediction}\\
    {\footnotesize $\hat{y} = \arg\max(P_{\text{final}})$ \textbar{} 1.44 ms on Host CPU}
};

\draw[arrow] (back1.south) -- (fuse.north -| back1.south);
\draw[arrow] (back2.south) -- (fuse.north -| back2.south);
\draw[arrow] (fuse) -- (pred);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 11. SkelGym-Full
# -----------------------------------------------------------------------------
DIAGRAMS["model_fusion_skelgym_full"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.6cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=5pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (6.0, 1.35) {SkelGym-Full Cross-Paradigm Ensemble \& Consensus Engine};

\node[block=bTrans, fill=cTrans, text width=2.0cm] (e1) at (0.8, 0) {
    \textbf{Transformer}\\
    {\scriptsize Seq Mix (117-d)}\\
    {\scriptsize $P_{\text{trans}} \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (e2) at (3.4, 0) {
    \textbf{AAGCN Joint}\\
    {\scriptsize Rel 3D Coords}\\
    {\scriptsize $P_{\text{joint}} \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (e3) at (6.0, 0) {
    \textbf{AAGCN Bone}\\
    {\scriptsize Bone 3D Vectors}\\
    {\scriptsize $P_{\text{bone}} \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (e4) at (8.6, 0) {
    \textbf{AAGCN J-Mot}\\
    {\scriptsize Joint Velocity}\\
    {\scriptsize $P_{\text{jmot}} \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (e5) at (11.2, 0) {
    \textbf{AAGCN B-Mot}\\
    {\scriptsize Bone Velocity}\\
    {\scriptsize $P_{\text{bmot}} \in \Delta^{21}$}
};

% Stage 1: Window-Level Weighted Soft Voting
\node[block=bFusion, fill=cFusion, text width=9.2cm, below=0.8cm of e3] (s1) {
    \textbf{Stage 1: Window-Level Weighted Soft Voting}\\
    {\footnotesize $P_{\text{win}}(w) = \sum_{m=1}^5 w_m^* \cdot P_m(w)$}\\
    {\scriptsize Weights $w^*$ calibrated via SLSQP simplex optimization on validation NLL}
};

\draw[arrow] (e1.south) -- (s1.north -| e1.south);
\draw[arrow] (e2.south) -- (s1.north -| e2.south);
\draw[arrow] (e3) -- (s1);
\draw[arrow] (e4.south) -- (s1.north -| e4.south);
\draw[arrow] (e5.south) -- (s1.north -| e5.south);

% Stage 2: Temporal Video Consensus Aggregation
\node[block=bInput, fill=cInput, text width=9.2cm, below=0.7cm of s1] (s2) {
    \textbf{Stage 2: Temporal Video Consensus Aggregation}\\
    {\footnotesize Soft-pooling across $K$ sliding windows over full video recording:}\\
    {\footnotesize $P_{\text{vid}}(c) = \sum_{m} w_{\text{vid}, m} \cdot \left[ \frac{1}{K} \sum_{k=1}^K P_{m, c}(w_k) \right]$}\\
    {\scriptsize Expectation-based low-pass filter eliminates occlusions and boundary pauses}
};

\draw[arrow] (s1) -- (s2);

% Final Output
\node[block=bHead, fill=cHead, text width=7.2cm, below=0.7cm of s2] (pred) {
    \textbf{Final Video Exercise Classification}\\
    {\footnotesize $\hat{y}_{\text{vid}} = \arg\max P_{\text{vid}}(c)$ across 22 Classes}\\
    {\footnotesize Video Consensus Benchmark: \textbf{79.11\% $\pm$ 0.25\%}}
};

\draw[arrow] (s2) -- (pred);

\end{tikzpicture}
\end{document}
"""

# -----------------------------------------------------------------------------
# 12. Stacking Meta-Learner
# -----------------------------------------------------------------------------
DIAGRAMS["model_ensemble_stacking"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.65cm,
    block/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=4pt,
        align=center,
        inner sep=5pt
    },
    arrow/.style={->, line width=1.1pt, color=tDark!80}
]

\node[font=\bfseries\large, text=tDark] (title) at (6.0, 1.35) {Stacking Meta-Classifier Architecture (Level-0 \& Level-1)};

\node[block=bTrans, fill=cTrans, text width=2.0cm] (m1) at (0.8, 0) {
    \textbf{Transformer}\\
    {\scriptsize Seq Mix (117-d)}\\
    {\scriptsize $P_1 \in \Delta^{21}$}
};
\node[block=bRecurr, fill=cRecurr, text width=2.0cm] (m2) at (3.4, 0) {
    \textbf{BiLSTM}\\
    {\scriptsize Seq Mix (117-d)}\\
    {\scriptsize $P_2 \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (m3) at (6.0, 0) {
    \textbf{AAGCN Joint}\\
    {\scriptsize Graph Coords}\\
    {\scriptsize $P_3 \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (m4) at (8.6, 0) {
    \textbf{AAGCN Bone}\\
    {\scriptsize Graph Limb}\\
    {\scriptsize $P_4 \in \Delta^{21}$}
};
\node[block=bGCN, fill=cGCN, text width=2.0cm] (m5) at (11.2, 0) {
    \textbf{AAGCN Mot}\\
    {\scriptsize Graph Velocity}\\
    {\scriptsize $P_5 \in \Delta^{21}$}
};

% Meta-Features Formation
\node[block=bFusion, fill=cFusion, text width=9.2cm, below=0.8cm of m3] (meta) {
    \textbf{Level-0 Meta-Feature Construction}\\
    {\footnotesize Concatenate base model probability vectors:}\\
    {\footnotesize $X_{\text{meta}} = [P_1 \,\|\, P_2 \,\|\, P_3 \,\|\, P_4 \,\|\, P_5] \in \mathbb{R}^{B \times 110}$}\\
    {\scriptsize ($5 \text{ models} \times 22 \text{ class probabilities} = 110 \text{ meta-features}$)}
};

\draw[arrow] (m1.south) -- (meta.north -| m1.south);
\draw[arrow] (m2.south) -- (meta.north -| m2.south);
\draw[arrow] (m3) -- (meta);
\draw[arrow] (m4.south) -- (meta.north -| m4.south);
\draw[arrow] (m5.south) -- (meta.north -| m5.south);

% Level-1 Meta-Learner
\node[block=bInput, fill=cInput, text width=9.2cm, below=0.7cm of meta] (lr) {
    \textbf{Level-1 Meta-Classifier: Multinomial Logistic Regression}\\
    {\footnotesize L-BFGS solver \textbar{} $L_2$ Regularization ($C=1.0$)}\\
    {\footnotesize Learns optimal cross-paradigm correlation \& decision boundaries}\\
    {\scriptsize Fitted strictly on out-of-fold validation set predictions}
};

\draw[arrow] (meta) -- (lr);

% Final Output
\node[block=bHead, fill=cHead, text width=7.2cm, below=0.7cm of lr] (pred) {
    \textbf{Final Meta-Ensemble Class Decision}\\
    {\footnotesize $\hat{y} = \arg\max P_{\text{meta}}(c) \in \{1, \dots, 22\}$}\\
    {\scriptsize Captures non-linear cross-paradigm consensus}
};

\draw[arrow] (lr) -- (pred);

\end{tikzpicture}
\end{document}
"""


def compile_diagram(name, latex_code):
    tex_path = os.path.join(TIKZ_DIR, f"{name}.tex")
    pdf_in_tikz = os.path.join(TIKZ_DIR, f"{name}.pdf")
    target_pdf = os.path.join(IMG_DIR, f"{name}.pdf")
    target_png = os.path.join(IMG_DIR, f"{name}.png")

    print(f"\n[Compiling TikZ] {name}.tex ...")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code.strip() + "\n")

    # Run tectonic
    cmd_tectonic = [TECTONIC_BIN, "--outdir", TIKZ_DIR, tex_path]
    res = subprocess.run(cmd_tectonic, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling {name}.tex:\n{res.stderr}\n{res.stdout}")
        return False

    # Copy vector PDF to paper/images/
    shutil.copy2(pdf_in_tikz, target_pdf)

    # Convert PDF to high-res PNG using qlmanage
    cmd_ql = ["qlmanage", "-t", "-s", "2400", "-o", IMG_DIR, target_pdf]
    subprocess.run(cmd_ql, capture_output=True)

    ql_rendered = os.path.join(IMG_DIR, f"{name}.pdf.png")
    if os.path.exists(ql_rendered):
        shutil.move(ql_rendered, target_png)
        print(f" -> Successfully exported: {name}.pdf & {name}.png (300 DPI+)")
        return True
    else:
        print(f" -> Warning: {name}.pdf created, but {ql_rendered} not found.")
        return False


if __name__ == "__main__":
    print(f"Starting TikZ compilation for {len(DIAGRAMS)} models...")
    print(f"TeX sources: {TIKZ_DIR}")
    print(f"Artifact outputs: {IMG_DIR}")

    success_count = 0
    for name, code in DIAGRAMS.items():
        if compile_diagram(name, code):
            success_count += 1

    print(f"\nFinished: {success_count}/{len(DIAGRAMS)} TikZ diagrams compiled successfully.")

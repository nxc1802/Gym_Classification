"""
Compile all 11 model architecture diagrams in publication-grade HORIZONTAL (landscape) layout.
Pure TikZ standalone -> PDF -> 300+ DPI PNG via macOS native qlmanage.

Notes:
- Strictly horizontal layout (flow left to right, compact height, wide width).
- Only essential architectural parameters (layers, dimensions, kernels, heads, pooling, params).
- Dual-Branch Early Fusion is excluded entirely.
"""

import os
import shutil
import subprocess

TIKZ_DIR = "/Volumes/WorkSpace/Project/Gym_Classification/paper/tikz"
IMG_DIR = "/Volumes/WorkSpace/Project/Gym_Classification/paper/images"
TECTONIC_BIN = "/opt/homebrew/bin/tectonic"

os.makedirs(TIKZ_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

PREAMBLE = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, calc, shapes.geometric, fit, backgrounds}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}

% Academic Color Palette
\definecolor{cInput}{RGB}{237, 242, 247}
\definecolor{bInput}{RGB}{74, 85, 104}

\definecolor{cProj}{RGB}{235, 248, 255}
\definecolor{bProj}{RGB}{49, 130, 206}

\definecolor{cRecurr}{RGB}{250, 245, 255}
\definecolor{bRecurr}{RGB}{128, 90, 213}

\definecolor{cTrans}{RGB}{238, 242, 255}
\definecolor{bTrans}{RGB}{79, 70, 229}

\definecolor{cGCN}{RGB}{230, 255, 250}
\definecolor{bGCN}{RGB}{13, 148, 136}

\definecolor{cPool}{RGB}{241, 245, 249}
\definecolor{bPool}{RGB}{100, 116, 139}

\definecolor{cHead}{RGB}{254, 242, 242}
\definecolor{bHead}{RGB}{225, 29, 72}

\definecolor{cFusion}{RGB}{254, 252, 232}
\definecolor{bFusion}{RGB}{202, 138, 4}

\definecolor{cOut}{RGB}{240, 253, 244}
\definecolor{bOut}{RGB}{22, 163, 74}

\definecolor{tDark}{RGB}{30, 41, 59}
\definecolor{tMuted}{RGB}{100, 116, 139}
"""

DIAGRAMS = {}

# =============================================================================
# 1. Unidirectional LSTM Baseline
# =============================================================================
DIAGRAMS["model_lstm"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.42cm,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.35cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

\node[box=bInput, fill=cInput, minimum width=2.0cm] (inp) {
    \textbf{Input Tensor}\\[2pt]
    {\scriptsize $[B, 32, 117]$}
};

\node[box=bProj, fill=cProj, minimum width=1.8cm, right=of inp] (norm) {
    \textbf{LayerNorm}\\[2pt]
    {\scriptsize $D = 117$}
};

\node[box=bRecurr, fill=cRecurr, minimum width=3.0cm, right=of norm] (lstm) {
    \textbf{2-Layer LSTM}\\[2pt]
    {\scriptsize $h=160$ \textbar{} drop=0.3}\\[1pt]
    {\scriptsize $[B, 32, 160]$}
};

\node[box=bPool, fill=cPool, minimum width=1.8cm, right=of lstm] (gap) {
    \textbf{Temp GAP}\\[2pt]
    {\scriptsize $\frac{1}{T}\sum_t \mathbf{h}_t \to [B, 160]$}
};

\node[box=bHead, fill=cHead, minimum width=2.2cm, right=of gap] (mlp) {
    \textbf{Dense + ReLU}\\[2pt]
    {\scriptsize $160 \to 128$ \textbar{} drop=0.3}
};

\node[box=bHead, fill=cHead, minimum width=2.0cm, right=of mlp] (cls) {
    \textbf{Classifier}\\[2pt]
    {\scriptsize $128 \to 21$}
};

\node[box=bOut, fill=cOut, minimum width=1.9cm, right=of cls] (out) {
    \textbf{Softmax $\hat{y}$}\\[2pt]
    {\scriptsize $\mathbf{p} \in \Delta^{21}$}
};

\draw[arr] (inp) -- (norm);
\draw[arr] (norm) -- (lstm);
\draw[arr] (lstm) -- (gap);
\draw[arr] (gap) -- (mlp);
\draw[arr] (mlp) -- (cls);
\draw[arr] (cls) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 2. Bidirectional LSTM Baseline
# =============================================================================
DIAGRAMS["model_bilstm"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.42cm,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.35cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

\node[box=bInput, fill=cInput, minimum width=2.0cm] (inp) {
    \textbf{Input Tensor}\\[2pt]
    {\scriptsize $[B, 32, 117]$}
};

\node[box=bProj, fill=cProj, minimum width=1.8cm, right=of inp] (norm) {
    \textbf{LayerNorm}\\[2pt]
    {\scriptsize $D = 117$}
};

\node[box=bRecurr, fill=cRecurr, minimum width=3.3cm, right=of norm] (bilstm) {
    \textbf{2-Layer BiLSTM}\\[2pt]
    {\scriptsize $h=96 \times 2 = 192$ \textbar{} drop=0.3}\\[1pt]
    {\scriptsize $[B, 32, 192]$}
};

\node[box=bPool, fill=cPool, minimum width=1.8cm, right=of bilstm] (gap) {
    \textbf{Temp GAP}\\[2pt]
    {\scriptsize $\frac{1}{T}\sum_t \mathbf{h}_t \to [B, 192]$}
};

\node[box=bHead, fill=cHead, minimum width=2.2cm, right=of gap] (mlp) {
    \textbf{Dense + ReLU}\\[2pt]
    {\scriptsize $192 \to 128$ \textbar{} drop=0.3}
};

\node[box=bHead, fill=cHead, minimum width=2.0cm, right=of mlp] (cls) {
    \textbf{Classifier}\\[2pt]
    {\scriptsize $128 \to 21$}
};

\node[box=bOut, fill=cOut, minimum width=1.9cm, right=of cls] (out) {
    \textbf{Softmax $\hat{y}$}\\[2pt]
    {\scriptsize $\mathbf{p} \in \Delta^{21}$}
};

\draw[arr] (inp) -- (norm);
\draw[arr] (norm) -- (bilstm);
\draw[arr] (bilstm) -- (gap);
\draw[arr] (gap) -- (mlp);
\draw[arr] (mlp) -- (cls);
\draw[arr] (cls) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 3. Transformer Encoder Baseline (Sequence Mix)
# =============================================================================
DIAGRAMS["model_transformer"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.42cm,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.35cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

\node[box=bInput, fill=cInput, minimum width=2.0cm] (inp) {
    \textbf{Input Tensor}\\[2pt]
    {\scriptsize $[B, 32, 117]$}
};

\node[box=bProj, fill=cProj, minimum width=2.2cm, right=of inp] (proj) {
    \textbf{Linear Proj + PE}\\[2pt]
    {\scriptsize $117 \to 128$ \textbar{} Sinusoid}\\[1pt]
    {\scriptsize $[B, 32, 128]$}
};

\node[box=bTrans, fill=cTrans, minimum width=3.8cm, right=of proj] (trans) {
    \textbf{4$\times$ Transformer Encoder}\\[2pt]
    {\scriptsize $H=4$ Heads \textbar{} $d_{\text{model}}=128$ \textbar{} $d_{\text{ff}}=256$}\\[1pt]
    {\scriptsize Pre-LN \textbar{} drop=0.1 \textbar{} $[B, 32, 128]$}
};

\node[box=bPool, fill=cPool, minimum width=1.8cm, right=of trans] (gap) {
    \textbf{Temp GAP}\\[2pt]
    {\scriptsize $\frac{1}{T}\sum_t \mathbf{z}_t \to [B, 128]$}
};

\node[box=bHead, fill=cHead, minimum width=2.2cm, right=of gap] (mlp) {
    \textbf{Dense + ReLU}\\[2pt]
    {\scriptsize $128 \to 64$ \textbar{} drop=0.1}
};

\node[box=bHead, fill=cHead, minimum width=2.0cm, right=of mlp] (cls) {
    \textbf{Classifier}\\[2pt]
    {\scriptsize $64 \to 21$}
};

\node[box=bOut, fill=cOut, minimum width=1.9cm, right=of cls] (out) {
    \textbf{Softmax $\hat{y}$}\\[2pt]
    {\scriptsize $\mathbf{p} \in \Delta^{21}$}
};

\draw[arr] (inp) -- (proj);
\draw[arr] (proj) -- (trans);
\draw[arr] (trans) -- (gap);
\draw[arr] (gap) -- (mlp);
\draw[arr] (mlp) -- (cls);
\draw[arr] (cls) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 4. ST-GCN Baseline
# =============================================================================
DIAGRAMS["model_stgcn"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.42cm,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.35cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

\node[box=bInput, fill=cInput, minimum width=2.0cm] (inp) {
    \textbf{Joint Graph}\\[2pt]
    {\scriptsize $[B, 3, 32, 13]$}
};

\node[box=bProj, fill=cProj, minimum width=1.6cm, right=of inp] (bn) {
    \textbf{DataBN}\\[2pt]
    {\scriptsize $C=3$}
};

\node[box=bGCN, fill=cGCN, minimum width=2.4cm, right=of bn] (s1) {
    \textbf{ST-GCN Block 1}\\[2pt]
    {\scriptsize $3 \to 48$, stride=1}\\[1pt]
    {\scriptsize Spatial GCN + TCN ($9\times 1$)}
};

\node[box=bGCN, fill=cGCN, minimum width=2.4cm, right=of s1] (s2) {
    \textbf{ST-GCN Block 2}\\[2pt]
    {\scriptsize $48 \to 96$, stride=2}\\[1pt]
    {\scriptsize Spatial GCN + TCN ($9\times 1$)}
};

\node[box=bGCN, fill=cGCN, minimum width=2.4cm, right=of s2] (s3) {
    \textbf{ST-GCN Block 3}\\[2pt]
    {\scriptsize $96 \to 150$, stride=2}\\[1pt]
    {\scriptsize Spatial GCN + TCN ($9\times 1$)}
};

\node[box=bPool, fill=cPool, minimum width=2.0cm, right=of s3] (gap) {
    \textbf{Global GAP}\\[2pt]
    {\scriptsize $V, T \to 1$}\\[1pt]
    {\scriptsize $[B, 150]$}
};

\node[box=bHead, fill=cHead, minimum width=1.8cm, right=of gap] (cls) {
    \textbf{Linear}\\[2pt]
    {\scriptsize $150 \to 21$}
};

\node[box=bOut, fill=cOut, minimum width=1.9cm, right=of cls] (out) {
    \textbf{Softmax $\hat{y}$}\\[2pt]
    {\scriptsize $\mathbf{p} \in \Delta^{21}$}
};

\draw[arr] (inp) -- (bn);
\draw[arr] (bn) -- (s1);
\draw[arr] (s1) -- (s2);
\draw[arr] (s2) -- (s3);
\draw[arr] (s3) -- (gap);
\draw[arr] (gap) -- (cls);
\draw[arr] (cls) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 5. AAGCN Backbone & Adaptive Graph Convolutional Block
# =============================================================================
DIAGRAMS["model_aagcn"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    node distance=0.42cm,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.35cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

% Macro pipeline
\node[box=bInput, fill=cInput, minimum width=2.0cm] (inp) {
    \textbf{Skeletal Stream}\\[2pt]
    {\scriptsize $[B, 3, 32, 13]$}
};

\node[box=bProj, fill=cProj, minimum width=1.5cm, right=of inp] (bn) {
    \textbf{DataBN}\\[2pt]
    {\scriptsize $C=3$}
};

\node[box=bGCN, fill=cGCN, minimum width=2.5cm, right=of bn] (stage1) {
    \textbf{Stage 1 (3 Blocks)}\\[2pt]
    {\scriptsize $C_{\text{out}}=48$, stride=1}\\[1pt]
    {\scriptsize Adaptive GCN + TCN}
};

\node[box=bGCN, fill=cGCN, minimum width=2.5cm, right=of stage1] (stage2) {
    \textbf{Stage 2 (3 Blocks)}\\[2pt]
    {\scriptsize $C_{\text{out}}=96$, stride=2}\\[1pt]
    {\scriptsize Adaptive GCN + TCN}
};

\node[box=bGCN, fill=cGCN, minimum width=2.5cm, right=of stage2] (stage3) {
    \textbf{Stage 3 (3 Blocks)}\\[2pt]
    {\scriptsize $C_{\text{out}}=150$, stride=2}\\[1pt]
    {\scriptsize Adaptive GCN + TCN}
};

\node[box=bPool, fill=cPool, minimum width=1.9cm, right=of stage3] (gap) {
    \textbf{Global GAP}\\[2pt]
    {\scriptsize $V, T \to 1$}\\[1pt]
    {\scriptsize $[B, 150]$}
};

\node[box=bHead, fill=cHead, minimum width=1.8cm, right=of gap] (fc) {
    \textbf{Classifier}\\[2pt]
    {\scriptsize $150 \to 21$}
};

\node[box=bOut, fill=cOut, minimum width=1.9cm, right=of fc] (out) {
    \textbf{Softmax $\hat{y}$}\\[2pt]
    {\scriptsize $\mathbf{p} \in \Delta^{21}$}
};

\draw[arr] (inp) -- (bn);
\draw[arr] (bn) -- (stage1);
\draw[arr] (stage1) -- (stage2);
\draw[arr] (stage2) -- (stage3);
\draw[arr] (stage3) -- (gap);
\draw[arr] (gap) -- (fc);
\draw[arr] (fc) -- (out);

% Inset badge explaining the Adaptive Topology rule: A_k = A_k \odot M_k + B_k + C_k
\node[draw=bGCN, fill=white, line width=0.8pt, rounded corners=2pt, font=\scriptsize, align=center, inner sep=3pt, below=0.35cm of stage2] (badge) {
    \textbf{Adaptive Graph Formulation:} $\mathbf{A}_k = A_k \odot M_k + B_k + C_k(\mathbf{X})$ \textbar{} TCN Kernel $K_t=9\times 1$
};
\draw[-, line width=0.6pt, dashed, color=bGCN] (stage2.south) -- (badge.north);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 6. Two-Stream AAGCN (Joint + Bone)
# =============================================================================
DIAGRAMS["model_fusion_2stream"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.15cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

% Stream 1 (Joint)
\node[box=bInput, fill=cInput, minimum width=2.2cm] (j_in) at (0, 0.85) {
    \textbf{Joint Stream}\\[1pt]
    {\scriptsize $\mathbf{X}_J \in \mathbb{R}^{B \times 3 \times 32 \times 13}$}
};

\node[box=bGCN, fill=cGCN, minimum width=3.4cm, right=0.6cm of j_in] (j_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize 9 Blocks ($48 \to 96 \to 150$)}
};

\node[box=bHead, fill=cHead, minimum width=1.9cm, right=0.6cm of j_net] (j_head) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_J \in \Delta^{21}$}
};

% Stream 2 (Bone)
\node[box=bInput, fill=cInput, minimum width=2.2cm] (b_in) at (0, -0.85) {
    \textbf{Bone Stream}\\[1pt]
    {\scriptsize $\mathbf{X}_B \in \mathbb{R}^{B \times 3 \times 32 \times 13}$}
};

\node[box=bGCN, fill=cGCN, minimum width=3.4cm, right=0.6cm of b_in] (b_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize 9 Blocks ($48 \to 96 \to 150$)}
};

\node[box=bHead, fill=cHead, minimum width=1.9cm, right=0.6cm of b_net] (b_head) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_B \in \Delta^{21}$}
};

% Fusion Node
\coordinate (mid_stream) at ($(j_head.east)!0.5!(b_head.east)$);
\node[box=bFusion, fill=cFusion, minimum width=3.2cm, right=0.8cm of mid_stream] (fuse) {
    \textbf{Weighted Late Fusion}\\[2pt]
    {\scriptsize $\mathbf{p} = 0.5\,\mathbf{p}_J + 0.5\,\mathbf{p}_B$}
};

% Output
\node[box=bOut, fill=cOut, minimum width=2.1cm, right=0.6cm of fuse] (out) {
    \textbf{Prediction $\hat{y}$}\\[1pt]
    {\scriptsize $\arg\max_{c} \mathbf{p}_c$}
};

% Routing
\draw[arr] (j_in) -- (j_net);
\draw[arr] (j_net) -- (j_head);
\draw[arr] (b_in) -- (b_net);
\draw[arr] (b_net) -- (b_head);

\draw[arr] (j_head.east) -| (fuse.north);
\draw[arr] (b_head.east) -| (fuse.south);
\draw[arr] (fuse) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 7. Three-Stream AAGCN (Joint + Bone + Motion)
# =============================================================================
DIAGRAMS["model_fusion_3stream"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=3.5pt,
        minimum height=1.05cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

% Stream 1 (Joint)
\node[box=bInput, fill=cInput, minimum width=2.2cm] (j_in) at (0, 1.25) {
    \textbf{Joint Stream}\\[1pt]
    {\scriptsize $\mathbf{X}_J \in [B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.2cm, right=0.5cm of j_in] (j_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize 9 Blocks ($48\to 96\to 150$)}
};
\node[box=bHead, fill=cHead, minimum width=1.8cm, right=0.5cm of j_net] (j_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_J \in \Delta^{21}$}
};

% Stream 2 (Bone)
\node[box=bInput, fill=cInput, minimum width=2.2cm] (b_in) at (0, 0.0) {
    \textbf{Bone Stream}\\[1pt]
    {\scriptsize $\mathbf{X}_B \in [B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.2cm, right=0.5cm of b_in] (b_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize 9 Blocks ($48\to 96\to 150$)}
};
\node[box=bHead, fill=cHead, minimum width=1.8cm, right=0.5cm of b_net] (b_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_B \in \Delta^{21}$}
};

% Stream 3 (Motion)
\node[box=bInput, fill=cInput, minimum width=2.2cm] (m_in) at (0, -1.25) {
    \textbf{Motion Stream}\\[1pt]
    {\scriptsize $\Delta_t \mathbf{X} \in [B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.2cm, right=0.5cm of m_in] (m_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize 9 Blocks ($48\to 96\to 150$)}
};
\node[box=bHead, fill=cHead, minimum width=1.8cm, right=0.5cm of m_net] (m_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_M \in \Delta^{21}$}
};

% Fusion Node
\node[box=bFusion, fill=cFusion, minimum width=3.3cm, right=0.8cm of b_p] (fuse) {
    \textbf{Late Soft Fusion}\\[2pt]
    {\scriptsize $\mathbf{p} = \frac{1}{3}\sum_{k \in \{J, B, M\}} \mathbf{p}_k$}
};

% Output
\node[box=bOut, fill=cOut, minimum width=2.0cm, right=0.5cm of fuse] (out) {
    \textbf{Prediction $\hat{y}$}\\[1pt]
    {\scriptsize $\arg\max \mathbf{p}$}
};

% Arrows
\draw[arr] (j_in) -- (j_net); \draw[arr] (j_net) -- (j_p);
\draw[arr] (b_in) -- (b_net); \draw[arr] (b_net) -- (b_p);
\draw[arr] (m_in) -- (m_net); \draw[arr] (m_net) -- (m_p);

\draw[arr] (j_p.east) -| (fuse.north);
\draw[arr] (b_p.east) -- (fuse.west);
\draw[arr] (m_p.east) -| (fuse.south);
\draw[arr] (fuse) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 8. Four-Stream AAGCN (Joint + Bone + J-Mot + B-Mot)
# =============================================================================
DIAGRAMS["model_fusion_4stream"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    box/.style={
        draw=#1,
        line width=1.05pt,
        rounded corners=3pt,
        align=center,
        inner sep=3pt,
        minimum height=0.92cm
    },
    arr/.style={->, line width=1.05pt, color=tDark!80}
]

% 4 Compact Streams
\node[box=bInput, fill=cInput, minimum width=2.1cm] (s1_in) at (0, 1.45) {
    \textbf{Joint $\mathbf{X}_J$}\\[1pt]
    {\scriptsize $[B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.0cm, right=0.45cm of s1_in] (s1_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize $48\to 96\to 150$}
};
\node[box=bHead, fill=cHead, minimum width=1.7cm, right=0.45cm of s1_net] (s1_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_1 \in \Delta^{21}$}
};

\node[box=bInput, fill=cInput, minimum width=2.1cm] (s2_in) at (0, 0.48) {
    \textbf{Bone $\mathbf{X}_B$}\\[1pt]
    {\scriptsize $[B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.0cm, right=0.45cm of s2_in] (s2_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize $48\to 96\to 150$}
};
\node[box=bHead, fill=cHead, minimum width=1.7cm, right=0.45cm of s2_net] (s2_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_2 \in \Delta^{21}$}
};

\node[box=bInput, fill=cInput, minimum width=2.1cm] (s3_in) at (0, -0.48) {
    \textbf{J-Motion $\Delta\mathbf{X}_J$}\\[1pt]
    {\scriptsize $[B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.0cm, right=0.45cm of s3_in] (s3_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize $48\to 96\to 150$}
};
\node[box=bHead, fill=cHead, minimum width=1.7cm, right=0.45cm of s3_net] (s3_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_3 \in \Delta^{21}$}
};

\node[box=bInput, fill=cInput, minimum width=2.1cm] (s4_in) at (0, -1.45) {
    \textbf{B-Motion $\Delta\mathbf{X}_B$}\\[1pt]
    {\scriptsize $[B, 3, 32, 13]$}
};
\node[box=bGCN, fill=cGCN, minimum width=3.0cm, right=0.45cm of s4_in] (s4_net) {
    \textbf{AAGCN Backbone}\\[1pt]
    {\scriptsize $48\to 96\to 150$}
};
\node[box=bHead, fill=cHead, minimum width=1.7cm, right=0.45cm of s4_net] (s4_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_4 \in \Delta^{21}$}
};

% Fusion Node
\coordinate (mid_p) at ($(s2_p.east)!0.5!(s3_p.east)$);
\node[box=bFusion, fill=cFusion, minimum width=3.2cm, right=0.7cm of mid_p] (fuse) {
    \textbf{Late Average Fusion}\\[2pt]
    {\scriptsize $\mathbf{p} = \frac{1}{4}\sum_{i=1}^4 \mathbf{p}_i$}
};

% Output
\node[box=bOut, fill=cOut, minimum width=2.0cm, right=0.5cm of fuse] (out) {
    \textbf{Prediction $\hat{y}$}\\[1pt]
    {\scriptsize $\arg\max \mathbf{p}$}
};

% Arrows
\draw[arr] (s1_in) -- (s1_net); \draw[arr] (s1_net) -- (s1_p);
\draw[arr] (s2_in) -- (s2_net); \draw[arr] (s2_net) -- (s2_p);
\draw[arr] (s3_in) -- (s3_net); \draw[arr] (s3_net) -- (s3_p);
\draw[arr] (s4_in) -- (s4_net); \draw[arr] (s4_net) -- (s4_p);

\draw[arr] (s1_p.east) -| (fuse.north);
\draw[arr] (s2_p.east) -- ([yshift=3pt]fuse.west);
\draw[arr] (s3_p.east) -- ([yshift=-3pt]fuse.west);
\draw[arr] (s4_p.east) -| (fuse.south);
\draw[arr] (fuse) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 9. SkelGym-Lite (Heterogeneous Dual-Stream: Sequence Mix + Bone GCN)
# =============================================================================
DIAGRAMS["model_fusion_skelgym_lite"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt,
        minimum height=1.25cm
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

% Stream 1: Sequence Mix (Transformer)
\node[box=bInput, fill=cInput, minimum width=2.4cm] (t_in) at (0, 0.95) {
    \textbf{Temporal Features}\\[1pt]
    {\scriptsize $[B, 32, 117]$}
};

\node[box=bTrans, fill=cTrans, minimum width=3.8cm, right=0.6cm of t_in] (t_net) {
    \textbf{Transformer Encoder}\\[1pt]
    {\scriptsize 4L \textbar{} $H=4$ \textbar{} $d=128$ \textbar{} GAP}
};

\node[box=bHead, fill=cHead, minimum width=1.9cm, right=0.6cm of t_net] (t_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_{\text{trans}} \in \Delta^{21}$}
};

% Stream 2: Bone GCN (AAGCN)
\node[box=bInput, fill=cInput, minimum width=2.4cm] (g_in) at (0, -0.95) {
    \textbf{Bone Graph}\\[1pt]
    {\scriptsize $[B, 3, 32, 13]$}
};

\node[box=bGCN, fill=cGCN, minimum width=3.8cm, right=0.6cm of g_in] (g_net) {
    \textbf{Bone Stream AAGCN}\\[1pt]
    {\scriptsize 9 Blocks ($48\to 96\to 150$) \textbar{} GAP}
};

\node[box=bHead, fill=cHead, minimum width=1.9cm, right=0.6cm of g_net] (g_p) {
    \textbf{Softmax}\\[1pt]
    {\scriptsize $\mathbf{p}_{\text{bone}} \in \Delta^{21}$}
};

% Fusion Node: SLSQP
\coordinate (mid_p) at ($(t_p.east)!0.5!(g_p.east)$);
\node[box=bFusion, fill=cFusion, minimum width=3.8cm, right=0.8cm of mid_p] (fuse) {
    \textbf{SLSQP Soft Fusion}\\[2pt]
    {\scriptsize $\mathbf{p}^* = w_1^* \mathbf{p}_{\text{trans}} + w_2^* \mathbf{p}_{\text{bone}}$}\\[1pt]
    {\scriptsize $\sum w_i = 1$ \textbar{} $w_i \ge 0$}
};

% Output
\node[box=bOut, fill=cOut, minimum width=2.0cm, right=0.6cm of fuse] (out) {
    \textbf{Prediction $\hat{y}$}\\[1pt]
    {\scriptsize $\arg\max \mathbf{p}^*$}
};

% Routing
\draw[arr] (t_in) -- (t_net); \draw[arr] (t_net) -- (t_p);
\draw[arr] (g_in) -- (g_net); \draw[arr] (g_net) -- (g_p);

\draw[arr] (t_p.east) -| (fuse.north);
\draw[arr] (g_p.east) -| (fuse.south);
\draw[arr] (fuse) -- (out);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 10. SkelGym-Full (Multi-Stream SLSQP + Temporal Video Consensus)
# =============================================================================
DIAGRAMS["model_fusion_skelgym_full"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

% Left: 5 Extractors for Window t
\node[box=bInput, fill=cInput, minimum width=4.6cm, minimum height=3.6cm] (streams) at (0, 0) {};
\node[font=\bfseries, text=bInput, anchor=north] at (streams.north) {5 Extractors (Window $t$)};

\node[box=bGCN, fill=cGCN, minimum width=4.0cm, minimum height=0.55cm, font=\scriptsize] (s1) at (0, 0.95) {1. Joint AAGCN $\to \mathbf{p}_1^{(t)}$};
\node[box=bGCN, fill=cGCN, minimum width=4.0cm, minimum height=0.55cm, font=\scriptsize] (s2) at (0, 0.35) {2. Bone AAGCN $\to \mathbf{p}_2^{(t)}$};
\node[box=bGCN, fill=cGCN, minimum width=4.0cm, minimum height=0.55cm, font=\scriptsize] (s3) at (0, -0.25) {3. Joint-Motion AAGCN $\to \mathbf{p}_3^{(t)}$};
\node[box=bGCN, fill=cGCN, minimum width=4.0cm, minimum height=0.55cm, font=\scriptsize] (s4) at (0, -0.85) {4. Bone-Motion AAGCN $\to \mathbf{p}_4^{(t)}$};
\node[box=bTrans, fill=cTrans, minimum width=4.0cm, minimum height=0.55cm, font=\scriptsize] (s5) at (0, -1.45) {5. Transformer Mix $\to \mathbf{p}_5^{(t)}$};

% Stage 1: Window SLSQP
\node[box=bFusion, fill=cFusion, minimum width=4.1cm, minimum height=2.2cm, right=1.3cm of streams] (stage1) {
    \textbf{Stage 1: Window SLSQP}\\[3pt]
    {\scriptsize $\mathbf{P}^{(t)} = \sum_{k=1}^5 w_k^* \mathbf{p}_k^{(t)}$}\\[2pt]
    {\scriptsize $\min_{\mathbf{w}} \mathcal{L}_{\text{val}} \text{ s.t. } \sum w_k = 1, w_k \ge 0$}
};

% Stage 2: Temporal Video Consensus
\node[box=bProj, fill=cProj, minimum width=4.2cm, minimum height=2.2cm, right=1.1cm of stage1] (stage2) {
    \textbf{Stage 2: Video Consensus}\\[3pt]
    {\scriptsize Aggregation across $W$ Windows:}\\[2pt]
    {\scriptsize $\bar{\mathbf{P}} = \frac{1}{W}\sum_{t=1}^W \mathbf{P}^{(t)}$}
};

% Output
\node[box=bOut, fill=cOut, minimum width=2.4cm, minimum height=2.2cm, right=0.9cm of stage2] (out) {
    \textbf{Prediction $\hat{y}$}\\[3pt]
    {\scriptsize $\arg\max_c \bar{\mathbf{P}}_c$}\\[2pt]
    {\scriptsize Video-Level}
};

\draw[arr] (streams.east) -- node[above=2pt, font=\scriptsize] {$\{\mathbf{p}_k^{(t)}\}_{k=1}^5$} (stage1.west);
\draw[arr] (stage1.east) -- node[above=2pt, font=\scriptsize] {$\mathbf{P}^{(t)}$} (stage2.west);
\draw[arr] (stage2.east) -- node[above=2pt, font=\scriptsize] {$\bar{\mathbf{P}}$} (out.west);

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# 11. Stacking Meta-Learner (Level-0 + Level-1)
# =============================================================================
DIAGRAMS["model_ensemble_stacking"] = PREAMBLE + r"""
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    box/.style={
        draw=#1,
        line width=1.1pt,
        rounded corners=3.5pt,
        align=center,
        inner sep=4pt
    },
    arr/.style={->, line width=1.1pt, color=tDark!80}
]

% Level-0 Box
\node[box=bInput, fill=cInput, minimum width=4.4cm, minimum height=3.6cm] (l0) at (0, 0) {};
\node[font=\bfseries, text=bInput, anchor=north] at (l0.north) {Level-0 Base Classifiers};

\node[box=bTrans, fill=cTrans, minimum width=3.8cm, minimum height=0.55cm, font=\scriptsize] (m1) at (0, 0.95) {Transformer ($d=128$, 4L)};
\node[box=bRecurr, fill=cRecurr, minimum width=3.8cm, minimum height=0.55cm, font=\scriptsize] (m2) at (0, 0.35) {LSTM ($h=160$, 2L)};
\node[box=bRecurr, fill=cRecurr, minimum width=3.8cm, minimum height=0.55cm, font=\scriptsize] (m3) at (0, -0.25) {BiLSTM ($h=192$, 2L)};
\node[box=bGCN, fill=cGCN, minimum width=3.8cm, minimum height=0.55cm, font=\scriptsize] (m4) at (0, -0.85) {ST-GCN (3 Stages)};
\node[box=bGCN, fill=cGCN, minimum width=3.8cm, minimum height=0.55cm, font=\scriptsize] (m5) at (0, -1.45) {AAGCN (9 Blocks)};

% Meta-Feature Generator
\node[box=bFusion, fill=cFusion, minimum width=3.6cm, minimum height=2.2cm, right=1.4cm of l0] (meta_feat) {
    \textbf{Meta-Features $\mathbf{z}$}\\[3pt]
    {\scriptsize OOF Probabilities: $5 \times 21 = 105$}\\[2pt]
    {\scriptsize Entropy Metrics: $5$}\\[2pt]
    {\scriptsize $\mathbf{z}_{\text{meta}} \in \mathbb{R}^{110}$}
};

% Level-1 Meta-Classifier
\node[box=bHead, fill=cHead, minimum width=3.8cm, minimum height=2.2cm, right=1.2cm of meta_feat] (l1) {
    \textbf{Level-1 Meta-Learner}\\[3pt]
    {\scriptsize Multinomial Logistic Reg.}\\[2pt]
    {\scriptsize $L_2$ Regularization ($C=1.0$)}\\[2pt]
    {\scriptsize Input: $\mathbb{R}^{110} \to \Delta^{21}$}
};

% Output
\node[box=bOut, fill=cOut, minimum width=2.2cm, minimum height=2.2cm, right=0.9cm of l1] (out) {
    \textbf{Final $\hat{y}$}\\[3pt]
    {\scriptsize Calibrated}\\[2pt]
    {\scriptsize Prediction}
};

\draw[arr] (l0.east) -- node[above=2pt, font=\scriptsize] {OOF $\mathbf{p}^{(k)}$} (meta_feat.west);
\draw[arr] (meta_feat.east) -- node[above=2pt, font=\scriptsize] {$\mathbf{z}_{\text{meta}}$} (l1.west);
\draw[arr] (l1.east) -- node[above=2pt, font=\scriptsize] {$\mathbf{p}_{\text{final}}$} (out.west);

\end{tikzpicture}
\end{document}
"""


def compile_diagram(name, latex_code):
    tex_path = os.path.join(TIKZ_DIR, f"{name}.tex")
    pdf_in_tikz = os.path.join(TIKZ_DIR, f"{name}.pdf")
    target_pdf = os.path.join(IMG_DIR, f"{name}.pdf")
    target_png = os.path.join(IMG_DIR, f"{name}.png")

    print(f"\n[Compiling Horizontal TikZ] {name}.tex ...")
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
        print(f" -> Exported: {name}.pdf & {name}.png (300+ DPI, Horizontal)")
        return True
    else:
        print(f" -> Warning: {name}.pdf created, but {ql_rendered} not found.")
        return False


if __name__ == "__main__":
    print(f"Generating and compiling {len(DIAGRAMS)} horizontal architecture diagrams...")
    print(f"TikZ Sources: {TIKZ_DIR}")
    print(f"Images Output: {IMG_DIR}")

    success_count = 0
    for name, code in DIAGRAMS.items():
        if compile_diagram(name, code):
            success_count += 1

    print(f"\n=======================================================")
    print(f"Completed: {success_count}/{len(DIAGRAMS)} horizontal diagrams compiled successfully.")
    print(f"=======================================================")

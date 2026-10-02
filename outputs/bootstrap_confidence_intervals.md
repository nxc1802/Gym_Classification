# Cluster Bootstrap 95% Confidence Intervals (B=1000 resamples)

Evaluated on held-out test partitions ($N=2743$ temporal windows across $K=233$ independent action videos).
Window-level uncertainty is evaluated via **Cluster Bootstrap by source video ID**, correctly accounting for intra-video temporal autocorrelation.

| Architecture | Window Accuracy (Cluster 95% CI) | Window Macro F1 (Cluster 95% CI) | Video Accuracy (95% CI) | Video Macro F1 (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| LSTM (Mix 117-d) | 59.29% [52.85%, 65.51%] | 0.5670 [0.5078, 0.6281] | 68.81% [63.09%, 74.68%] | 0.6656 [0.6054, 0.7275] |
| BiLSTM (Mix 117-d) | 60.44% [53.62%, 66.61%] | 0.5733 [0.5109, 0.6324] | 67.93% [61.79%, 73.39%] | 0.6429 [0.5763, 0.7049] |
| ST-GCN (Rel 3D) | 58.83% [52.77%, 64.61%] | 0.5505 [0.4895, 0.6085] | 67.01% [60.94%, 72.97%] | 0.6349 [0.5638, 0.7011] |
| Transformer (Mix 117-d) | 71.55% [64.67%, 77.99%] | 0.6822 [0.6239, 0.7438] | 81.98% [76.81%, 86.70%] | 0.7835 [0.7254, 0.8429] |
| AAGCN (Bone 3D) | 65.40% [58.93%, 71.64%] | 0.6369 [0.5780, 0.6907] | 73.32% [67.81%, 78.97%] | 0.7199 [0.6672, 0.7755] |
| **SkelGym-Lite** | 69.57% [62.88%, 75.83%] | 0.6693 [0.6118, 0.7263] | 77.63% [72.10%, 82.83%] | 0.7530 [0.6946, 0.8096] |
| **SkelGym-Full** | 71.02% [64.30%, 77.31%] | 0.6871 [0.6278, 0.7441] | 79.35% [73.82%, 84.55%] | 0.7546 [0.6941, 0.8151] |

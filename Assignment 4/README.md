# Assignment 4 – Computer Vision Lab

| File | Content |
|---|---|
| Q1_Three_Geometric_Models.ipynb | H vs F for poster (A), chair scene (B), synthetic pure rotation (C) |
| Q2_Parallax_Trap.ipynb | Fulgar pair: background-H vs near-object-H, APAP improvement |
| Q3_Stereo_Depth_Reliability.ipynb | chair pair: rectification, SGBM, depth, failure regions, reliability map |
| Q4_Optical_Flow.ipynb | SIFT + LK + DIS flow, flow-dance video |
| Q4_algorithm_description.txt | algorithm text file for Q4 |
| video/ | input clip, flow_dance_on_black.mp4, flow_vs_original_side_by_side.mp4 |
| images/ | input photos (+ synthetic rotation pair) |
| outputs/ | all figures and CSV tables (outputs/cache = cached matches) |

Run notebooks from this folder (needs opencv-contrib, numpy, scipy, matplotlib, pandas, ffmpeg).
Assumptions: Q1-C is synthetic (no real rotation pair given); Q3 assumes f=1400 px (full res), B=0.15 m (no calibration given).

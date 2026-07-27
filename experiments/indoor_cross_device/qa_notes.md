# QA notes

## Claim and evidence contract

- Core conclusion: one Handle1 prior map supports two independent complete
  localization replays, and the conditional dual-envelope residual recovers the
  Vehicle1 sensor-origin height difference.
- Hero evidence: the real prior-map raster and all three measured trajectories.
- Validation evidence: online initialization fitness/overlap and estimated
  versus localization-derived pseudo-reference `delta_z`.
- Scope limits: all recordings use the same MID-360 model; the height reference
  is not survey ground truth; 14/17 are the numbers of spatially correct
  hypotheses on which vertical error is defined; two successful complete bags
  are case studies rather than a statistical success-rate estimate.
- Table-III conclusion: the new sparse indoor columns document transfer
  behavior but do not support retrieval superiority; LiDAR Iris leads both.
- Table archetype: quantitative comparison grid. Bold marks the column best and
  underline marks the column second-best, including ties.

## Automated and visual checks

- Python syntax and repository `git diff --check`: pass.
- The generic Nature static validator parses the source and reports no simulated
  data, unsafe colour map, row sampling, or mixed plotting backend.
- Its matplotlib-specific font-rcParams, SVG/PDF, TIFF, DPI, and physical-width
  checks are not applicable to this Manim video source. The delivered artifact
  is H.264 video, not a manuscript figure export.
- Final video: 1920 x 1080, 30 fps, H.264/yuv420p.
- Inspected frames: intro, real map/trajectory overlay, query-label transition,
  circular online-initialization dashboard, vertical ruler, closing card, and
  both integrated-video cross-fades.
- No clipping or persistent text overlap was observed after the final rerender.
- The expanded double-column Table III was inspected from both English and
  Chinese compiled PDFs. All eight columns remain inside the rule width and
  labels and best/second-best marks remain legible.

## Traceability

The video reads values from
`results/experiment_summary.json`; the paper table reports the same rounded
values. The real map and trajectories are loaded from `scans.pcd`,
`optimized_poses_tum.txt`, and the two `trusted_pose.csv` files.
The two added retrieval columns are traceable to
`results/table3_retrieval_columns.csv`. They use 58 and 57 eligible
single-scan queries, a 2 m correctness radius, and the same gravity
canonicalization front end for every reported method; empty candidate sets are
retained as failures. No confidence interval is claimed because each column is
one deterministic traversal rather than repeated random trials.

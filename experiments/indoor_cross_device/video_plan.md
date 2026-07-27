# Indoor cross-device/platform experiment chapter

## Overview

- **Topic**: one hand-carried indoor map reused by hand-carried and
  vehicle-mounted query traversals.
- **Hook**: one prior map, two independently recorded acquisition setups.
- **Aha moment**: both independently recorded query bags are replayed
  continuously on the same real prior map.
- **Audience**: robotics and LiDAR place-recognition reviewers.
- **Length**: approximately 33 seconds, appended to the existing experiment
  video.
- **Resolution**: 1920 x 1080, 30 fps, no narration or music.

## Evidence and integrity

- Map raster: real Handle1 prior PCD.
- Blue trajectory: real Handle1 mapping keyframes.
- Orange/green trajectories: trusted online localization poses recorded from
  the complete Handle2 and Vehicle1 bag replays.
- No simulated trajectories, metrics, or point clouds.

## Color palette

- Background: `#08111F`
- Map context: `#657B8C`
- Mapping track: `#438BDE`
- Handle query: `#F39A45`
- Vehicle query: `#37C68A`
- Vertical/reference accent: `#E6C65A`

## Arc: problem--solution data story

1. **Setup (~3 s)**: introduce one map and two query devices/platforms.
2. **Trajectory context (~12 s)**: draw the shared prior-map trajectory, then
   replay the complete Handle2 and Vehicle1 trusted localization tracks.
3. **Measured retrieval replay (~14 s)**: a two-by-two panel grid compares
   Scan Context + G and UpDown-SC (columns) on Handle2 and Vehicle1 (rows).
   Every measured query-to-Top-1 association and running R@1 is animated.
4. **Measured ending (~2 s)**: hold the completed four-panel retrieval result.
   Do not append a last-dataset-specific closing card to the integrated video.

## Review checks

- Keep every query center inside the real map bounds.
- Label this as cross-device/platform with the same MID-360 model, not
  heterogeneous-sensor generalization.
- Do not present two successful replays as a population success rate.
- Do not show sensor-height estimation or ICP timing in this chapter.
- Query-to-candidate lines and running recall must come from the measured
  per-query result CSVs.

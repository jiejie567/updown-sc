# Public Factory-Hall Cross-Sequence Relocalization

## Overview

- **Topic**: Cross-traversal LiDAR place recognition and relocalization in the
  public RTK-SLAM factory hall.
- **Hook**: Can a map built clockwise recognize a counter-clockwise revisit?
- **Misconception corrected**: Fixed-time query sampling is not a neutral
  keyframe protocol when platform speed varies.
- **Aha moment**: The proposed descriptor preserves complementary vertical
  evidence, so the correct map place rises in rank on the reverse traversal.
- **Target audience**: LiDAR SLAM and place-recognition researchers.
- **Length**: approximately 34 seconds as one chapter of the integrated video.
- **Resolution**: 854x480 draft, 1920x1080 production.
- **Data policy**: Every cloud, pose, match and metric comes from the released
  RTK-SLAM factory-hall sequences or generated experiment CSVs. No mock values.

## Color palette

- Background: `#071521` — dark point-cloud plate.
- Map traversal: `#55A7FF` — Seq.1 database.
- Query traversal: `#F3C969` — Seq.2 queries.
- Proposed method: `#E76F51`.
- Correct match: `#57CC99`.
- Incorrect candidate/context: `#8496A8` at reduced opacity.

## Arc: Problem--Solution

## Scene 1: Two traversals, one place (~9 s)

**Purpose**: Establish the real cross-sequence protocol.

**Layout**: Full-screen top-view point-cloud map.

1. Reveal the public factory-hall map.
2. Draw the longer Seq.1 trajectory and label it `Map: Seq.1 (0.48 km)`.
3. Draw the reverse Seq.2 trajectory and label it `Query: Seq.2 (0.39 km)`.
4. Keep a thin divider between the two traversal labels.

**Subtitle**: “The longer clockwise traversal builds the database; the
counter-clockwise traversal is never inserted into it.”

## Scene 2: Experiment keyframes follow traveled distance (~11 s)

**Purpose**: Make the sampling protocol visually auditable.

**Layout**: Enlarged trajectory segment with a small clock in the background.

1. Move a query marker along the continuous trajectory.
2. Fade regular clock ticks to low opacity.
3. Retain the first frame, then emit a keyframe dot when 3-D translation from
   the last retained frame reaches 2 m.
4. State that time and yaw triggers are disabled for the experiment and that
   the production FAST-LIO keyframe policy is unchanged.

**Subtitle**: “Both database and query use the same translation-only 2 m
experiment rule.”

## Scene 3: One real reverse-view query (~13 s)

**Purpose**: Show the experimental process rather than a generic FAST-LIO demo.

**Layout**: Left: real query cloud. Right: one SC circle followed by two
equally sized, side-by-side UpDown-SC envelope circles. A vertical line
separates the query from the descriptors; method and channel labels occupy
separate rows above the circles.

1. Reveal one actual gravity-canonicalized Seq.2 cloud.
2. Show `SC + gravity` above the first circular descriptor.
3. Show `UpDown-SC` above the second circular descriptor.
4. Split the second descriptor into `Lower envelope` and `Upper envelope`.
5. Animate candidate ranks changing; mark the first geometrically correct
   Seq.1 place.

**Subtitle**: “Complementary lower and upper envelopes retain vertical evidence
that a single maximum-height image can saturate.”

## Scene 4: Retrieval along the traversal (~16 s)

**Purpose**: Visualize the actual cross-sequence evaluation.

**Layout**: Map with moving query marker; circular dashboard at lower right.

1. Advance through spatial keyframes in timestamp order.
2. Draw the Top-1 candidate link for each method.
3. Color the link green only when the candidate lies within the predeclared
   correctness radius.
4. Update circular Recall@1 counters from the exported per-query CSVs.
5. Do not suppress failures or non-overlap intervals; mark excluded non-overlap
   keyframes in gray.

**Subtitle**: “Every eligible query contributes once; failures remain visible.”

## Scene 5: Final comparison and real localization (~14 s)

**Purpose**: Connect retrieval recall to the full relocalization use case.

**Layout**: Recall chart transitions into the real localized trajectory.

1. Reveal the latest Recall@1 results for the shared
   gravity-canonicalized single-scan protocol and the final adaptive-split
   UpDown-SC result.
2. Highlight UpDown-SC without hiding competing methods.
3. Transform the proposed-method bar into the Seq.2 localized trajectory on
   the Seq.1 prior map.
4. End with the map/query protocol and keyframe rule as small reproducibility
   text.

**Subtitle**: “Higher retrieval recall supplies better global hypotheses for
prior-map localization.”

## Review checklist

- Method labels sit above, never inside, the circular descriptors.
- A divider separates baseline and proposed descriptor panels.
- Map, query, correct match and incorrect match colors remain consistent.
- No element uses fixed-time or yaw-triggered experimental subsampling.
- Production FAST-LIO remains unchanged; only experiment datasets are thinned.
- All counters are computed from CSVs during rendering.
- The SC comparison reads the same +G protocol as the proposed method.
- Every reveal has breathing room and a clean visual exit.

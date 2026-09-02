---
name: updown-sc-deployment
description: Build, validate, or reproduce UpDown-SC from this repository, including the standalone C++ tools, optional Python utilities, and released evaluation bundles. Do not use for unrelated LiDAR projects or for training new learned models.
---

# Deploy UpDown-SC

Produce a verified local research deployment from this checkout. Treat
"deploy" as installing and validating the standalone command-line tools, not
as publishing a hosted service.

## Select the requested scope

- For a normal deployment, build the four C++ tools and verify their presence.
- Add the Python environment only when evaluation or figure scripts are needed.
- Run the release replay only when the checked result bundles are available.
- Run learned baselines only when the user explicitly requests them and the
  upstream repositories and official checkpoints are available.

Do not download checkpoints, clone upstream repositories, install system
packages, or publish artifacts without the user's authorization. If a
dependency is missing, report the exact missing component and the appropriate
package-manager command, but do not assume the operating system.

## Build the standalone tools

Work from the repository root, which must contain `updown_sc/CMakeLists.txt`.
The required toolchain is CMake 3.16 or newer, a C++17 compiler, Eigen3, PCL
(`common` and `io`), and yaml-cpp.

```bash
cmake -S updown_sc -B build/updown-sc -DCMAKE_BUILD_TYPE=Release
cmake --build build/updown-sc -j2
```

Verify that these executables exist:

```text
build/updown-sc/scan_context_rebuild
build/updown-sc/scan_context_cross_sequence_evaluator
build/updown-sc/scan_context_convert
build/updown-sc/scan_context_subset
```

Keep build products under `build/`; do not write generated files into
`updown_sc/`.

## Optional Python utilities

Use Python 3.10 or newer for the non-ROS evaluation and plotting utilities.
Create an isolated environment only when those utilities are requested:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-figures.txt` is optional and should be installed only for
figure or presentation regeneration.

## Validate released evidence

For a lightweight integrity check:

```bash
(cd data/package && sha256sum -c SHA256SUMS)
```

For the repository's complete build, archive-safety, and deterministic M2DGR
replay check, run:

```bash
./scripts/release_check.sh
```

This check requires the compact archives named by `data/package/SHA256SUMS`.
If an archive is absent, distinguish a missing data input from a code or build
failure. Read `data/README.md` for archive layout and `docs/PROVENANCE.md` only
when reproducing paper tables or learned-baseline rows.

Archived path fields may contain the literal `${UPDOWN_SC_ROOT}` placeholder.
Interpret it as the extraction root; do not replace it with a contributor's
absolute home path in committed files.

## Learned-baseline boundary

The OverlapTransformer and MinkLoc3Dv2 adapters do not vendor upstream model
code or weights. Before running either adapter, read its `--help` output and
the corresponding section of `docs/PROVENANCE.md`. Preserve the recorded
upstream commit, checkpoint checksum, preprocessing, and runtime; do not
silently substitute a different checkpoint or environment.

## Report completion

State the build directory, the verified executables, checksum/replay results,
and any skipped optional scope. Include exact error output for a failure, but
do not claim full reproducibility when only the C++ build was tested.

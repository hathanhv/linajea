# Biohub I/O on Kaggle

The Biohub adapter reads Zarr v3 images and GEFF annotations directly. It
does not copy the input dataset. The spatial coordinate unit passed to Linajea
is **1/32 micrometer**; the sample spacing `1.625, 0.40625, 0.40625` micrometer
therefore becomes the exact integer Gunpowder spacing `52, 13, 13`.

You can upload [`check_biohub_io_kaggle.ipynb`](check_biohub_io_kaggle.ipynb)
to Kaggle or copy its cells into a new notebook. Attach the competition data
and run with Internet enabled. Clone the branch or, for a reproducible run, replace
`biohub-io` with the commit hash you pushed:

```python
!git clone --branch biohub-io https://github.com/hathanhv/linajea.git
%cd linajea
!pip install -r requirements-biohub-io.txt
!pip install -e . --no-deps
```

```python
!python run_scripts/check_biohub_io.py \
  --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train
```

Use `--sample SAMPLE_STEM` to check a specific pair. If Kaggle mounts the
competition under a different path, pass that train directory to `--data-dir`.
The command reads a small image chunk, validates the full GEFF graph, then
requests one `3 x 8 x 64 x 64` image/graph batch through Gunpowder. Success
ends with `Biohub I/O smoke test passed`. No training or database is started.

For later training, set `general.sparse = true`, point `datafile.filename` to
the Biohub `.zarr`, set `datafile.array = "0"`, and point `tracksfile` to the
paired `.geff`. Use these per-sample normalization settings in both train and
predict config:

```toml
[train.normalization]
type = "percminmax"
perc_min = "q001"
perc_max = "q099"
```

The reader maps `q001` and `q099` to the `0.01` and `0.99` quantiles in the
image root metadata. Values for spatial radii and distance thresholds in
Linajea config must also use 1/32 micrometer units. Do not reuse the physical
distance values from the MSKCC examples unchanged.

## Short training check

Upload [`train_biohub_smoke_kaggle.ipynb`](train_biohub_smoke_kaggle.ipynb)
to Kaggle, attach the same competition data, and enable Internet and a GPU.
After cloning your pushed `biohub-io` branch, it installs the training
dependencies and runs two iterations on the annotated sample
`6bba_05b6850b`. The notebook prints the exact commit, iteration losses,
and the checkpoint path. Training output, including the resolved config,
is written to `/kaggle/working/biohub_train_smoke`; the competition Input is
read-only.

For a shell run, install `requirements-biohub-io.txt` and
`requirements-biohub-train.txt`, then install the pinned Daisy and UNet
packages without their legacy dependencies:

```sh
python -m pip install -r requirements-biohub-io.txt
python -m pip install -r requirements-biohub-train.txt
python -m pip install --no-deps "git+https://github.com/abred/daisy@ab7b82054126a50ab4ad5dcf2ef885b02d33aab8"
python -m pip install --no-deps "git+https://github.com/Kainmueller-Lab/funlib.learn.torch@36ef666d16bddda0bf3146d4c0efbe9b95c8463d"
python -m pip install -e . --no-deps
python -m run_scripts.train_biohub_smoke \
  --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --sample 6bba_05b6850b \
  --output-dir /kaggle/working/biohub_train_smoke \
  --iterations 2
```

This check covers sampling, label rasterization, forward and backward passes,
and a finite checkpoint. It does not measure tracking quality. Full inference,
solver, and submission still require separate work.

## Random node hide pilot

Upload [`random_hide_kaggle.ipynb`](random_hide_kaggle.ipynb) to Kaggle after
pushing this branch. Attach the Biohub competition data, enable Internet and a
GPU, and run all cells. To pin a pushed revision, set `PINNED_COMMIT` in the
settings cell to its SHA. The notebook checks out and records the resolved
commit in `commit.txt`, then uses sample `6bba_05b6850b`, seed 42,
and a target hide rate of 20%. The annotation-rich evaluation ROI is selected
before hiding. Its 20 frames and `32 × 128 × 128` voxels are for prediction
and evaluation; training samples the visible graph over the full 100 frames.

The split writes `visible.geff` with the original node IDs and only edges
whose endpoints remain visible. It also writes `split.json` with the hidden
IDs, held-out edges, actual hide rate, and ROI. The training config points
only to `visible.geff`. The original GEFF is read during splitting and
evaluation, never by the training or inference stages. Hidden centers within
the 2 µm training mask radius are grouped before splitting and checked for
overlap afterwards.

The notebook trains for 2,000 iterations and saves checkpoints at iterations
500, 1000, 1500, and 2000. Inference uses the training quantile
normalization, non-overlapping output tiles, center NMS, predicted movement
vectors, and SciPy MILP with Linajea's basic forest constraints and a fixed
example parameter set. The final cell displays the same raw image projection
in two panels: visible/hidden GT on the left and the selected reconstructed
graph on the right. An optional interactive 3D view is saved as HTML.

All outputs are under `/kaggle/working/biohub_random_hide/`: `train/config.toml`,
`train/train.log`, checkpoints, `prediction/pred_nodes.csv`,
`prediction/pred_edges.csv`, candidate CSVs, `prediction/reconstruction.json`,
`evaluation/metrics.json`, `plots/comparison.png`, and
`plots/comparison_3d.html`. Use Kaggle **Save Version / Save & Run All** to
retain them as notebook outputs. The reported hidden-confirmed pseudo-label
rate is a lower bound: an unmatched prediction can be a real cell absent from
Biohub's sparse GT. A 2,000-iteration pilot validates the workflow and does
not establish final reconstruction quality.

### Re-solve an empty graph without retraining

If candidate nodes exist but the example ILP weights select no cells, push the
updated `biohub-io` branch, then **copy the code cells** from
[`calibrate_ilp_kaggle.ipynb`](calibrate_ilp_kaggle.ipynb) into the existing
Kaggle pilot notebook and run them in the same live session. A separate Kaggle
notebook has its own `/kaggle/working` and cannot see the pilot outputs by
default. Keep `/kaggle/working/biohub_random_hide` intact. The follow-up cells
fetch the new commit, reuse the cached candidate CSVs,
and never reopens the training checkpoint or repeats tiled inference.

Calibration reads only `visible.geff` and the ROI coordinates from `split.json`.
It tries a fixed grid of node selection constants and track appearance costs;
the score weight, edge weight, division cost, and ILP constraints stay fixed.
At a 4 µm match distance, it selects the smallest graph retaining at least
80% of the visible node and edge matches supported by the candidate graph.
Unmatched candidates are not counted as false positives. If the grid cannot
meet both targets, the report says so and chooses the best visible coverage
found in that grid.

The selected costs, all 36 trials, and the graph are saved under
`prediction_calibrated/`. Hidden-GT evaluation and comparison figures are
saved separately under `evaluation_calibrated/` and `plots_calibrated/`.
Original pilot outputs are preserved. Since the first hidden-GT result was
already inspected, this follow-up is **exploratory** and should not be reported
as an independent test of pseudo-label quality.

### Compare training checkpoints in a fresh Kaggle session

Upload [`compare_checkpoints_kaggle.ipynb`](compare_checkpoints_kaggle.ipynb)
as a **new standalone Kaggle notebook**, attach the Biohub competition data,
enable GPU and Internet, and run all cells. This version starts from an empty
`/kaggle/working`: it clones the fork, creates the same seeded visible/hidden
split, trains for 2,000 iterations, and saves checkpoints at 500, 1000, 1500,
and 2000. It then creates and calibrates the 2,000-iteration graph. The
comparison reuses those 2,000-iteration candidates and runs tiled inference
for only the first three checkpoints. All four are evaluated on the same ROI
with the same 2, 4, and 7 µm matching thresholds. The calibrated ILP costs
from iteration 2000 are held fixed for the earlier selected graphs.

The final cell shows a row per iteration with candidate and selected counts,
visible and hidden cell recovery, and hidden edge recovery at 4 µm. Complete
metrics and a machine-readable report are saved in
`/kaggle/working/biohub_random_hide/checkpoint_comparison/`.
Use the **visible candidate recall trend** as a training diagnostic; hidden
metrics remain exploratory. If candidate coverage is still rising at iteration
2000, continue training to later checkpoints. If it plateaus, inspect
sampling, candidate threshold, and model capacity before adding iterations.

### Three 1500-iteration seeds and a full GEFF export

Upload [`multiseed_1500_full_geff_kaggle.ipynb`](multiseed_1500_full_geff_kaggle.ipynb)
to a new Kaggle notebook, attach the competition train data, and enable GPU
and Internet. Push the notebook and Python changes to `biohub-io` first. The
notebook runs three fresh 1500-iteration experiments with hide seeds 42, 43,
and 44, each in a separate output folder under
`/kaggle/working/biohub_random_hide_multiseed/`. Its table compares ROI
candidate and selected graph recovery at 4 µm. Training uses each seed's
visible GEFF only. Loss logs, checkpoints, split manifests, calibration grids,
and metrics are saved for every seed.

The full-sequence export always uses seed 42, fixed before reading hidden
results. It streams inference across all 100 frames of sample
`6bba_05b6850b`, solves disconnected candidate components with the
visible-only calibrated ILP costs, and writes
`seed_42/full_export/predicted_full.geff` plus
`seed_42/full_export/predicted_full_geff.zip`. The `.geff` is a Zarr directory;
download and extract the ZIP before opening that folder with the
[`napari-geff`](https://github.com/live-image-tracking-tools/napari-geff)
plugin. Open the matching Biohub `.zarr` image separately to overlay tracks.
This export is a prediction for the **one training sample**; it is not an
independent test or a Kaggle submission. Full-volume inference is much longer
than ROI inference and reports progress every five frames.

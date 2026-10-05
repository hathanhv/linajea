# Biohub I/O on Kaggle

The Biohub adapter reads Zarr v3 images and GEFF annotations directly. It
does not copy the input dataset. The spatial coordinate unit passed to Linajea
is **1/32 micrometer**; the sample spacing `1.625, 0.40625, 0.40625` micrometer
therefore becomes the exact integer Gunpowder spacing `52, 13, 13`.

Create a Kaggle notebook, attach the competition data, and run these cells
with Internet enabled. Clone the branch or, for a reproducible run, replace
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

The full training and tracking pipeline still needs its other dependencies,
including legacy Daisy, MongoDB, and `pylp`; this I/O check does not test them.

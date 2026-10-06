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

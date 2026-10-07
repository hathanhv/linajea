"""Side-by-side hidden-label recovery views for one Biohub frame."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import match_nodes, read_rows


def plot_comparison(data_dir, manifest_path, nodes_path, edges_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sample = manifest["sample"]
    image = BiohubImage(Path(data_dir) / f"{sample}.zarr")
    gt, _ = read_geff_tracks(Path(data_dir) / f"{sample}.geff",
                             image.voxel_size, image.shape)
    start = np.asarray(manifest["roi_start"], dtype=int)
    end = start + np.asarray(manifest["roi_shape"], dtype=int)
    hidden = set(manifest["hidden_node_ids"])
    roi_gt = []
    for record in gt:
        voxel = (np.array([record[k] for k in ("t", "z", "y", "x")]) /
                 np.asarray(image.voxel_size))
        if np.all(voxel >= start) and np.all(voxel < end):
            roi_gt.append(record)
    frame_counts = {t: sum(int(r["cell_id"] in hidden) for r in roi_gt
                           if int(r["t"]) == t)
                    for t in range(start[0], end[0])}
    t = max(frame_counts, key=frame_counts.get)
    predicted = read_rows(nodes_path)
    pred_edges = read_rows(edges_path)
    matched = match_nodes(predicted, roi_gt, 4.0)
    recovered = {pred_id for pred_id, gt_id in matched.items() if gt_id in hidden}
    volume = image.read((slice(t, t + 1),
                         slice(start[1], end[1]),
                         slice(start[2], end[2]),
                         slice(start[3], end[3])))[0]
    mip = volume.max(axis=0)
    lo, hi = np.percentile(mip, (1, 99.7))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax in axes:
        ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        ax.set_xlim(0, end[3] - start[3])
        ax.set_ylim(0, end[2] - start[2])
        ax.set_xlabel("x (voxel in ROI)")
        ax.set_ylabel("y (voxel in ROI)")
    current_gt = [r for r in roi_gt if int(r["t"]) == t]
    gt_by_id = {int(r["cell_id"]): r for r in roi_gt}
    for child in current_gt:
        parent = gt_by_id.get(int(child["parent_id"]))
        if parent is None:
            continue
        color = "magenta" if int(child["cell_id"]) in hidden or \
            int(parent["cell_id"]) in hidden else "deepskyblue"
        axes[0].plot([parent["x"] / image.voxel_size[3] - start[3],
                      child["x"] / image.voxel_size[3] - start[3]],
                     [parent["y"] / image.voxel_size[2] - start[2],
                      child["y"] / image.voxel_size[2] - start[2]],
                     color=color, alpha=0.6, linewidth=1)
    for flag, color, name in ((False, "deepskyblue", "Visible GT"),
                              (True, "magenta", "Hidden GT")):
        selected = [r for r in current_gt if (int(r["cell_id"]) in hidden) == flag]
        if selected:
            axes[0].scatter([r["x"] / image.voxel_size[3] - start[3] for r in selected],
                            [r["y"] / image.voxel_size[2] - start[2] for r in selected],
                            s=75, facecolors="none", edgecolors=color,
                            linewidths=1.8, label=f"{name} ({len(selected)})")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right")
    axes[0].set_title(f"GT at t={t}: visible and held-out cells")
    current_pred = [r for r in predicted if int(r["t"]) == t]
    by_id = {int(r["node_id"]): r for r in predicted}
    for edge in pred_edges:
        parent = by_id.get(int(edge["source_id"]))
        child = by_id.get(int(edge["target_id"]))
        if parent is None or child is None or int(child["t"]) != t:
            continue
        axes[1].plot([float(parent["x"]) - start[3],
                      float(child["x"]) - start[3]],
                     [float(parent["y"]) - start[2],
                      float(child["y"]) - start[2]],
                     color="lime", alpha=0.5, linewidth=1)
    if current_pred:
        axes[1].scatter([float(r["x"]) - start[3] for r in current_pred],
                        [float(r["y"]) - start[2] for r in current_pred],
                        s=48, facecolors="none", edgecolors="lime",
                        linewidths=1.3, label=f"Predicted ({len(current_pred)})")
        hits = [r for r in current_pred if int(r["node_id"]) in recovered]
        if hits:
            axes[1].scatter([float(r["x"]) - start[3] for r in hits],
                            [float(r["y"]) - start[2] for r in hits],
                            s=110, facecolors="none", edgecolors="yellow",
                            linewidths=2.0, label=f"Matches hidden ({len(hits)})")
        axes[1].legend(loc="upper right")
    else:
        candidate_file = Path(nodes_path).with_name("candidate_nodes.csv")
        if candidate_file.is_file():
            candidate = [r for r in read_rows(candidate_file)
                         if int(r["t"]) == t]
            if candidate:
                axes[1].scatter(
                    [float(r["x"]) - start[3] for r in candidate],
                    [float(r["y"]) - start[2] for r in candidate],
                    s=25, facecolors="none", edgecolors="gray",
                    label=f"Unselected candidates ({len(candidate)})")
                axes[1].legend(loc="upper right")
    axes[1].set_title(f"Linajea selected graph ({len(current_pred)} cells at t={t})")
    fig.suptitle(f"{sample}, t={t}, z max projection; same image and coordinates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        figure = make_subplots(rows=1, cols=2,
                               specs=[[{"type": "scene"}, {"type": "scene"}]],
                               subplot_titles=("GT: visible / hidden", "Linajea predictions"))
        for group, color, name in (([r for r in current_gt if int(r["cell_id"]) not in hidden],
                                     "deepskyblue", "Visible GT"),
                                    ([r for r in current_gt if int(r["cell_id"]) in hidden],
                                     "magenta", "Hidden GT")):
            figure.add_trace(go.Scatter3d(x=[r["x"] / image.voxel_size[3] for r in group],
                                          y=[r["y"] / image.voxel_size[2] for r in group],
                                          z=[r["z"] / image.voxel_size[1] for r in group],
                                          mode="markers", name=name,
                                          marker={"size": 4, "color": color}),
                             row=1, col=1)
        figure.add_trace(go.Scatter3d(x=[float(r["x"]) for r in current_pred],
                                      y=[float(r["y"]) for r in current_pred],
                                      z=[float(r["z"]) for r in current_pred],
                                      mode="markers", name="Predicted",
                                      marker={"size": 4, "color": "lime"}),
                         row=1, col=2)
        for scene in ("scene", "scene2"):
            figure.update_layout(**{scene: {"xaxis": {"range": [start[3], end[3]]},
                                            "yaxis": {"range": [start[2], end[2]]},
                                            "zaxis": {"range": [start[1], end[1]]},
                                            "aspectmode": "data"}})
        figure.write_html(output_dir / "comparison_3d.html", include_plotlyjs=True)
    except ImportError:
        pass
    return path

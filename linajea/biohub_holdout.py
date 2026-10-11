"""Embryo-disjoint Biohub train/validation split and sparse-GT diagnostics."""

import json
from collections import defaultdict
from pathlib import Path

from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import match_nodes


def choose_samples(data_dir, train_embryo, val_embryo,
                   train_count=8, val_count=2, seed=42):
    """Choose paired samples reproducibly from two different embryos."""
    import numpy as np

    if train_embryo == val_embryo:
        raise ValueError("Training and validation embryos must differ")
    data_dir = Path(data_dir)
    paired = sorted(path.stem for path in data_dir.glob("*.zarr")
                    if path.is_dir() and
                    (data_dir / f"{path.stem}.geff").is_dir())
    groups = defaultdict(list)
    for name in paired:
        groups[name.split("_", 1)[0]].append(name)
    for embryo, count in ((train_embryo, train_count), (val_embryo, val_count)):
        if len(groups[embryo]) < count:
            raise ValueError(f"Embryo {embryo} has {len(groups[embryo])} "
                             f"paired samples; need {count}")
    rng = np.random.default_rng(seed)
    train = sorted(rng.choice(groups[train_embryo], train_count,
                                   replace=False).tolist())
    val = sorted(rng.choice(groups[val_embryo], val_count,
                                 replace=False).tolist())
    assert not set(train) & set(val)
    return {"data_dir": str(data_dir.resolve()), "train_embryo": train_embryo,
            "val_embryo": val_embryo, "seed": seed,
            "train": train, "validation": val,
            "available_by_embryo": {key: len(value)
                                    for key, value in sorted(groups.items())}}


def validate_split(data_dir, split):
    """Check no sample/embryo leakage and validate ten image/GEFF pairs."""
    data_dir = Path(data_dir)
    train, val = split["train"], split["validation"]
    if len(train) != 8 or len(val) != 2 or len(set(train + val)) != 10:
        raise ValueError("Expected eight train and two distinct validation samples")
    if {s.split("_", 1)[0] for s in train} & \
            {s.split("_", 1)[0] for s in val}:
        raise ValueError("Embryo leakage between train and validation")
    details = {}
    scales = set()
    for name in train + val:
        image = BiohubImage(data_dir / f"{name}.zarr")
        records, edge_count = read_geff_tracks(
            data_dir / f"{name}.geff", image.voxel_size, image.shape)
        if not records or not edge_count:
            raise ValueError(f"Sample {name} has no annotated nodes or edges")
        if not {"q001", "q099"} <= set(image.attrs["stats"]):
            raise ValueError(f"Sample {name} lacks image quantile metadata")
        scales.add(image.voxel_size)
        details[name] = {"shape": image.shape, "voxel_size": image.voxel_size,
                         "annotated_nodes": len(records),
                         "annotated_edges": edge_count}
    if len(scales) != 1:
        raise ValueError(f"Linajea requires one voxel size; found {scales}")
    return details


def sparse_diagnostics(nodes, edges, gt_records, radius_um=7.0):
    """Compare to known sparse annotations without calling unknowns false positives."""
    matches = match_nodes(nodes, gt_records, radius_um)
    gt_by_id = {int(row["cell_id"]): row for row in gt_records}
    gt_edges = {(int(row["parent_id"]), int(row["cell_id"]))
                for row in gt_records
                if int(row["parent_id"]) in gt_by_id}
    selected_edges = {(int(row["source_id"]), int(row["target_id"]))
                      for row in edges}
    inverse = {gt_id: pred_id for pred_id, gt_id in matches.items()}
    recovered_edges = sum((inverse.get(a), inverse.get(b)) in selected_edges
                          for a, b in gt_edges)
    endpoint_detectable = sum(a in inverse and b in inverse
                              for a, b in gt_edges)
    return {
        "match_radius_um": radius_um,
        "predicted_nodes": len(nodes), "predicted_edges": len(edges),
        "known_gt_nodes": len(gt_records), "known_gt_edges": len(gt_edges),
        "matched_gt_nodes": len(set(matches.values())),
        "known_node_recall": len(set(matches.values())) / len(gt_records)
        if gt_records else None,
        "matched_gt_edges": recovered_edges,
        "known_edge_recall": recovered_edges / len(gt_edges)
        if gt_edges else None,
        "known_edges_with_both_endpoints": endpoint_detectable,
        "conditional_edge_recall": recovered_edges / endpoint_detectable
        if endpoint_detectable else None,
        "predictions_unmatched_by_sparse_gt": len(nodes) - len(matches),
        "precision_note": "Unmatched predictions may be real because GT is sparse.",
    }


def save_split(path, split, details):
    Path(path).write_text(json.dumps({**split, "sample_details": details},
                                     indent=2), encoding="utf-8")

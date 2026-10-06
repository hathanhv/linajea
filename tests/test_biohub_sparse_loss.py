"""Sparse annotations must not label distant voxels as background."""

from types import SimpleNamespace

import torch

from linajea.training.torch_loss import LossWrapper


def _loss(sparse, prediction, cell_mask=None):
    config = SimpleNamespace(
        general=SimpleNamespace(sparse=sparse),
        model=SimpleNamespace(
            train_only_cell_indicator=cell_mask is None,
            cell_indicator_cutoff=0.5,
            cell_indicator_weighted=None,
        ),
        train_data=SimpleNamespace(voxel_size=(1, 52, 13, 13)),
        train=SimpleNamespace(movement_vectors_loss_transition_offset=None),
    )
    loss = LossWrapper(config)
    loss.metric_summaries = lambda *args: None
    gt = torch.zeros((1, 1, 3, 3))
    gt[0, 0, 1, 1] = 1
    movement = torch.zeros((3,) + tuple(gt.shape))
    return loss(
        gt_cell_indicator=gt,
        cell_indicator=prediction,
        maxima=torch.zeros_like(gt),
        gt_cell_center=gt,
        cell_mask=cell_mask,
        gt_movement_vectors=movement,
        movement_vectors=movement,
    )[0].item()


def test_sparse_cell_loss_ignores_unannotated_voxel():
    prediction = torch.zeros((1, 1, 3, 3))
    prediction[0, 0, 0, 0] = 1
    assert _loss(True, prediction) == _loss(True, torch.zeros_like(prediction))
    assert _loss(False, prediction) > _loss(False, torch.zeros_like(prediction))


def test_sparse_cell_loss_uses_local_background_mask():
    cell_mask = torch.zeros((1, 1, 3, 3))
    cell_mask[0, 0, 1, 0] = 1
    prediction = torch.zeros_like(cell_mask)
    prediction[0, 0, 1, 0] = 1
    assert _loss(True, prediction, cell_mask) > _loss(
        True, torch.zeros_like(prediction), cell_mask)

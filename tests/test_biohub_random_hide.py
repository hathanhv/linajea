"""Toy graph checks for held-out matching and the Linajea basic ILP."""

from linajea.biohub_random_hide import match_nodes
from linajea.biohub_reconstruct import candidate_edges, solve_basic_ilp


def test_physical_matching_is_one_to_one():
    gt = [
        {"cell_id": 10, "t": 0, "z": 0, "y": 0, "x": 0},
        {"cell_id": 11, "t": 0, "z": 0, "y": 0, "x": 13},
    ]
    predicted = [
        {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0},
        {"node_id": 2, "t": 0, "z": 0, "y": 0, "x": 0},
    ]
    matches = match_nodes(predicted, gt, 0.2)
    assert len(matches) == 1
    assert set(matches.values()) == {10}


def test_matching_maximizes_recovered_nodes_before_distance():
    gt = [
        {"cell_id": 10, "t": 0, "z": 0, "y": 0, "x": 0},
        {"cell_id": 11, "t": 0, "z": 0, "y": 0, "x": 64},
    ]
    predicted = [
        {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0},
        {"node_id": 2, "t": 0, "z": 0, "y": 0, "x": -4},
    ]
    assert match_nodes(predicted, gt, 2.0) == {1: 11, 2: 10}


def test_candidate_edges_and_basic_ilp_select_chain():
    nodes = [{"node_id": t, "t": t, "z": 10, "y": 20, "x": 30,
              "score": 0.95, "mv_z_um": 0, "mv_y_um": 0,
              "mv_x_um": 0} for t in range(5)]
    edges = candidate_edges(nodes)
    assert len(edges) == 4
    assert [(e["source_id"], e["target_id"]) for e in edges] == [
        (0, 1), (1, 2), (2, 3), (3, 4)]
    selected_nodes, selected_edges, status = solve_basic_ilp(nodes, edges)
    assert len(selected_nodes) == 5
    assert len(selected_edges) == 4
    assert status["selected_edges"] == 4

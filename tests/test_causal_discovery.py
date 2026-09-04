"""Unit tests for causal discovery module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_factor_research.causal_discovery import (
    distance_correlation_matrix,
    parse_causal_learn_graph,
)


def test_distance_correlation_matrix():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 2 * x + rng.normal(0, 0.1, 100)
    z = rng.normal(0, 1, 100)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    dcor = distance_correlation_matrix(df, ["x", "y", "z"])
    assert dcor.shape == (3, 3)
    assert dcor.loc["x", "y"] > 0.8
    assert dcor.loc["x", "z"] < 0.3


def test_parse_causal_learn_graph_dummy():
    """Test graph parsing with a minimal mock graph object."""

    class MockNode:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    class MockEdge:
        def __init__(self, n1, n2, ep1, ep2):
            self.node1 = MockNode(n1)
            self.node2 = MockNode(n2)
            self.endpoint1 = ep1
            self.endpoint2 = ep2

        def get_node1(self):
            return self.node1

        def get_node2(self):
            return self.node2

        def get_endpoint1(self):
            return self.endpoint1

        def get_endpoint2(self):
            return self.endpoint2

    class MockGraph:
        def __init__(self, edges):
            self._edges = edges

        def get_graph_edges(self):
            return self._edges

    edges = [MockEdge("x", "y", -1, 1)]  # x -> y (tail at x, arrow at y)
    G = MockGraph(edges)
    adj, endpoints, edge_list, v_structures = parse_causal_learn_graph(G, ["x", "y"])
    assert adj.loc["x", "y"] == 1
    assert "x -> y" in edge_list


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

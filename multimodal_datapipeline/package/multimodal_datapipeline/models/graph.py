"""Minimal pure-PyTorch graph neural network building blocks.

Used by the molecular graph encoder and the AlphaFold residue-graph encoder so
the project does not depend on PyTorch Geometric or DGL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install PyTorch before using graph models: pip install torch")


@dataclass
class Graph:
    """One graph. Undirected edges are stored in both directions."""

    x: "torch.Tensor"  # [num_nodes, node_dim]
    edge_index: "torch.Tensor"  # [2, num_edges], rows are (source, destination)
    edge_attr: "torch.Tensor | None" = None  # [num_edges, edge_dim]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def to(self, device: "torch.device | str") -> "Graph":
        return Graph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=None if self.edge_attr is None else self.edge_attr.to(device),
        )


@dataclass
class GraphBatch:
    """Several graphs merged into one disconnected graph."""

    x: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor | None"
    batch: "torch.Tensor"  # [num_nodes], graph id of every node
    num_graphs: int

    def to(self, device: "torch.device | str") -> "GraphBatch":
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=None if self.edge_attr is None else self.edge_attr.to(device),
            batch=self.batch.to(device),
            num_graphs=self.num_graphs,
        )


def batch_graphs(graphs: Sequence[Graph]) -> GraphBatch:
    _require_torch()
    if not graphs:
        raise ValueError("Cannot batch an empty list of graphs.")
    has_edge_attr = graphs[0].edge_attr is not None

    xs, edge_indices, edge_attrs, batch = [], [], [], []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        if (graph.edge_attr is not None) != has_edge_attr:
            raise ValueError("Either all graphs or none must have edge_attr.")
        xs.append(graph.x)
        edge_indices.append(graph.edge_index + offset)
        if has_edge_attr:
            edge_attrs.append(graph.edge_attr)
        batch.append(torch.full((graph.num_nodes,), graph_id, dtype=torch.long, device=graph.x.device))
        offset += graph.num_nodes

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0) if has_edge_attr else None,
        batch=torch.cat(batch, dim=0),
        num_graphs=len(graphs),
    )


def global_mean_pool(x: "torch.Tensor", batch: "torch.Tensor", num_graphs: int) -> "torch.Tensor":
    summed = torch.zeros(num_graphs, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, batch, x)
    counts = torch.zeros(num_graphs, dtype=x.dtype, device=x.device).index_add_(
        0, batch, torch.ones_like(batch, dtype=x.dtype)
    )
    return summed / counts.clamp_min(1.0).unsqueeze(-1)


class GraphConvLayer(nn.Module if nn is not None else object):
    """Mean-aggregation message passing with optional edge features and a residual.

    h_i' = h_i + Dropout(ReLU(LayerNorm(W_s h_i + mean_j ReLU(W_n h_j + W_e e_ij))))
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.0) -> None:
        _require_torch()
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.edge_linear = nn.Linear(edge_dim, hidden_dim) if edge_dim else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        source, destination = edge_index
        messages = self.neighbor_linear(x)[source]
        if self.edge_linear is not None:
            if edge_attr is None:
                raise ValueError("This layer was built with edge features but edge_attr is None.")
            messages = messages + self.edge_linear(edge_attr)
        messages = F.relu(messages)

        aggregated = torch.zeros_like(x).index_add_(0, destination, messages)
        degree = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device).index_add_(
            0, destination, torch.ones_like(destination, dtype=x.dtype)
        )
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)

        updated = F.relu(self.norm(self.self_linear(x) + aggregated))
        return x + self.dropout(updated)


class GraphEncoder(nn.Module if nn is not None else object):
    """Node projection -> stacked GraphConvLayers -> mean pool -> graph embedding."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        self.input = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConvLayer(hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.output = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.ReLU())

    def node_embeddings(self, batch: GraphBatch) -> "torch.Tensor":
        hidden = F.relu(self.input(batch.x.float()))
        edge_attr = None if batch.edge_attr is None else batch.edge_attr.float()
        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index, edge_attr)
        return hidden

    def forward(self, batch: GraphBatch) -> "torch.Tensor":
        hidden = self.node_embeddings(batch)
        return self.output(global_mean_pool(hidden, batch.batch, batch.num_graphs))


def to_dense_batch(
    x: "torch.Tensor", batch: "torch.Tensor", num_graphs: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Node rows [N, d] -> padded [num_graphs, max_nodes, d] plus a bool mask of real nodes."""
    counts = torch.bincount(batch, minlength=num_graphs)
    max_nodes = int(counts.max().item()) if num_graphs else 0
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(x.shape[0], device=x.device) - starts[batch]
    dense = x.new_zeros(num_graphs, max_nodes, x.shape[1])
    mask = torch.zeros(num_graphs, max_nodes, dtype=torch.bool, device=x.device)
    dense[batch, positions] = x
    mask[batch, positions] = True
    return dense, mask

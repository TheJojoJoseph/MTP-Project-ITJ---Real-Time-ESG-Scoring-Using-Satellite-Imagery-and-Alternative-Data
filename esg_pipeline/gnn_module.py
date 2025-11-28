from typing import Dict, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE, GATConv, global_mean_pool
import pandas as pd
import numpy as np


def build_company_graph(
    companies: pd.DataFrame,
    vit_embeddings: Dict[str, Tensor],
) -> Tuple[Data, Dict[str, int]]:
    """Construct a simple company graph.

    Nodes: companies.
    Edges: same-sector or same-supply-chain-region connections.
    Node features: concatenation of ViT embedding + tabular features.
    """
    company_ids = companies["company_id"].tolist()
    id_to_idx = {cid: i for i, cid in enumerate(company_ids)}

    # Build edges based on shared sector / region
    edge_index = []
    for i, row_i in companies.iterrows():
        for j, row_j in companies.iterrows():
            if i == j:
                continue
            if row_i["sector"] == row_j["sector"] or row_i["supply_chain_regions"] == row_j["supply_chain_regions"]:
                edge_index.append((i, j))

    if not edge_index:
        edge_index = [(i, i) for i in range(len(company_ids))]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Tabular features (normalized roughly)
    tabular_cols = [
        "emissions_intensity",
        "board_independence",
        "labor_controversies",
        "env_sentiment",
        "soc_sentiment",
        "gov_sentiment",
        "num_subsidiaries",
        "incorporation_age",
    ]

    tab = companies[tabular_cols].to_numpy(dtype=np.float32)
    tab = (tab - tab.mean(axis=0, keepdims=True)) / \
        (tab.std(axis=0, keepdims=True) + 1e-6)
    tab = torch.from_numpy(tab)

    # Stack ViT embeddings in same order
    vit_dim = next(iter(vit_embeddings.values())).shape[0]
    vit_mat = torch.zeros((len(company_ids), vit_dim), dtype=torch.float32)
    for cid, idx in id_to_idx.items():
        vit_mat[idx] = vit_embeddings[cid]

    x = torch.cat([vit_mat, tab], dim=1)

    data = Data(x=x, edge_index=edge_index)
    data.company_ids = company_ids
    return data, id_to_idx


class GraphSAGEESG(nn.Module):
    """GraphSAGE-based ESG embedding model."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.sage = GraphSAGE(
            in_channels=in_dim, hidden_channels=hidden_dim, num_layers=2, out_channels=out_dim)
        self.act = nn.ReLU()

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        h = self.sage(x, edge_index)
        return self.act(h)


class GATESG(nn.Module):
    """GAT-based ESG embedding model (node-level)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, heads: int = 2) -> None:
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim,
                             heads=1, concat=False)
        self.act = nn.ELU()

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

"""Visualization helpers for the ESG pipeline."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image
import torch


def plot_satellite_images(image_map: Dict[str, Image.Image], max_images: int = 4) -> None:
    """Plot a small grid of company satellite images."""
    items = list(image_map.items())[:max_images]
    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (cid, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(cid)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    fig.tight_layout()


def plot_attention_heatmap(image: Image.Image, attention_map: torch.Tensor) -> None:
    """Overlay a coarse attention heatmap on top of an image."""
    import numpy as np

    img = image.resize((224, 224))
    heat = attention_map.numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat = Image.fromarray((255 * heat).astype("uint8"))
    heat = heat.resize(img.size)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img)
    ax.imshow(heat, cmap="jet", alpha=0.4)
    ax.axis("off")
    ax.set_title("Attention heatmap (mock)")


def plot_company_graph(companies: pd.DataFrame, id_to_idx: Dict[str, int]) -> None:
    """Visualize the company graph using NetworkX."""
    G = nx.Graph()
    for cid in id_to_idx.keys():
        G.add_node(cid, sector=str(companies.set_index(
            "company_id").loc[cid, "sector"]))

    # Add edges for shared sector/region
    for i, row_i in companies.iterrows():
        for j, row_j in companies.iterrows():
            if i >= j:
                continue
            if row_i["sector"] == row_j["sector"] or row_i["supply_chain_regions"] == row_j["supply_chain_regions"]:
                G.add_edge(row_i["company_id"], row_j["company_id"])

    pos = nx.spring_layout(G, seed=42)
    sectors = list({d["sector"] for _, d in G.nodes(data=True)})
    color_map = {sec: i for i, sec in enumerate(sectors)}
    node_colors = [color_map[G.nodes[n]["sector"]] for n in G.nodes]

    plt.figure(figsize=(6, 5))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            cmap="tab10", node_size=600)
    plt.title("Company Graph (color = sector)")


def plot_esg_dashboard(scores: pd.DataFrame) -> None:
    """Static ESG dashboard-style visualization."""
    scores = scores.sort_values("ESG_score", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall ESG
    axes[0].bar(scores["company_id"], scores["ESG_score"], color="C0")
    axes[0].set_title("Overall ESG scores")
    axes[0].set_ylabel("Score (0-100)")
    axes[0].set_xlabel("Company")
    axes[0].set_ylim(0, 100)

    # Stacked E, S, G
    axes[1].bar(scores["company_id"], scores["E"], label="E", color="C2")
    axes[1].bar(scores["company_id"], scores["S"],
                bottom=scores["E"], label="S", color="C1")
    bottom = scores["E"] + scores["S"]
    axes[1].bar(scores["company_id"], scores["G"],
                bottom=bottom, label="G", color="C3")
    axes[1].set_title("E/S/G breakdown")
    axes[1].set_ylabel("Score components")
    axes[1].set_xlabel("Company")
    axes[1].legend()

    fig.suptitle("ESG Dashboard (static mock-up)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

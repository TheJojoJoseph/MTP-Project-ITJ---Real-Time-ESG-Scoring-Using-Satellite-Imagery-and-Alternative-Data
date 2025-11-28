"""ESG scoring engine combining ViT + GNN embeddings and tabular signals."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
import pandas as pd


def compute_esg_subscores(
    companies: pd.DataFrame,
    gnn_embeddings: Dict[str, Tensor],
) -> pd.DataFrame:
    """Compute interpretable E, S, G subscores from embeddings + features.

    This uses a transparent linear-style formula rather than a black-box
    deep classifier, making it suitable for research and auditability.
    """
    records = []
    for _, row in companies.iterrows():
        cid = row["company_id"]
        emb = gnn_embeddings[cid]

        # Simple interpretable aggregations (placeholders for more
        # rigorous models):
        e_score = (
            -row["emissions_intensity"]
            + 0.5 * row["env_sentiment"]
            - 0.1 * row["labor_controversies"]
            + 0.2 * float(emb.mean())
        )
        s_score = (
            -0.2 * row["labor_controversies"]
            + 0.5 * row["soc_sentiment"]
            + 0.2 * float(emb.std())
        )
        g_score = (
            0.8 * row["board_independence"]
            + 0.5 * row["gov_sentiment"]
            - 0.01 * row["num_subsidiaries"]
            + 0.1 * float(emb.norm())
        )

        records.append({
            "company_id": cid,
            "E_raw": e_score,
            "S_raw": s_score,
            "G_raw": g_score,
        })

    df = pd.DataFrame(records)

    # Normalize to 0-100 range per dimension
    for col in ["E_raw", "S_raw", "G_raw"]:
        vals = df[col].to_numpy(dtype=float)
        min_v, max_v = vals.min(), vals.max()
        if max_v - min_v < 1e-6:
            df[col.replace("_raw", "")] = 50.0
        else:
            df[col.replace("_raw", "")] = 100.0 * \
                (vals - min_v) / (max_v - min_v)

    # Overall ESG: weighted average
    df["ESG_score"] = 0.4 * df["E"] + 0.3 * df["S"] + 0.3 * df["G"]
    return df


def update_scores_with_stream_event(
    current_scores: pd.DataFrame,
    event: Dict,
) -> pd.DataFrame:
    """Apply a simple streaming update rule to ESG scores.

    This mimics a Flink/Spark-style transformation where each incoming
    event updates a stateful score table.
    """
    cid = event["company_id"]
    etype = event["event_type"]
    payload = event["payload"]

    df = current_scores.copy()
    mask = df["company_id"] == cid
    if not mask.any():
        return df

    if etype == "satellite_update" and "ndvi_delta" in payload:
        df.loc[mask, "E"] += 5.0 * payload["ndvi_delta"]
    elif etype == "news_update" and "env_news_delta" in payload:
        df.loc[mask, "E"] += 2.0 * payload["env_news_delta"]
        df.loc[mask, "S"] += 2.0 * payload["env_news_delta"]
    elif etype == "disclosure" and "governance_flag" in payload:
        df.loc[mask, "G"] += 10.0 * (1 if payload["governance_flag"] else -1)

    # Re-clamp scores to 0-100
    for col in ["E", "S", "G"]:
        df[col] = df[col].clip(0, 100)
    df["ESG_score"] = 0.4 * df["E"] + 0.3 * df["S"] + 0.3 * df["G"]
    return df

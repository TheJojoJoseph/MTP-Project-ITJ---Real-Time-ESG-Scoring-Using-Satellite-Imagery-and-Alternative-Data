import sys
from pathlib import Path

import streamlit as st
import torch
import pandas as pd

from esg_pipeline.data_ingestion import build_merged_entity_frame
from esg_pipeline.vit_module import ViTFeatureExtractor, encode_company_images, get_dummy_attention_map
from esg_pipeline.gnn_module import build_company_graph, GraphSAGEESG, GATESG
from esg_pipeline.streaming import InMemoryTopic, ESGConsumer, start_background_producer
from esg_pipeline.scoring import compute_esg_subscores, update_scores_with_stream_event
from esg_pipeline.visualization import (
    plot_satellite_images,
    plot_attention_heatmap,
    plot_company_graph,
    plot_esg_dashboard,
)


@st.cache_resource(show_spinner=False)
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=True)
def get_vit_extractor(model_name: str, device: str) -> ViTFeatureExtractor:
    return ViTFeatureExtractor(model_name=model_name, device=device)


st.set_page_config(
    page_title="Real-Time ESG Scoring Demo",
    layout="wide",
)

st.title("Real-Time ESG Scoring Using Satellite Imagery & Alternative Data")

with st.sidebar:
    st.header("Pipeline Controls")
    n_companies = st.slider("Number of companies",
                            min_value=3, max_value=20, value=6, step=1)
    vit_model_name = st.text_input(
        "ViT model",
        value="google/vit-base-patch16-224",
        help="Hugging Face ViT model identifier",
    )
    gnn_type = st.selectbox("GNN type", ["GraphSAGE", "GAT"], index=0)
    hidden_dim = st.slider("GNN hidden dim", 32, 256, 128, step=32)
    out_dim = st.slider("Embedding dim", 16, 128, 64, step=16)

    st.markdown("---")
    st.subheader("Streaming simulation")
    event_rate_hz = st.slider("Event rate (Hz)", 0.5, 5.0, 3.0, step=0.5)
    duration = st.slider("Simulation duration (s)", 1.0, 15.0, 5.0, step=1.0)
    max_events = st.slider("Max events to consume", 10, 200, 50, step=10)

    run_button = st.button("Run full ESG pipeline", type="primary")


device = get_device()
st.caption(f"Using device: {device}")

if not run_button:
    st.info("Configure parameters in the sidebar and click **Run full ESG pipeline**.")
    st.stop()


with st.spinner("1/6 – Ingesting synthetic multi-source ESG data"):
    companies, image_map = build_merged_entity_frame(n_companies=n_companies)

st.subheader("1. Company universe & tabular ESG features")
st.dataframe(companies, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Sample satellite images**")
    import matplotlib.pyplot as plt

    fig1 = plt.figure(figsize=(6, 4))
    plot_satellite_images(image_map, max_images=4)
    st.pyplot(fig1, clear_figure=True)

with col2:
    st.markdown("**Mock attention heatmap on first image**")
    if image_map:
        first_cid = list(image_map.keys())[0]
        attn_map = get_dummy_attention_map(image_map[first_cid])
        fig2 = plt.figure(figsize=(4, 4))
        plot_attention_heatmap(image_map[first_cid], attn_map)
        st.pyplot(fig2, clear_figure=True)


with st.spinner("2/6 – Extracting ViT embeddings for satellite images"):
    extractor = get_vit_extractor(vit_model_name, device=device)
    company_vit_embs = encode_company_images(image_map, extractor)

st.subheader("2. ViT embeddings summary")
example_emb = next(iter(company_vit_embs.values())
                   ) if company_vit_embs else None
if example_emb is not None:
    st.write("Embedding shape per company:", tuple(example_emb.shape))


with st.spinner("3/6 – Building company graph & running GNN"):
    graph_data, id_to_idx = build_company_graph(companies, company_vit_embs)
    in_dim = graph_data.x.shape[1]

    if gnn_type == "GAT":
        gnn_model = GATESG(in_dim, hidden_dim=hidden_dim,
                           out_dim=out_dim).to(device)
    else:
        gnn_model = GraphSAGEESG(
            in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)

    gnn_model.eval()
    with torch.inference_mode():
        graph_data = graph_data.to(device)
        node_embs = gnn_model(graph_data).cpu()

st.subheader("3. Company graph & node embeddings")
colg1, colg2 = st.columns([1, 1])
with colg1:
    st.write("Node feature matrix shape:", tuple(graph_data.x.shape))
    st.write("Node embedding matrix shape:", tuple(node_embs.shape))

with colg2:
    fig_g = plt.figure(figsize=(5, 4))
    plot_company_graph(companies, id_to_idx)
    st.pyplot(fig_g, clear_figure=True)

idx_to_id = {idx: cid for cid, idx in id_to_idx.items()}
gnn_embeddings = {idx_to_id[i]: node_embs[i]
                  for i in range(node_embs.shape[0])}


with st.spinner("4/6 – Computing static ESG scores from embeddings + tabular features"):
    scores = compute_esg_subscores(companies, gnn_embeddings)

st.subheader("4. Static ESG scores (before streaming)")
st.dataframe(scores, use_container_width=True)

fig_esg_static = plt.figure(figsize=(10, 4))
plot_esg_dashboard(scores)
st.pyplot(fig_esg_static, clear_figure=True)


with st.spinner("5/6 – Simulating ESG event stream and updating scores"):
    topic = InMemoryTopic()
    consumer = ESGConsumer(topic)
    company_ids = companies["company_id"].tolist()

    producer_thread = start_background_producer(
        topic,
        company_ids,
        event_rate_hz=float(event_rate_hz),
        stop_after=float(duration),
    )
    producer_thread.join()

    updated_scores = scores.copy()
    event_log = []
    for ev in consumer.iterate(max_events=int(max_events), timeout=0.5):
        d = {
            "company_id": ev.company_id,
            "event_type": ev.event_type,
            "payload": ev.payload,
            "timestamp": ev.timestamp,
        }
        event_log.append(d)
        updated_scores = update_scores_with_stream_event(updated_scores, d)

st.subheader("5. Streaming events applied to ESG scores")
if event_log:
    events_df = pd.DataFrame(event_log)
    st.markdown("**Event log**")
    st.dataframe(events_df, use_container_width=True)
else:
    st.warning(
        "No events were produced by the simulator. Try increasing duration or max_events.")

comparison = scores[["company_id", "ESG_score"]].merge(
    updated_scores[["company_id", "ESG_score"]],
    on="company_id",
    suffixes=("_orig", "_updated"),
)

colc1, colc2 = st.columns(2)
with colc1:
    st.markdown("**Original vs updated ESG scores**")
    st.dataframe(comparison, use_container_width=True)

with colc2:
    fig_esg_updated = plt.figure(figsize=(10, 4))
    plot_esg_dashboard(updated_scores)
    st.pyplot(fig_esg_updated, clear_figure=True)


st.subheader("6. Final ESG scores (post-streaming)")
final_scores = updated_scores.sort_values(
    "ESG_score", ascending=False).reset_index(drop=True)
st.dataframe(final_scores, use_container_width=True)

st.success("Pipeline run completed.")

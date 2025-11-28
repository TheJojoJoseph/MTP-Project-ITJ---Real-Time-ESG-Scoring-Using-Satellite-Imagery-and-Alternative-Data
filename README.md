# Real-Time ESG Scoring Using Satellite Imagery & Alternative Data

This repository contains an M.Tech-level prototype for a **real-time ESG (Environmental, Social, Governance) scoring pipeline** built in Python using a Jupyter notebook plus small reusable helper modules.

The pipeline is fully runnable on **synthetic data** and is designed so that real data sources (Sentinel/Landsat satellite imagery, OpenCorporates, news feeds, Kafka, etc.) can be plugged in later without changing the main notebook structure.

## Approach

The project follows a modular architecture with five main layers:

1. **Data Ingestion** (`esg_pipeline/data_ingestion.py`)
   - Synthetic company universe with ESG-relevant attributes (emissions intensity, sector, board independence, controversies, etc.).
   - Synthetic news-based E/S/G sentiment scores.
   - OpenCorporates-style registry stub (subsidiaries, incorporation age).
   - Synthetic satellite image tiles as placeholders for Sentinel/Landsat/Google Earth Engine.

2. **Vision Transformer (ViT) for Satellite Features** (`esg_pipeline/vit_module.py`)
   - Wraps a pre-trained ViT model (e.g., `google/vit-base-patch16-224`) from Hugging Face.
   - Encodes each company’s satellite tile into a CLS embedding vector.
   - Provides a simple attention heatmap mock for visualization.

3. **Graph Neural Network (GNN) for ESG Embeddings** (`esg_pipeline/gnn_module.py`)
   - Builds a company graph where nodes are companies and edges connect firms with the same sector or supply-chain region.
   - Node features = [ViT embedding || normalized tabular features].
   - Implements both **GraphSAGE** and **GAT** models using PyTorch Geometric to produce ESG embeddings per company.

4. **Real-Time Streaming Simulation** (`esg_pipeline/streaming.py`)
   - In-memory Kafka-like topic (`InMemoryTopic`) with a small producer/consumer API.
   - Background producer emits random ESG-relevant events (satellite updates, news deltas, governance disclosures).
   - Used to demonstrate **Flink/Spark-style stateful updates** to ESG scores in real time.

5. **ESG Scoring & Visualization** (`esg_pipeline/scoring.py`, `esg_pipeline/visualization.py`)
   - Transparent scoring engine that combines GNN embeddings and tabular features into interpretable E, S, G subscores and a 0–100 overall ESG score.
   - Visualization helpers for:
     - Satellite image grids and attention heatmaps.
     - Company graph structure.
     - Static ESG dashboard (overall ESG + E/S/G breakdown).

The main notebook, `notebooks/esg_scoring_pipeline.ipynb`, stitches all of these components into an end-to-end experiment that is suitable as a basis for M.Tech thesis sections (methodology, experiments, and discussion).

## Project Structure

```text
.
├── esg_pipeline/
│   ├── __init__.py
│   ├── data_ingestion.py      # Synthetic satellite + alternative data + stubs
│   ├── vit_module.py          # ViT feature extractor + attention heatmap mock
│   ├── gnn_module.py          # GraphSAGE / GAT ESG embedding models
│   ├── streaming.py           # In-memory Kafka-like streaming simulator
│   ├── scoring.py             # ESG scoring formulas + streaming updates
│   └── visualization.py       # Satellite, graph, and dashboard plots
├── notebooks/
│   └── esg_scoring_pipeline.ipynb  # Main end-to-end ESG pipeline notebook
├── requirements.txt
└── README.md
```

## Requirements

- Python **3.10+**
- Packages in `requirements.txt`, including:
  - `torch`, `torchvision`, `torch-geometric`, `transformers`
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
  - `Pillow`, `networkx`, `kafka-python`, `jupyter`

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

> Note: the first run of the notebook will download pre-trained ViT weights from Hugging Face, so an internet connection is required once.

## Running the ESG Pipeline Notebook

1. **Launch Jupyter Notebook** from the project root:

   ```bash
   jupyter notebook
   ```

2. Open:

   ```text
   notebooks/esg_scoring_pipeline.ipynb
   ```

3. Run all cells from top to bottom:
   - Step 1: Imports and environment.
   - Step 2: Synthetic data ingestion (companies + alternative data + satellite tiles).
   - Step 3: ViT feature extraction + attention heatmap visualization.
   - Step 4: Graph construction and GNN embeddings.
   - Step 5: Static ESG scoring and dashboard.
   - Step 6: Streaming simulation and online ESG score updates.
   - Step 7: Final ESG score table for the sample companies.

## Possible Research Extensions

- Replace synthetic satellite tiles with real Sentinel/Landsat imagery via Google Earth Engine or ESA APIs.
- Fine-tune the ViT model on ESG-labelled remote sensing datasets.
- Train the GNN with supervision (e.g., known ESG ratings or controversy labels).
- Swap the in-memory stream with a real **Apache Kafka** setup and integrate with **Flink** or **Spark Structured Streaming**.
- Add explainability (e.g., SHAP, Grad-CAM for ViT, and GNN explainers) and fairness/bias analysis across sectors/regions.

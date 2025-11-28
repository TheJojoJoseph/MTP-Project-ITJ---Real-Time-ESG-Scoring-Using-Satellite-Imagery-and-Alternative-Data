import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image


def generate_synthetic_satellite_image(size: int = 224) -> Image.Image:
    """Generate a synthetic RGB satellite-like image as a placeholder.

    This mimics a crop from Sentinel/Landsat; in practice you would
    integrate official APIs or Google Earth Engine.
    """
    arr = np.clip(np.random.normal(loc=120, scale=40, size=(
        size, size, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate_synthetic_satellite_batch(n_images: int = 4, size: int = 224) -> List[Image.Image]:
    """Return a list of synthetic satellite images."""
    return [generate_synthetic_satellite_image(size=size) for _ in range(n_images)]


def synthetic_company_metadata(n_companies: int = 5) -> pd.DataFrame:
    """Create synthetic company-level alternative data.

    Columns roughly mimic ESG-relevant indicators.
    """
    rng = np.random.default_rng(42)
    company_ids = [f"C{i:03d}" for i in range(n_companies)]

    df = pd.DataFrame(
        {
            "company_id": company_ids,
            "sector": rng.choice(["Energy", "Manufacturing", "Technology", "Retail"], size=n_companies),
            # tCO2e / revenue
            "emissions_intensity": rng.uniform(0.1, 3.0, size=n_companies),
            # fraction independent
            "board_independence": rng.uniform(0.2, 0.9, size=n_companies),
            "labor_controversies": rng.integers(0, 10, size=n_companies),
            "supply_chain_regions": rng.choice(["RegionA", "RegionB", "RegionC"], size=n_companies),
        }
    )
    return df


def synthetic_news_signals(company_ids: List[str]) -> pd.DataFrame:
    """Mock ESG sentiment scores derived from news feeds.

    In a real system you would plug in an NLP pipeline over RSS / news APIs.
    """
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "company_id": company_ids,
            "env_sentiment": rng.uniform(-1, 1, size=len(company_ids)),
            "soc_sentiment": rng.uniform(-1, 1, size=len(company_ids)),
            "gov_sentiment": rng.uniform(-1, 1, size=len(company_ids)),
        }
    )
    return df


def attach_satellite_to_companies(
    companies: pd.DataFrame,
    images: List[Image.Image],
) -> Dict[str, Image.Image]:
    """Assign one synthetic satellite image per company (round-robin)."""
    mapping: Dict[str, Image.Image] = {}
    for i, cid in enumerate(companies["company_id"].tolist()):
        mapping[cid] = images[i % len(images)]
    return mapping


# --- Connectors / placeholders for real external data sources ---


def fetch_sentinal_or_landsat_stub(*args: Any, **kwargs: Any) -> List[Image.Image]:
    """Placeholder for a real Sentinel/Landsat connector.

    Returns synthetic images but exposes a realistic function signature
    so it can be swapped with a production connector later.
    """
    n_images = kwargs.get("n_images", 4)
    size = kwargs.get("size", 224)
    return generate_synthetic_satellite_batch(n_images=n_images, size=size)


def fetch_opencorporates_stub(company_ids: List[str]) -> pd.DataFrame:
    """Placeholder for OpenCorporates / corporate registry connector.

    Returns simple synthetic features.
    """
    base = pd.DataFrame({"company_id": company_ids})
    rng = np.random.default_rng(999)
    base["num_subsidiaries"] = rng.integers(0, 20, size=len(company_ids))
    base["incorporation_age"] = rng.integers(
        1, 100, size=len(company_ids))  # years
    return base


def build_merged_entity_frame(n_companies: int = 5) -> Tuple[pd.DataFrame, Dict[str, Image.Image]]:
    """End-to-end synthetic ingestion step.

    Returns
    -------
    companies : DataFrame
        Tabular alternative data + news + registry-like signals.
    image_map : dict
        Mapping company_id -> synthetic satellite image.
    """
    companies = synthetic_company_metadata(n_companies=n_companies)
    news = synthetic_news_signals(companies["company_id"].tolist())
    registry = fetch_opencorporates_stub(companies["company_id"].tolist())

    companies = companies.merge(news, on="company_id", how="left")
    companies = companies.merge(registry, on="company_id", how="left")

    images = fetch_sentinal_or_landsat_stub(n_images=n_companies, size=224)
    image_map = attach_satellite_to_companies(companies, images)
    return companies, image_map

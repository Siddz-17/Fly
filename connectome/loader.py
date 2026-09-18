"""
connectome/loader.py

Connectome data loader supporting both:
1. Fast offline querying of local MaleCNS feather datasets via PyArrow dataset filters.
2. Online live querying of Janelia neuPrint server (male-cns:v1.0 and optic-lobe:v1.1).

Enforces Rule 1: No synthetic data is ever substituted for missing connectome data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.feather as feather

# Default local data paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "malecns"
ANNOTATIONS_PATH = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_PATH = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_PATH = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


class ConnectomeLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.annotations_path = self.data_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        self.nt_path = self.data_dir / "body-neurotransmitters-male-cns-v1.0.feather"
        self.weights_path = self.data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

        self._annotations: Optional[pd.DataFrame] = None
        self._neurotransmitters: Optional[pd.DataFrame] = None
        self._weights_dataset: Optional[ds.Dataset] = None

    def load_annotations(self) -> pd.DataFrame:
        """Load neuron annotations with bodyId, type, hex coordinates, somaSide."""
        if self._annotations is None:
            if not self.annotations_path.exists():
                raise FileNotFoundError(
                    f"Annotations feather file not found at {self.annotations_path}. "
                    "Refusing to generate synthetic neuron IDs (Rule 1)."
                )
            self._annotations = feather.read_feather(self.annotations_path)
        return self._annotations

    def load_neurotransmitters(self) -> pd.DataFrame:
        """Load consensus neurotransmitter predictions."""
        if self._neurotransmitters is None:
            if not self.nt_path.exists():
                raise FileNotFoundError(
                    f"Neurotransmitters feather file not found at {self.nt_path}."
                )
            self._neurotransmitters = feather.read_feather(self.nt_path)
        return self._neurotransmitters

    def get_weights_dataset(self) -> ds.Dataset:
        """Access PyArrow dataset for zero-copy, filtered querying of synaptic weights."""
        if self._weights_dataset is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(
                    f"Synaptic weights feather file not found at {self.weights_path}."
                )
            self._weights_dataset = ds.dataset(self.weights_path, format="feather")
        return self._weights_dataset

    def query_connections_for_bodies(
        self,
        pre_ids: Optional[list[int]] = None,
        post_ids: Optional[list[int]] = None,
        min_weight: int = 1,
    ) -> pd.DataFrame:
        """
        Query exact pairwise synaptic weights between candidate body IDs.
        Efficiently filtered via PyArrow without loading the 1.05 GB file into RAM.
        """
        dataset = self.get_weights_dataset()

        filter_expr = ds.field("weight") >= min_weight
        if pre_ids is not None:
            filter_expr = filter_expr & ds.field("body_pre").isin(pre_ids)
        if post_ids is not None:
            filter_expr = filter_expr & ds.field("body_post").isin(post_ids)

        table = dataset.to_table(filter=filter_expr)
        df = table.to_pandas()
        return df.rename(columns={"body_pre": "pre_id", "body_post": "post_id", "weight": "synapse_count"})

    def get_online_neuprint_client(self, dataset: str = "optic-lobe:v1.1"):
        """
        Connect to Janelia neuPrint API using token from environment.
        """
        try:
            from neuprint import Client
        except ImportError:
            raise ImportError("neuprint-python is not installed. Run: pip install neuprint-python")

        token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS") or os.environ.get("NEUPRINT_TOKEN")
        if not token:
            raise ValueError(
                "No neuPrint authentication token found in NEUPRINT_APPLICATION_CREDENTIALS "
                "or NEUPRINT_TOKEN. Refusing to continue with fake data."
            )

        client = Client("neuprint.janelia.org", dataset=dataset, token=token)
        return client

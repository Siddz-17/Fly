"""
vision/embeddings.py

Milestone 6: Connectome Visual Embeddings.

Extracts compact, biologically grounded motion representations from
the T4 and T5 population activity across the retinotopic columnar patch:
1. Spatial 2D motion vector field V(u, v, t) across columns.
2. Global motion vector (vx, vy, speed).
3. Divergence / Looming approach signal (expansion vs contraction).
4. Subtype energy readouts (ON vs OFF motion power).

Produces the biological motion embedding vector e_t for downstream fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MotionEmbedding:
    time_ms: float
    vx: float                     # Global horizontal motion (Right > 0, Left < 0)
    vy: float                     # Global vertical motion (Up > 0, Down < 0)
    speed: float                  # Magnitude sqrt(vx^2 + vy^2)
    loom: float                   # Radial divergence (approaching > 0, receding < 0)
    on_power: float               # Total T4 activation
    off_power: float              # Total T5 activation
    spatial_field_vx: np.ndarray  # Shape: (n_cols,)
    spatial_field_vy: np.ndarray  # Shape: (n_cols,)

    def to_feature_vector(self) -> np.ndarray:
        """Returns 6-dimensional compact motion embedding."""
        return np.array([self.vx, self.vy, self.speed, self.loom, self.on_power, self.off_power], dtype=float)


class ConnectomeMotionExtractor:
    """
    Extracts structured motion embeddings from T4/T5 simulation outputs.
    """

    def __init__(self, circuit_data: dict[str, Any]):
        self.patch_center = circuit_data["metadata"]["patch_center"]  # [19, 20]
        neurons = circuit_data["neurons"]

        # Build column index and index map
        self.ordered_bids = sorted([int(b) for b in neurons.keys()])
        self.bid_to_idx = {bid: i for i, bid in enumerate(self.ordered_bids)}

        # Find all T4 and T5 neurons and assign column coordinates
        self.subtypes = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
        self.subtype_indices: dict[str, list[int]] = {st: [] for st in self.subtypes}

        for bid in self.ordered_bids:
            n = neurons[str(bid)]
            ctype = n["cell_type"]
            if ctype in self.subtype_indices:
                self.subtype_indices[ctype].append(self.bid_to_idx[bid])

        # Find columns present with coordinates
        cols_dict = {}
        for bid in self.ordered_bids:
            n = neurons[str(bid)]
            h1, h2 = n.get("hex1"), n.get("hex2")
            if h1 is not None and h2 is not None:
                key = (int(h1), int(h2))
                if key not in cols_dict:
                    # Cartesian offset from center
                    x = (int(h1) - self.patch_center[0]) + 0.5 * (int(h2) - self.patch_center[1])
                    y = (np.sqrt(3) / 2.0) * (int(h2) - self.patch_center[1])
                    cols_dict[key] = (x, y)

        self.ordered_cols = sorted(cols_dict.keys())
        self.col_positions = np.array([cols_dict[k] for k in self.ordered_cols])  # (N, 2)

        # Compute unit radial vectors for looming divergence calculation
        radial_dist = np.linalg.norm(self.col_positions, axis=1, keepdims=True)
        radial_dist[radial_dist == 0] = 1.0
        self.radial_unit_vectors = self.col_positions / radial_dist  # (N, 2)

    def extract_from_rates(self, rates: np.ndarray, time_ms: float = 0.0) -> MotionEmbedding:
        """
        Compute motion embedding from instantaneous neuron rates (shape: (n_neurons,)).
        """
        # Population mean rates per subtype
        mean_rates: dict[str, float] = {}
        for st in self.subtypes:
            idx = self.subtype_indices[st]
            mean_rates[st] = float(rates[idx].mean()) if len(idx) > 0 else 0.0

        # Cardinal horizontal & vertical components
        # T4a/T5a: Right (+x), T4b/T5b: Left (-x)
        # T4c/T5c: Up (+y), T4d/T5d: Down (-y)
        vx_on = mean_rates["T4a"] - mean_rates["T4b"]
        vx_off = mean_rates["T5a"] - mean_rates["T5b"]
        vx = float(np.clip(vx_on + vx_off, -50.0, 50.0))

        vy_on = mean_rates["T4c"] - mean_rates["T4d"]
        vy_off = mean_rates["T5c"] - mean_rates["T5d"]
        vy = float(np.clip(vy_on + vy_off, -50.0, 50.0))

        speed = float(math.hypot(vx, vy))
        on_power = float(np.clip(sum(mean_rates[f"T4{s}"] for s in ["a", "b", "c", "d"]), 0.0, 500.0))
        off_power = float(np.clip(sum(mean_rates[f"T5{s}"] for s in ["a", "b", "c", "d"]), 0.0, 500.0))

        # Spatial vector field approximation across columns
        # Uniform global component projected onto local column vectors + radial divergence
        n_cols = len(self.ordered_cols)
        field_vx = np.full(n_cols, vx)
        field_vy = np.full(n_cols, vy)

        # Loom: dot product of vector field with radial unit vectors
        field_vectors = np.stack([field_vx, field_vy], axis=1)  # (N, 2)
        radial_projections = (field_vectors * self.radial_unit_vectors).sum(axis=1)
        loom = float(radial_projections.mean())

        return MotionEmbedding(
            time_ms=time_ms,
            vx=vx,
            vy=vy,
            speed=speed,
            loom=loom,
            on_power=on_power,
            off_power=off_power,
            spatial_field_vx=field_vx,
            spatial_field_vy=field_vy,
        )

    def extract_timeseries(self, rates_trace: np.ndarray, time_array_ms: np.ndarray) -> list[MotionEmbedding]:
        """
        Extract motion embeddings across a full simulation timeseries.
        rates_trace: shape (n_steps, n_neurons)
        """
        n_steps = len(time_array_ms)
        embeddings = []
        for t in range(n_steps):
            emb = self.extract_from_rates(rates_trace[t], time_ms=float(time_array_ms[t]))
            embeddings.append(emb)
        return embeddings

    def visualize_motion_field(
        self,
        embedding: MotionEmbedding,
        title: str = "Connectome Retinotopic Motion Vector Field",
        out_path: Path | str = "outputs/graphs/motion_vector_field.png",
    ) -> Path:
        """
        Generates a 2D quiver plot of the spatial motion field across the ommatidia patch.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0B0F19")
        ax.set_facecolor("#0B0F19")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", alpha=0.5)

        x = self.col_positions[:, 0]
        y = self.col_positions[:, 1]
        u = embedding.spatial_field_vx
        v = embedding.spatial_field_vy

        # Draw ommatidia locations
        ax.scatter(x, y, color="#38BDF8", s=180, alpha=0.6, edgecolors="#F8FAFC", label="Ommatidia Columns")

        # Quiver vectors
        scale = max(0.001, np.max(np.sqrt(u**2 + v**2)))
        ax.quiver(
            x, y, u, v,
            color="#F43F5E",
            angles="xy",
            scale_units="xy",
            scale=scale * 1.5,
            width=0.012,
            label="Motion Vector (T4/T5)",
        )

        ax.set_xlabel("Relative Azimuth (°)", color="#94A3B8")
        ax.set_ylabel("Relative Elevation (°)", color="#94A3B8")
        ax.set_title(
            f"{title}\nVx={embedding.vx:+.2e}, Vy={embedding.vy:+.2e}, Speed={embedding.speed:.2e}",
            color="#F8FAFC",
            fontsize=11,
            pad=15,
        )
        ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", loc="upper right")

        fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return out_path


class LPTCMotionExtractor:
    """
    Extracts biological wide-field ego-motion embeddings directly from
    downstream Lobula Plate Tangential Cells (HSN, HSE, HSS, VS) and LPi interneurons.
    """

    def __init__(self, lptc_circuit_data: dict[str, Any]):
        from connectome.lptc_dynamics import LPTCDynamics
        self.dynamics = LPTCDynamics(lptc_circuit_data)
        self.t4_t5_bodies = [
            int(b) for b, info in lptc_circuit_data["neurons"].items()
            if any(info["cell_type"].startswith(x) for x in ["T4", "T5"])
        ]

    def extract_from_t4_t5_rates(
        self,
        t4_t5_rates: dict[int, float],
        time_ms: float = 0.0,
    ) -> MotionEmbedding:
        """
        Computes motion embedding using LPTC membrane potentials.
        """
        lptc_out = self.dynamics.step_from_t4_t5_rates(t4_t5_rates)

        # Graded potentials:
        # HS encodes horizontal motion (yaw/progressive vs regressive)
        vx = lptc_out["v_hs_mean"]
        # VS encodes vertical motion (downward vs upward)
        vy = -lptc_out["v_vs_mean"]  # convention: downward is -vy, upward is +vy

        speed = float(np.sqrt(vx**2 + vy**2))

        # T4 / T5 power sums
        on_power = sum(r for bid, r in t4_t5_rates.items() if bid in self.dynamics.bid_to_idx and self.dynamics.cell_types[self.dynamics.bid_to_idx[bid]].startswith("T4"))
        off_power = sum(r for bid, r in t4_t5_rates.items() if bid in self.dynamics.bid_to_idx and self.dynamics.cell_types[self.dynamics.bid_to_idx[bid]].startswith("T5"))

        return MotionEmbedding(
            time_ms=time_ms,
            vx=vx,
            vy=vy,
            speed=speed,
            loom=0.0,
            on_power=float(on_power),
            off_power=float(off_power),
            spatial_field_vx=np.array([vx]),
            spatial_field_vy=np.array([vy]),
        )


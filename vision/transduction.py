"""
vision/transduction.py

Milestone 2: Retinotopic Transduction Hierarchy.

Transforms continuous visual scenes into biological neural drive:
    Visual Scene S(x, y, t)
          │
          ▼
    Ommatidia Array (25 hex columns, interommatidial angle Δφ ≈ 5°)
          │
          ▼
    Photoreceptors R(u, v, t) with spatial Gaussian acceptance angle
          │
          ▼
    Lamina Monopolar Cells (L1, L2, L3)
          ├─► L1(u, v, t) = max(0, +dR/dt)   [ON brightness increment]
          ├─► L2(u, v, t) = max(0, -dR/dt)   [OFF brightness decrement]
          └─► L3(u, v, t) = LowPass(R(t))     [Slow / adaptation channel]
          │
          ▼
    Direct synaptic input currents into male connectome L1, L2, L3 neurons.

Strictly adheres to:
- [ENGINEERED INTERFACE]: Spatial optics & Gaussian receptive fields.
- [REAL CONNECTOME]: Targets exact body IDs of L1, L2, L3 per column in male-cns:v1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class Ommatidium:
    hex1: int
    hex2: int
    x_deg: float
    y_deg: float
    l1_id: int | None = None
    l2_id: int | None = None
    l3_id: int | None = None


class RetinotopicTransducer:
    """
    Simulates optical sampling and lamina preprocessing across the 25-column patch.
    """

    def __init__(
        self,
        circuit_data: dict[str, Any],
        interommatidial_angle_deg: float = 5.0,
        acceptance_angle_deg: float = 5.7,
        tau_photo_ms: float = 10.0,
        tau_lamina_ms: float = 15.0,
        dt_ms: float = 0.5,
    ):
        self.delta_phi = interommatidial_angle_deg
        self.sigma_deg = acceptance_angle_deg / 2.355  # FWHM to Gaussian sigma
        self.tau_photo = tau_photo_ms
        self.tau_lamina = tau_lamina_ms
        self.dt = dt_ms

        patch_center = circuit_data["metadata"]["patch_center"]  # [19, 20]
        self.u0, self.v0 = patch_center[0], patch_center[1]

        # Build column index and map Lamina body IDs
        self.ommatidia: dict[tuple[int, int], Ommatidium] = {}
        neurons = circuit_data["neurons"]

        # Collect unique columns within the patch bounds
        h1_min, h1_max = circuit_data["metadata"]["hex1_range"]
        h2_min, h2_max = circuit_data["metadata"]["hex2_range"]
        cols_present = set()
        for n in neurons.values():
            h1, h2 = n.get("hex1"), n.get("hex2")
            if h1 is not None and h2 is not None:
                u, v = int(h1), int(h2)
                if h1_min <= u <= h1_max and h2_min <= v <= h2_max:
                    cols_present.add((u, v))

        for u, v in cols_present:
            # Hexagonal to Cartesian angle conversion
            # Axial coordinates: u along x, v along 60-deg axis
            x = (u - self.u0) * self.delta_phi + 0.5 * (v - self.v0) * self.delta_phi
            y = (np.sqrt(3) / 2.0) * (v - self.v0) * self.delta_phi
            self.ommatidia[(u, v)] = Ommatidium(hex1=u, hex2=v, x_deg=x, y_deg=y)

        # Assign Lamina cell body IDs
        for n in neurons.values():
            h1, h2 = n.get("hex1"), n.get("hex2")
            if h1 is not None and h2 is not None:
                key = (int(h1), int(h2))
                if key in self.ommatidia:
                    bid = n["body_id"]
                    ctype = n["cell_type"]
                    if ctype == "L1":
                        self.ommatidia[key].l1_id = bid
                    elif ctype == "L2":
                        self.ommatidia[key].l2_id = bid
                    elif ctype == "L3":
                        self.ommatidia[key].l3_id = bid

        # Array of ommatidia positions (N, 2)
        self.ordered_keys = sorted(self.ommatidia.keys())
        self.positions_deg = np.array(
            [[self.ommatidia[k].x_deg, self.ommatidia[k].y_deg] for k in self.ordered_keys]
        )
        self.n_ommatidia = len(self.ordered_keys)

        # Precompute vectorized sampling grid across all ommatidia simultaneously
        n_sub = 5
        offsets = np.linspace(-2.0 * self.sigma_deg, 2.0 * self.sigma_deg, n_sub)
        dx_grid, dy_grid = np.meshgrid(offsets, offsets)
        self.dx_flat = dx_grid.flatten()
        self.dy_flat = dy_grid.flatten()
        sub_weights = np.exp(-(self.dx_flat**2 + self.dy_flat**2) / (2.0 * self.sigma_deg**2))
        self.sub_weights = sub_weights / sub_weights.sum()

        # Shape: (n_ommatidia, n_sub_points)
        self.grid_x = self.positions_deg[:, 0:1] + self.dx_flat.reshape(1, -1)
        self.grid_y = self.positions_deg[:, 1:2] + self.dy_flat.reshape(1, -1)

    def sample_scene(
        self,
        scene_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        t_ms: float,
    ) -> np.ndarray:
        """
        Sample continuous visual scene S(x, y, t) through ommatidial Gaussian apertures.
        Fully vectorized across all ommatidia and sub-points.
        """
        lum = scene_fn(self.grid_x, self.grid_y, t_ms)
        # Weighted sum across sub-points (axis=1)
        return (lum * self.sub_weights).sum(axis=1)

    def transduce_timeseries(
        self,
        scene_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        duration_ms: float,
    ) -> dict[str, Any]:
        """
        Run full temporal transduction over a visual stimulus timeseries.

        Returns:
            {
                "time_ms": array (n_steps,),
                "photoreceptors": array (n_steps, n_ommatidia),
                "L1": array (n_steps, n_ommatidia),  # ON
                "L2": array (n_steps, n_ommatidia),  # OFF
                "L3": array (n_steps, n_ommatidia),  # Slow adaptation
                "body_currents": dict mapping body_id -> array (n_steps,)
            }
        """
        n_steps = int(np.ceil(duration_ms / self.dt))
        time_ms = np.arange(n_steps) * self.dt

        photo_state = np.zeros(self.n_ommatidia)
        lamina_slow = np.zeros(self.n_ommatidia)

        photo_records = np.zeros((n_steps, self.n_ommatidia))
        l1_records = np.zeros((n_steps, self.n_ommatidia))
        l2_records = np.zeros((n_steps, self.n_ommatidia))
        l3_records = np.zeros((n_steps, self.n_ommatidia))

        # Initial optical sample
        init_sample = self.sample_scene(scene_fn, 0.0)
        photo_state[:] = init_sample
        lamina_slow[:] = init_sample

        alpha_photo = self.dt / self.tau_photo
        alpha_lamina = self.dt / self.tau_lamina

        for t_idx in range(n_steps):
            t = time_ms[t_idx]
            raw_input = self.sample_scene(scene_fn, t)

            # Photoreceptor temporal low-pass filter
            photo_state += alpha_photo * (raw_input - photo_state)
            photo_records[t_idx] = photo_state

            # Lamina high-pass filter: signal minus slow running average
            lamina_slow += alpha_lamina * (photo_state - lamina_slow)
            temporal_diff = photo_state - lamina_slow

            # ON (L1) and OFF (L2) half-wave rectification
            l1 = np.maximum(0.0, temporal_diff)
            l2 = np.maximum(0.0, -temporal_diff)
            l3 = photo_state  # L3 carries sustained/slow signal

            l1_records[t_idx] = l1
            l2_records[t_idx] = l2
            l3_records[t_idx] = l3

        # Map to specific Lamina body IDs
        body_currents: dict[int, np.ndarray] = {}
        for i, key in enumerate(self.ordered_keys):
            om = self.ommatidia[key]
            if om.l1_id is not None:
                body_currents[om.l1_id] = l1_records[:, i]
            if om.l2_id is not None:
                body_currents[om.l2_id] = l2_records[:, i]
            if om.l3_id is not None:
                body_currents[om.l3_id] = l3_records[:, i]

        return {
            "time_ms": time_ms,
            "photoreceptors": photo_records,
            "L1": l1_records,
            "L2": l2_records,
            "L3": l3_records,
            "body_currents": body_currents,
            "ordered_keys": self.ordered_keys,
            "positions_deg": self.positions_deg,
        }

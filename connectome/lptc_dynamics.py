"""
connectome/lptc_dynamics.py

Milestone MVP-B: Lobula Plate Tangential Cell (LPTC) Temporal Dynamics Simulator.

Models the wide-field integration layer of the male Drosophila visual system:
1. Cholinergic direct excitation (+1) from T4a/T5a onto Horizontal System (HSN, HSE, HSS, HST).
2. Disynaptic GABAergic inhibition (-1) via LPi21 (excited by T4b/T5b, inhibiting HS cells).
3. Cholinergic direct excitation (+1) from T4d/T5d onto Vertical System (VS cells).
4. Disynaptic Glutamatergic/GluClalpha inhibition (-1) via LPi34 (excited by T4c/T5c, inhibiting VS cells).
5. Continuous graded membrane potential integration with push-pull motion opponency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from connectome.dynamics import WeightNormalization


class LPTCDynamics:
    """
    Simulates wide-field Lobula Plate Tangential Cells (LPTCs) and their
    reciprocal inhibitory interneurons (LPi) receiving presynaptic inputs
    from the 25-column retinotopic T4/T5 motion detectors.
    """

    def __init__(
        self,
        circuit_data: dict[str, Any],
        normalization: str = WeightNormalization.ROW_NORM,
        dt_ms: float = 0.5,
        tau_lptc_ms: float = 20.0,
        tau_lpi_ms: float = 10.0,
        synaptic_gain: float = 1.0,
        baseline_mode: str = "real",  # 'real', 'shuffled_weights', 'scrambled_topology', 'no_inhibition'
        random_seed: int = 42,
    ):
        self.dt = dt_ms
        self.tau_lptc = tau_lptc_ms
        self.tau_lpi = tau_lpi_ms
        self.synaptic_gain = synaptic_gain
        self.normalization = normalization
        self.baseline_mode = baseline_mode
        self.rng = np.random.default_rng(random_seed)

        neurons = circuit_data["neurons"]
        self.all_bids = sorted([int(b) for b in neurons.keys()])
        self.bid_to_idx = {bid: i for i, bid in enumerate(self.all_bids)}
        self.idx_to_bid = {i: bid for i, bid in enumerate(self.all_bids)}
        self.n_neurons = len(self.all_bids)

        self.cell_types = [neurons[str(bid)]["cell_type"] for bid in self.all_bids]
        self.neurotransmitters = [neurons[str(bid)].get("neurotransmitter") for bid in self.all_bids]

        # Group indices
        self.hs_indices: list[int] = []
        self.vs_indices: list[int] = []
        self.lpi_indices: list[int] = []
        self.t4_indices: dict[str, list[int]] = {"a": [], "b": [], "c": [], "d": []}
        self.t5_indices: dict[str, list[int]] = {"a": [], "b": [], "c": [], "d": []}

        for i, ctype in enumerate(self.cell_types):
            if ctype in {"HSN", "HSE", "HSS", "HST"}:
                self.hs_indices.append(i)
            elif ctype in {"VS", "VSm", "VST1", "VST2"}:
                self.vs_indices.append(i)
            elif ctype.startswith("LPi"):
                self.lpi_indices.append(i)
            elif ctype.startswith("T4"):
                sub = ctype[-1]
                if sub in self.t4_indices:
                    self.t4_indices[sub].append(i)
            elif ctype.startswith("T5"):
                sub = ctype[-1]
                if sub in self.t5_indices:
                    self.t5_indices[sub].append(i)

        # Build synaptic weight matrix
        self.weights = self._build_weight_matrix(circuit_data["connections"])

        # Membrane states
        self.v_rest = -60.0  # mV
        self.v = np.full(self.n_neurons, self.v_rest, dtype=float)

    def _build_weight_matrix(self, connections: list[dict[str, Any]]) -> np.ndarray:
        raw_w = np.zeros((self.n_neurons, self.n_neurons), dtype=float)
        signs = np.ones((self.n_neurons, self.n_neurons), dtype=float)

        for c in connections:
            pre_id = c["pre_id"]
            post_id = c["post_id"]
            if pre_id in self.bid_to_idx and post_id in self.bid_to_idx:
                i = self.bid_to_idx[pre_id]
                j = self.bid_to_idx[post_id]

                pre_type = self.cell_types[i]
                post_type = self.cell_types[j]
                syn = float(c["synapse_count"])
                raw_w[i, j] += syn

                # Inferred biological sign:
                nt = self.neurotransmitters[i]
                if nt == "gaba":
                    signs[i, j] = -1.0
                elif nt == "glutamate":
                    # Inhibitory in optic lobe via GluClalpha
                    signs[i, j] = -1.0
                elif nt == "acetylcholine":
                    signs[i, j] = +1.0
                else:
                    signs[i, j] = +1.0

        # Baseline ablations
        if self.baseline_mode == "no_inhibition":
            # Zero out all inhibitory connections from LPi
            for i in self.lpi_indices:
                raw_w[i, :] = 0.0

        elif self.baseline_mode == "shuffled_weights":
            nonzero_mask = raw_w > 0
            nonzero_vals = raw_w[nonzero_mask]
            self.rng.shuffle(nonzero_vals)
            raw_w[nonzero_mask] = nonzero_vals

        elif self.baseline_mode == "scrambled_topology":
            nonzero_vals = raw_w[raw_w > 0]
            raw_w = np.zeros_like(raw_w)
            n_edges = len(nonzero_vals)
            rand_pre = self.rng.integers(0, self.n_neurons, size=n_edges)
            rand_post = self.rng.integers(0, self.n_neurons, size=n_edges)
            for k in range(n_edges):
                raw_w[rand_pre[k], rand_post[k]] += nonzero_vals[k]

        # Normalization
        norm = self.normalization
        eff_w = np.zeros_like(raw_w)

        if norm == WeightNormalization.RAW:
            scale = 0.05 * self.synaptic_gain
            eff_w = signs * raw_w * scale
        elif norm == WeightNormalization.ROW_NORM:
            row_sums = raw_w.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            eff_w = signs * (raw_w / row_sums) * self.synaptic_gain
        elif norm == WeightNormalization.COLUMN_NORM:
            col_sums = raw_w.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            eff_w = signs * (raw_w / col_sums) * self.synaptic_gain
        else:
            scale = 0.05 * self.synaptic_gain
            eff_w = signs * raw_w * scale

        return eff_w

    def reset(self) -> None:
        self.v.fill(self.v_rest)

    def step_from_t4_t5_rates(self, t4_t5_rates: dict[int, float]) -> dict[str, float]:
        """
        Advance LPTC and LPi dynamics by one timestep dt given presynaptic T4/T5 firing rates.

        t4_t5_rates: map of body_id -> rate (Hz or normalized activation)
        Returns:
            dict containing mean membrane potentials for HS, VS, and individual components.
        """
        # Form presynaptic rate vector
        r = np.zeros(self.n_neurons, dtype=float)
        for bid, rate in t4_t5_rates.items():
            if bid in self.bid_to_idx:
                r[self.bid_to_idx[bid]] = rate

        # LPi interneuron rates from previous membrane potentials
        for idx in self.lpi_indices:
            r[idx] = max(0.0, (self.v[idx] - self.v_rest) / 10.0)

        # Synaptic current into each postsynaptic neuron: I = W^T @ r
        i_syn = self.weights.T @ r

        # Membrane integration with separate time constants
        tau = np.full(self.n_neurons, self.tau_lptc)
        for idx in self.lpi_indices:
            tau[idx] = self.tau_lpi

        dv = (-(self.v - self.v_rest) + i_syn) * (self.dt / tau)
        self.v += dv

        # Extract graded potentials delta_V = V - V_rest
        delta_v = self.v - self.v_rest

        v_hsn = float(np.mean([delta_v[i] for i in self.hs_indices if self.cell_types[i] == "HSN"])) if any(self.cell_types[i] == "HSN" for i in self.hs_indices) else 0.0
        v_hse = float(np.mean([delta_v[i] for i in self.hs_indices if self.cell_types[i] == "HSE"])) if any(self.cell_types[i] == "HSE" for i in self.hs_indices) else 0.0
        v_hss = float(np.mean([delta_v[i] for i in self.hs_indices if self.cell_types[i] == "HSS"])) if any(self.cell_types[i] == "HSS" for i in self.hs_indices) else 0.0
        v_hs_mean = float(np.mean([delta_v[i] for i in self.hs_indices])) if self.hs_indices else 0.0
        v_vs_mean = float(np.mean([delta_v[i] for i in self.vs_indices])) if self.vs_indices else 0.0
        v_lpi21 = float(np.mean([delta_v[i] for i in self.lpi_indices if self.cell_types[i] == "LPi21"])) if any(self.cell_types[i] == "LPi21" for i in self.lpi_indices) else 0.0
        v_lpi34 = float(np.mean([delta_v[i] for i in self.lpi_indices if self.cell_types[i] == "LPi34"])) if any(self.cell_types[i] == "LPi34" for i in self.lpi_indices) else 0.0

        return {
            "v_hs_mean": v_hs_mean,
            "v_hsn": v_hsn,
            "v_hse": v_hse,
            "v_hss": v_hss,
            "v_vs_mean": v_vs_mean,
            "v_lpi21": v_lpi21,
            "v_lpi34": v_lpi34,
        }

    def simulate_sequence(
        self,
        t4_t5_rate_sequence: list[dict[int, float]],
    ) -> dict[str, np.ndarray]:
        """
        Simulates an entire sequence of timesteps and returns time-series arrays.
        """
        self.reset()
        n_steps = len(t4_t5_rate_sequence)
        hs_trace = np.zeros(n_steps)
        hsn_trace = np.zeros(n_steps)
        hse_trace = np.zeros(n_steps)
        hss_trace = np.zeros(n_steps)
        vs_trace = np.zeros(n_steps)
        lpi21_trace = np.zeros(n_steps)
        lpi34_trace = np.zeros(n_steps)

        for t, rate_dict in enumerate(t4_t5_rate_sequence):
            out = self.step_from_t4_t5_rates(rate_dict)
            hs_trace[t] = out["v_hs_mean"]
            hsn_trace[t] = out["v_hsn"]
            hse_trace[t] = out["v_hse"]
            hss_trace[t] = out["v_hss"]
            vs_trace[t] = out["v_vs_mean"]
            lpi21_trace[t] = out["v_lpi21"]
            lpi34_trace[t] = out["v_lpi34"]

        return {
            "hs": hs_trace,
            "hsn": hsn_trace,
            "hse": hse_trace,
            "hss": hss_trace,
            "vs": vs_trace,
            "lpi21": lpi21_trace,
            "lpi34": lpi34_trace,
        }

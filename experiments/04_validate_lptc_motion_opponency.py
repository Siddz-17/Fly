"""
experiments/04_validate_lptc_motion_opponency.py

Milestone MVP-B Experiment: Biological Motion Opponency in Lobula Plate Tangential Cells.

Evaluates Hypothesis H2:
    H2: Lobula Plate Tangential Cells (HSN, HSE, HSS, VS) constrained by real male
    connectome connectivity and LPi interneuron wiring exhibit biological push-pull
    motion opponency (preferred-direction excitation, null-direction hyperpolarization)
    that collapses under sign-shuffled, weight-shuffled, or no-inhibition controls.

Generates:
- outputs/graphs/lptc_motion_opponency.png (Push-pull voltage traces over time)
- outputs/graphs/lptc_polar_tuning.png (Directional tuning curves)
- outputs/audit/lptc_opponency_benchmark.csv (Opponency indices across conditions)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connectome.dynamics import CircuitDynamics, WeightNormalization
from connectome.lptc_dynamics import LPTCDynamics
from vision.stimuli import create_moving_bar_stimulus
from vision.transduction import RetinotopicTransducer

CIRCUIT_A_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"
CIRCUIT_B_PATH = REPO_ROOT / "data" / "processed" / "mvp_b_circuit.json"
OUT_TRACES = REPO_ROOT / "outputs" / "graphs" / "lptc_motion_opponency.png"
OUT_POLAR = REPO_ROOT / "outputs" / "graphs" / "lptc_polar_tuning.png"
OUT_CSV = REPO_ROOT / "outputs" / "audit" / "lptc_opponency_benchmark.csv"


def run_lptc_opponency_experiment():
    print("[M-MVP-B] Loading circuit models...")
    with open(CIRCUIT_A_PATH, "r", encoding="utf-8") as f:
        circuit_a = json.load(f)
    with open(CIRCUIT_B_PATH, "r", encoding="utf-8") as f:
        circuit_b = json.load(f)

    dt_ms = 0.5
    duration_ms = 400.0
    n_steps = int(duration_ms / dt_ms)
    time_ms = np.arange(n_steps) * dt_ms

    transducer = RetinotopicTransducer(circuit_a, dt_ms=dt_ms)
    directions = ["right", "left", "up", "down"]
    dir_labels = {
        "right": "Front-to-Back (Progressive)",
        "left": "Back-to-Front (Regressive)",
        "up": "Upward",
        "down": "Downward",
    }

    print("[M-MVP-B] Precomputing visual stimulus transduction across cardinal directions...")
    stimuli = {}
    for d in directions:
        scene_fn = create_moving_bar_stimulus(
            direction=d,
            velocity_deg_s=80.0,
            bar_width_deg=6.0,
            contrast=1.0,
        )
        stimuli[d] = transducer.transduce_timeseries(scene_fn, duration_ms)

    # 1. Simulate MVP-A circuit to obtain T4/T5 firing rate timeseries for each direction
    print("[M-MVP-B] Simulating core retinotopic circuit (Lamina -> Medulla -> T4/T5)...")
    core_sim = CircuitDynamics(circuit_a, normalization=WeightNormalization.ROW_NORM, dt_ms=dt_ms)
    all_raw_rates = {}
    for d in directions:
        sim_out = core_sim.simulate(stimuli[d]["body_currents"], n_steps)
        all_raw_rates[d] = sim_out["rates"]
        bids = sim_out["ordered_bids"]

    # Normalize strictly by peak T4/T5 firing rate across all directions to obtain normalized biological drive [0, 1]
    t4_t5_indices = [i for i, b in enumerate(bids) if any(circuit_a["neurons"][str(b)]["cell_type"].startswith(x) for x in ["T4", "T5"])]
    max_t4_rate = max(np.max(all_raw_rates[d][:, t4_t5_indices]) for d in directions)
    if max_t4_rate <= 0:
        max_t4_rate = 1.0

    print(f"[M-MVP-B] Peak T4/T5 rate across directions: {max_t4_rate:.2e} (normalizing to [0, 1])")

    t4_t5_timeseries_by_dir = {}
    for d in directions:
        rates = all_raw_rates[d]
        dir_seq = []
        for t in range(n_steps):
            rate_dict = {bids[i]: float(rates[t, i] / max_t4_rate) for i in range(len(bids))}
            dir_seq.append(rate_dict)
        t4_t5_timeseries_by_dir[d] = dir_seq

    # 2. Test LPTC conditions
    conditions = [
        ("real", "Real Connectome (Push-Pull ACh + GABA)"),
        ("no_inhibition", "No Inhibition (LPi Ablated)"),
        ("shuffled_weights", "Shuffled Weights Control"),
        ("scrambled_topology", "Scrambled Topology Control"),
    ]

    all_results = {}
    csv_rows = []

    for cond_key, cond_name in conditions:
        print(f"[M-MVP-B] Simulating LPTC condition: {cond_name}...")
        lptc_sim = LPTCDynamics(
            circuit_b,
            normalization=WeightNormalization.ROW_NORM,
            dt_ms=dt_ms,
            synaptic_gain=3.0,
            baseline_mode=cond_key,
        )

        cond_traces = {}
        for d in directions:
            traces = lptc_sim.simulate_sequence(t4_t5_timeseries_by_dir[d])
            cond_traces[d] = traces

        all_results[cond_key] = cond_traces

        # Compute peak responses:
        # For HS cells: Preferred is Right (Front-to-Back), Null is Left (Back-to-Front)
        v_hs_pref = float(np.max(cond_traces["right"]["hs"]))
        v_hs_null = float(np.max(cond_traces["left"]["hs"]))
        hs_diff = v_hs_pref - v_hs_null
        hs_dsi = (v_hs_pref - v_hs_null) / (abs(v_hs_pref) + abs(v_hs_null) + 1e-6)

        # For VS cells: Preferred is Downward, Null is Upward
        v_vs_pref = float(np.max(cond_traces["down"]["vs"]))
        v_vs_null = float(np.max(cond_traces["up"]["vs"]))
        vs_diff = v_vs_pref - v_vs_null
        vs_dsi = (v_vs_pref - v_vs_null) / (abs(v_vs_pref) + abs(v_vs_null) + 1e-6)

        csv_rows.append({
            "condition": cond_key,
            "condition_name": cond_name,
            "hs_v_pref_mv": round(v_hs_pref, 3),
            "hs_v_null_mv": round(v_hs_null, 3),
            "hs_delta_mv": round(hs_diff, 3),
            "hs_dsi": round(hs_dsi, 4),
            "vs_v_pref_mv": round(v_vs_pref, 3),
            "vs_v_null_mv": round(v_vs_null, 3),
            "vs_delta_mv": round(vs_diff, 3),
            "vs_dsi": round(vs_dsi, 4),
        })

    # Save benchmark CSV
    df = pd.DataFrame(csv_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[M-MVP-B] Saved benchmark metrics to {OUT_CSV}")
    print(df[["condition", "hs_v_pref_mv", "hs_v_null_mv", "hs_dsi", "vs_v_pref_mv", "vs_dsi"]].to_string())

    # 3. Generate Push-Pull Voltage Traces Plot
    plot_push_pull_traces(all_results, time_ms, OUT_TRACES)

    # 4. Generate Polar Tuning Curves Plot
    plot_lptc_polar_tuning(all_results, OUT_POLAR)

    print(f"[M-MVP-B] Opponency validation complete. Figures saved.")
    return df


def plot_push_pull_traces(results: dict[str, Any], time_ms: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0B0F19")
    fig.suptitle(
        "Biological Motion Opponency in Lobula Plate Tangential Cells (LPTCs)\nReal Male Drosophila Connectome (male-cns:v1.0)",
        color="#F8FAFC",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    style_map = {
        "right": ("#38BDF8", "-", "Front-to-Back (Pref)"),
        "left": ("#F43F5E", "-", "Back-to-Front (Null)"),
        "down": ("#34D399", "--", "Downward"),
        "up": ("#F59E0B", "--", "Upward"),
    }

    cond_axes = [
        ("real", axes[0, 0], "A. Real Connectome: Intact Push-Pull Opponency"),
        ("no_inhibition", axes[0, 1], "B. No-Inhibition Control (LPi Interneurons Ablated)"),
        ("shuffled_weights", axes[1, 0], "C. Shuffled Weights Control"),
        ("scrambled_topology", axes[1, 1], "D. Scrambled Topology Control"),
    ]

    for cond_key, ax, title in cond_axes:
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", alpha=0.6)
        ax.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.0)

        traces = results[cond_key]
        for d, (color, ls, label) in style_map.items():
            hs_trace = traces[d]["hs"]
            ax.plot(time_ms, hs_trace, color=color, linestyle=ls, linewidth=2.0, label=label)

        # Plot LPi21 activity for real connectome to show inhibitory timing
        if cond_key == "real":
            lpi_null = traces["left"]["lpi21"]
            ax.plot(time_ms, -lpi_null, color="#E879F9", linestyle=":", linewidth=1.8, label="LPi21 Inhibitory Drive (x-1)")

        ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="semibold", pad=8)
        ax.set_xlabel("Time (ms)", color="#94A3B8", fontsize=10)
        ax.set_ylabel(r"Horizontal System $\Delta V_{\mathrm{HS}}$ (mV)", color="#94A3B8", fontsize=10)
        ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[M-MVP-B] Saved push-pull voltage traces to {out_path}")


def plot_lptc_polar_tuning(results: dict[str, Any], out_path: Path):
    angles = np.array([0, np.pi, np.pi / 2, 3 * np.pi / 2])  # right, left, up, down
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(13, 6), facecolor="#0B0F19")
    fig.suptitle("LPTC Directional Polar Tuning Curves", color="#F8FAFC", fontsize=13, fontweight="bold", y=0.98)

    for ax, target, title in [(ax1, "hs", "Horizontal System (HSN, HSE, HSS)"), (ax2, "vs", "Vertical System (VS1–VS6)")]:
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", alpha=0.6)
        ax.set_theta_zero_location("E")
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0° (Right)", "90° (Up)", "180° (Left)", "270° (Down)"], color="#94A3B8")

        # Plot real connectome
        r_real = [np.max(results["real"][d][target]) for d in ["right", "left", "up", "down"]]
        r_real_closed = np.concatenate([r_real, [r_real[0]]])
        ax.plot(angles_closed, r_real_closed, color="#38BDF8", linewidth=2.5, marker="o", label="Real Connectome")
        ax.fill(angles_closed, r_real_closed, color="#38BDF8", alpha=0.2)

        # Plot shuffled weights
        r_shuff = [np.max(results["shuffled_weights"][d][target]) for d in ["right", "left", "up", "down"]]
        r_shuff_closed = np.concatenate([r_shuff, [r_shuff[0]]])
        ax.plot(angles_closed, r_shuff_closed, color="#F59E0B", linewidth=1.5, linestyle="--", label="Shuffled Weights")

        ax.set_title(title, color="#F8FAFC", fontsize=11, fontweight="semibold", pad=15)
        ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", loc="upper left", bbox_to_anchor=(0.9, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[M-MVP-B] Saved polar tuning curves to {out_path}")


if __name__ == "__main__":
    run_lptc_opponency_experiment()

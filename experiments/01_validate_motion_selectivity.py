"""
experiments/01_validate_motion_selectivity.py

Milestone 5: Biological Direction Selectivity Validation & Baseline Experiments.

Evaluates Central Hypothesis H1:
    H1: A computational model constrained by real male Drosophila optic-lobe connectivity
    produces measurable direction-selective responses to visual motion, and these
    responses differ from topology/weight-shuffled controls.

Tests:
1. Real Connectome vs. Shuffled Weights vs. Scrambled Topology.
2. Weight Normalization Ablation (Column-norm, Row-norm, Raw, Log, Degree-norm).
3. Produces Directional Polar Tuning Curves, Activation Traces, and Ablation Metrics.

Saves:
- outputs/audit/normalization_ablation.csv
- outputs/graphs/t4_polar_tuning.png
- outputs/rasters/t4_directional_activation.png
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
from vision.stimuli import create_moving_bar_stimulus, create_static_flash_stimulus
from vision.transduction import RetinotopicTransducer

CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"
OUT_AUDIT = REPO_ROOT / "outputs" / "audit" / "normalization_ablation.csv"
OUT_POLAR = REPO_ROOT / "outputs" / "graphs" / "t4_polar_tuning.png"
OUT_RASTER = REPO_ROOT / "outputs" / "rasters" / "t4_directional_activation.png"


def precompute_directional_stimuli(
    transducer: RetinotopicTransducer,
    duration_ms: float = 600.0,
    velocity_deg_s: float = 60.0,
) -> dict[str, dict[str, Any]]:
    """Precompute stimulus transduction once per direction to avoid redundant simulation."""
    directions = ["right", "left", "up", "down"]
    precomputed = {}
    for d in directions:
        scene_fn = create_moving_bar_stimulus(
            direction=d,
            velocity_deg_s=velocity_deg_s,
            bar_width_deg=6.0,
            contrast=1.0,
        )
        precomputed[d] = transducer.transduce_timeseries(scene_fn, duration_ms)
    return precomputed


def run_direction_test_with_cache(
    circuit_data: dict[str, Any],
    stimuli_cache: dict[str, dict[str, Any]],
    dt_ms: float = 0.5,
    normalization: str = WeightNormalization.COLUMN_NORM,
    baseline_mode: str = "real",
    duration_ms: float = 600.0,
) -> dict[str, dict[str, float]]:
    """
    Run visual stimulus in 4 cardinal directions and measure peak response of T4 subtypes.
    Reuses precomputed Lamina currents.
    """
    directions = ["right", "left", "up", "down"]
    results: dict[str, dict[str, float]] = {}

    sim = CircuitDynamics(
        circuit_data,
        normalization=normalization,
        baseline_mode=baseline_mode,
        dt_ms=dt_ms,
    )

    t4_indices: dict[str, list[int]] = {
        "T4a": [], "T4b": [], "T4c": [], "T4d": [],
        "T5a": [], "T5b": [], "T5c": [], "T5d": [],
    }
    for i, ctype in enumerate(sim.cell_types):
        if ctype in t4_indices:
            t4_indices[ctype].append(i)

    n_steps = int(np.ceil(duration_ms / dt_ms))

    for direction in directions:
        trans_out = stimuli_cache[direction]
        sim_out = sim.simulate(trans_out["body_currents"], n_steps)

        rates = sim_out["rates"]
        dir_summary: dict[str, float] = {}

        for subtype, idx_list in t4_indices.items():
            if idx_list:
                sub_rates = rates[:, idx_list]
                mean_pop = sub_rates.mean(axis=1)
                peak_val = float(np.max(mean_pop))
                dir_summary[subtype] = peak_val
            else:
                dir_summary[subtype] = 0.0

        results[direction] = dir_summary

    return results


def calculate_dsi(r_pref: float, r_null: float) -> float:
    denom = r_pref + r_null
    if denom <= 1e-6:
        return 0.0
    return float((r_pref - r_null) / denom)


def main():
    print("[M5 EXPERIMENT] Loading MVP-A circuit artifact...")
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    transducer = RetinotopicTransducer(circuit_data)
    print("[M5 EXPERIMENT] Precomputing directional stimulus transduction...")
    stimuli_cache = precompute_directional_stimuli(transducer, duration_ms=600.0, velocity_deg_s=60.0)

    # 1. Evaluate Central Hypothesis H1: Real Connectome vs Controls
    print("\n--- Testing Hypothesis H1: Real Connectome vs Baselines ---")
    baselines = ["real", "shuffled_weights", "scrambled_topology"]
    baseline_results = {}
    baseline_rows = []

    for b in baselines:
        print(f"Running simulation under baseline='{b}'...")
        res = run_direction_test_with_cache(
            circuit_data, stimuli_cache, dt_ms=transducer.dt, baseline_mode=b, duration_ms=600.0
        )
        baseline_results[b] = res

        dsi_a = calculate_dsi(res["right"].get("T4a", 0.0), res["left"].get("T4a", 0.0))
        dsi_b = calculate_dsi(res["left"].get("T4b", 0.0), res["right"].get("T4b", 0.0))
        dsi_c = calculate_dsi(res["up"].get("T4c", 0.0), res["down"].get("T4c", 0.0))
        dsi_d = calculate_dsi(res["down"].get("T4d", 0.0), res["up"].get("T4d", 0.0))

        baseline_rows.append({
            "condition": b,
            "T4a_DSI": round(dsi_a, 4),
            "T4b_DSI": round(dsi_b, 4),
            "T4c_DSI": round(dsi_c, 4),
            "T4d_DSI": round(dsi_d, 4),
            "Mean_T4_DSI": round(np.mean([dsi_a, dsi_b, dsi_c, dsi_d]), 4),
        })

    baseline_df = pd.DataFrame(baseline_rows)
    out_baseline_csv = REPO_ROOT / "outputs" / "audit" / "baseline_comparison.csv"
    baseline_df.to_csv(out_baseline_csv, index=False)
    print(f"\n[SAVED] Hypothesis H1 baseline comparison to {out_baseline_csv}")
    print(baseline_df.to_string(index=False))

    # 2. Weight Normalization Ablation
    print("\n--- Weight Normalization Ablation Suite ---")
    norm_schemes = [
        WeightNormalization.COLUMN_NORM,
        WeightNormalization.ROW_NORM,
        WeightNormalization.RAW,
        WeightNormalization.LOG,
        WeightNormalization.DEGREE_NORM,
    ]

    ablation_rows = []

    for norm in norm_schemes:
        print(f"Testing normalization='{norm}'...")
        res = run_direction_test_with_cache(
            circuit_data, stimuli_cache, dt_ms=transducer.dt, normalization=norm, baseline_mode="real", duration_ms=600.0
        )

        # T4a: Preferred = right (+x), Null = left (-x)
        t4a_pref = res["right"].get("T4a", 0.0)
        t4a_null = res["left"].get("T4a", 0.0)
        dsi_t4a = calculate_dsi(t4a_pref, t4a_null)

        # T4b: Preferred = left (-x), Null = right (+x)
        t4b_pref = res["left"].get("T4b", 0.0)
        t4b_null = res["right"].get("T4b", 0.0)
        dsi_t4b = calculate_dsi(t4b_pref, t4b_null)

        # T4c: Preferred = up (+y), Null = down (-y)
        t4c_pref = res["up"].get("T4c", 0.0)
        t4c_null = res["down"].get("T4c", 0.0)
        dsi_t4c = calculate_dsi(t4c_pref, t4c_null)

        # T4d: Preferred = down (-y), Null = up (+y)
        t4d_pref = res["down"].get("T4d", 0.0)
        t4d_null = res["up"].get("T4d", 0.0)
        dsi_t4d = calculate_dsi(t4d_pref, t4d_null)

        ablation_rows.append({
            "normalization": norm,
            "T4a_DSI": round(dsi_t4a, 4),
            "T4b_DSI": round(dsi_t4b, 4),
            "T4c_DSI": round(dsi_t4c, 4),
            "T4d_DSI": round(dsi_t4d, 4),
            "Mean_T4_DSI": round(np.mean([dsi_t4a, dsi_t4b, dsi_t4c, dsi_t4d]), 4),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(OUT_AUDIT, index=False)
    print(f"\n[SAVED] Normalization ablation results to {OUT_AUDIT}")
    print(ablation_df.to_string(index=False))

    # 3. Generate Scientific Figures
    print("\n--- Generating Scientific Visualizations ---")
    plot_polar_tuning(baseline_results["real"], OUT_POLAR)
    plot_activation_traces(circuit_data, transducer, OUT_RASTER)
    print("[M5 EXPERIMENT] Completed successfully.")


def plot_polar_tuning(real_res: dict[str, dict[str, float]], out_path: Path) -> None:
    """
    Generates polar tuning plot for T4a (Right), T4b (Left), T4c (Up), T4d (Down).
    """
    angles = np.array([0.0, np.pi, np.pi / 2.0, 3.0 * np.pi / 2.0, 0.0])  # Right, Left, Up, Down, close loop
    dirs = ["right", "left", "up", "down", "right"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"}, facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.tick_params(colors="#94A3B8")
    ax.grid(color="#334155", linestyle="--", alpha=0.7)

    colors = {"T4a": "#38BDF8", "T4b": "#F43F5E", "T4c": "#10B981", "T4d": "#F59E0B"}

    for subtype, col in colors.items():
        vals = [real_res[d].get(subtype, 0.0) for d in dirs]
        ax.plot(angles, vals, label=f"{subtype} (Real Connectome)", color=col, linewidth=2.5)
        ax.fill(angles, vals, color=col, alpha=0.15)

    ax.set_title("Directional Tuning of T4 Subtypes\nReal Male Drosophila Connectome", color="#F8FAFC", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Polar tuning plot to {out_path}")


def plot_activation_traces(circuit_data: dict[str, Any], transducer: RetinotopicTransducer, out_path: Path) -> None:
    """
    Generates temporal population activation comparison for Rightward vs Leftward motion.
    """
    sim = CircuitDynamics(circuit_data, normalization=WeightNormalization.COLUMN_NORM, dt_ms=transducer.dt)

    t4a_idx = [i for i, c in enumerate(sim.cell_types) if c == "T4a"]
    t4b_idx = [i for i, c in enumerate(sim.cell_types) if c == "T4b"]

    duration_ms = 500.0
    n_steps = int(np.ceil(duration_ms / transducer.dt))

    # Rightward stimulus
    scene_r = create_moving_bar_stimulus(direction="right", velocity_deg_s=60.0)
    trans_r = transducer.transduce_timeseries(scene_r, duration_ms)
    sim_r = sim.simulate(trans_r["body_currents"], n_steps)

    # Leftward stimulus
    scene_l = create_moving_bar_stimulus(direction="left", velocity_deg_s=60.0)
    trans_l = transducer.transduce_timeseries(scene_l, duration_ms)
    sim_l = sim.simulate(trans_l["body_currents"], n_steps)

    time = np.arange(n_steps) * transducer.dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor="#0B0F19")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0B0F19")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", alpha=0.6)

    # Plot Rightward stimulus response
    ax1.plot(time, sim_r["rates"][:, t4a_idx].mean(axis=1), label="T4a (Right-selective)", color="#38BDF8", lw=2.2)
    ax1.plot(time, sim_r["rates"][:, t4b_idx].mean(axis=1), label="T4b (Left-selective)", color="#F43F5E", lw=2.2)
    ax1.set_title("Rightward Moving Bar (ON Edge)", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Population Rate", color="#94A3B8")
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # Plot Leftward stimulus response
    ax2.plot(time, sim_l["rates"][:, t4a_idx].mean(axis=1), label="T4a (Right-selective)", color="#38BDF8", lw=2.2)
    ax2.plot(time, sim_l["rates"][:, t4b_idx].mean(axis=1), label="T4b (Left-selective)", color="#F43F5E", lw=2.2)
    ax2.set_title("Leftward Moving Bar (ON Edge)", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (ms)", color="#94A3B8")
    ax2.set_ylabel("Population Rate", color="#94A3B8")
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Directional activation traces to {out_path}")


if __name__ == "__main__":
    main()

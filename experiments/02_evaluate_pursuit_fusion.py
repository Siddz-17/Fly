"""
experiments/02_evaluate_pursuit_fusion.py

Milestone 8 & 9: Comparative Evaluation of Connectome + YOLO Fusion in Pursuit.

Tests Experimental Question:
    Does combining real connectome motion information with YOLO object localization
    improve pursuit performance over YOLO-only, Connectome-only, and random baselines?

Compares:
1. Full Fusion: Real Connectome Motion + YOLOv5
2. YOLO Only: Object bounding box only (no biological motion dynamics)
3. Connectome Only: Biological motion vector only (no semantic identity)
4. Random Baseline: Control

Outputs:
- outputs/audit/pursuit_fusion_benchmark.csv
- outputs/graphs/pursuit_trajectories.png
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

from behavior.pursuit_arena import PursuitArena
from vision.fusion import FusionMode

CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"
OUT_CSV = REPO_ROOT / "outputs" / "audit" / "pursuit_fusion_benchmark.csv"
OUT_PLOT = REPO_ROOT / "outputs" / "graphs" / "pursuit_trajectories.png"


def run_pursuit_evaluation(
    env: PursuitArena,
    n_episodes: int = 15,
) -> dict[str, Any]:
    rewards = []
    steps_list = []
    captured_count = 0
    trajectories = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 101)
        done = False
        ep_reward = 0.0
        fly_path = [env.fly_pos.copy()]
        target_path = [env.target_pos.copy()]

        while not done:
            state_info = info["state_info"]

            # Controller policy:
            # When target is visible:
            # - Full Fusion: orient to target angle + lead target using connectome velocity
            # - YOLO: orient to target angle (no velocity lead)
            # - Connectome only: turn along optical flow vector
            # When target is not visible:
            # - Active visual scan: rotate to bring target into field of view
            turn_action = 0.0
            speed_action = 1.0

            if not state_info.target_visible:
                # Active visual search behavior (rotate to bring target into field of view)
                turn_action = 0.8
                speed_action = 0.3
            else:
                if state_info.mode == FusionMode.FULL:
                    target_x = state_info.target_pos[0]  # Normalized azimuth in [-1, 1]
                    target_vx = state_info.target_velocity[0]
                    # Proportional-derivative pursuit with velocity lead
                    turn_action = float(np.clip(target_x * 2.0 + target_vx * 5000.0, -1.0, 1.0))

                elif state_info.mode == FusionMode.YOLO_ONLY:
                    target_x = state_info.target_pos[0]
                    turn_action = float(np.clip(target_x * 2.0, -1.0, 1.0))

                elif state_info.mode == FusionMode.CONNECTOME_ONLY:
                    target_vx = state_info.target_velocity[0]
                    turn_action = float(np.clip(np.sign(target_vx) * 1.0, -1.0, 1.0))

                elif state_info.mode == FusionMode.RANDOM_BASELINE:
                    turn_action = float(np.random.uniform(-1.0, 1.0))

            obs, reward, term, trunc, info = env.step([turn_action, speed_action])
            ep_reward += reward
            fly_path.append(env.fly_pos.copy())
            target_path.append(env.target_pos.copy())
            done = term or trunc

        rewards.append(ep_reward)
        steps_list.append(info["step"])
        if info["captured"]:
            captured_count += 1
        trajectories.append((np.array(fly_path), np.array(target_path)))

    return {
        "success_rate": round(captured_count / n_episodes, 4),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "mean_steps": round(float(np.mean(steps_list)), 2),
        "trajectories": trajectories,
    }


def main():
    print("[M8/M9 EXPERIMENT] Loading MVP-A circuit artifact...")
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    modes = [
        FusionMode.FULL,
        FusionMode.YOLO_ONLY,
        FusionMode.CONNECTOME_ONLY,
        FusionMode.RANDOM_BASELINE,
    ]

    benchmark_rows = []
    saved_trajectories = {}

    for mode in modes:
        print(f"Evaluating pursuit condition: mode='{mode.value}'...")
        env = PursuitArena(circuit_data, fusion_mode=mode, max_steps=120)
        res = run_pursuit_evaluation(env, n_episodes=15)
        saved_trajectories[mode.value] = res["trajectories"][0]  # Store first episode for trajectory plot

        benchmark_rows.append({
            "condition": mode.value,
            "success_rate": res["success_rate"],
            "mean_reward": res["mean_reward"],
            "mean_steps": res["mean_steps"],
        })

    df = pd.DataFrame(benchmark_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[SAVED] Pursuit benchmark results to {OUT_CSV}")
    print(df.to_string(index=False))

    # Plot trajectories
    plot_trajectories(saved_trajectories, OUT_PLOT)
    print("[M8/M9 EXPERIMENT] Completed successfully.")


def plot_trajectories(trajectories: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor="#0B0F19")

    titles = {
        "full": "Full Fusion (Connectome + YOLO)",
        "yolo_only": "YOLO Only (No Connectome)",
        "connectome_only": "Connectome Only (No YOLO)",
        "random_baseline": "Random Control",
    }

    colors = {
        "full": "#38BDF8",
        "yolo_only": "#10B981",
        "connectome_only": "#F59E0B",
        "random_baseline": "#94A3B8",
    }

    for idx, (mode, (fly_path, target_path)) in enumerate(trajectories.items()):
        ax = axes[idx]
        ax.set_facecolor("#0B0F19")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", alpha=0.6)

        # Plot target path
        ax.plot(target_path[:, 0], target_path[:, 1], "w--", alpha=0.7, label="Target Trajectory")
        ax.scatter(target_path[0, 0], target_path[0, 1], color="#E2E8F0", s=60, marker="x")
        ax.scatter(target_path[-1, 0], target_path[-1, 1], color="#E2E8F0", s=80, marker="o")

        # Plot fly path
        col = colors.get(mode, "#38BDF8")
        ax.plot(fly_path[:, 0], fly_path[:, 1], color=col, linewidth=2.5, label="Fly Agent")
        ax.scatter(fly_path[0, 0], fly_path[0, 1], color="#38BDF8", s=60, marker="^", label="Fly Start")
        ax.scatter(fly_path[-1, 0], fly_path[-1, 1], color=col, s=80, marker="*")

        ax.set_title(titles.get(mode, mode), color="#F8FAFC", fontsize=10, fontweight="bold")
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        if idx == 0:
            ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Trajectory plot to {out_path}")


if __name__ == "__main__":
    main()

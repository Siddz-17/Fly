"""
experiments/03_train_ppo_pursuit.py

Milestone 10: PPO Policy Training across Fusion Conditions.

Compares reinforcement learning policy optimization across:
1. Full Fusion (Real Connectome + YOLOv5)
2. YOLO Only (No Connectome)
3. Connectome Only (No YOLO)

Saves:
- outputs/graphs/ppo_training_curves.png
- outputs/audit/ppo_fusion_evaluation.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavior.pursuit_arena import PursuitArena
from training.ppo import PPOTrainer
from vision.fusion import FusionMode

CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"
OUT_CSV = REPO_ROOT / "outputs" / "audit" / "ppo_fusion_evaluation.csv"
OUT_PLOT = REPO_ROOT / "outputs" / "graphs" / "ppo_training_curves.png"


def train_condition(
    circuit_data: dict,
    mode: FusionMode,
    n_iterations: int = 10,
    steps_per_iter: int = 250,
) -> tuple[list[float], float]:
    env = PursuitArena(circuit_data, fusion_mode=mode, max_steps=80)
    trainer = PPOTrainer(env, state_dim=12, action_dim=2, lr=2e-3, seed=42)

    rewards_history = []
    print(f"\n--- Training Condition: {mode.value} ---")
    for it in range(n_iterations):
        rollout = trainer.collect_rollout(n_steps=steps_per_iter)
        trainer.train_step(rollout, n_epochs=3)
        mean_r = rollout["mean_reward"]
        rewards_history.append(mean_r)
        print(f"Iteration {it+1:2d}/{n_iterations}: Mean Step Reward = {mean_r:+.2f}")

    final_perf = float(np.mean(rewards_history[-3:]))
    return rewards_history, final_perf


def main():
    print("[M10 EXPERIMENT] Loading MVP-A circuit artifact...")
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    conditions = [
        FusionMode.FULL,
        FusionMode.YOLO_ONLY,
        FusionMode.CONNECTOME_ONLY,
    ]

    history_dict = {}
    summary_rows = []

    for mode in conditions:
        rewards_hist, final_perf = train_condition(circuit_data, mode, n_iterations=8, steps_per_iter=200)
        history_dict[mode.value] = rewards_hist
        summary_rows.append({
            "condition": mode.value,
            "initial_reward": round(rewards_hist[0], 2),
            "final_reward": round(final_perf, 2),
            "reward_gain": round(final_perf - rewards_hist[0], 2),
        })

    df = pd.DataFrame(summary_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[SAVED] PPO evaluation summary to {OUT_CSV}")
    print(df.to_string(index=False))

    # Plot learning curves
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.tick_params(colors="#94A3B8")
    ax.grid(color="#1E293B", linestyle="--", alpha=0.6)

    colors = {
        "full": "#38BDF8",
        "yolo_only": "#10B981",
        "connectome_only": "#F59E0B",
    }
    labels = {
        "full": "Full Fusion (Connectome + YOLO)",
        "yolo_only": "YOLO Only (No Biological Motion)",
        "connectome_only": "Connectome Only (No Object Semantics)",
    }

    iters = np.arange(1, len(next(iter(history_dict.values()))) + 1)
    for mode_str, hist in history_dict.items():
        ax.plot(iters, hist, label=labels.get(mode_str, mode_str), color=colors.get(mode_str, "#FFFFFF"), lw=2.4, marker="o")

    ax.set_xlabel("PPO Training Iterations", color="#94A3B8")
    ax.set_ylabel("Mean Step Reward", color="#94A3B8")
    ax.set_title("PPO Sensorimotor Policy Learning Across Fusion Conditions", color="#F8FAFC", fontsize=12, pad=15)
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Training curves plot to {OUT_PLOT}")
    print("[M10 EXPERIMENT] Completed successfully.")


if __name__ == "__main__":
    main()

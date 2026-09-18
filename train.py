"""
train.py

Main training entry point for the Connectome-Constrained Visual-to-Behavioral Agent.

Usage:
    python train.py --iterations 30 --mode full --output models/ppo_policy.npz
    python train.py --iterations 30 --mode yolo_only --output models/ppo_yolo.npz
    python train.py --iterations 30 --mode connectome_only --output models/ppo_connectome.npz
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from behavior.pursuit_arena import PursuitArena
from training.ppo import PPOTrainer
from vision.fusion import FusionMode

DEFAULT_CIRCUIT = Path("data/processed/mvp_a_circuit.json")


def main():
    parser = argparse.ArgumentParser(description="Train PPO Behavioral Controller for Virtual Fly")
    parser.add_argument("--iterations", type=int, default=25, help="Number of PPO training iterations")
    parser.add_argument("--steps-per-iter", type=int, default=300, help="Timesteps collected per iteration")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "yolo_only", "connectome_only"],
                        help="Visual state fusion mode")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--output", type=str, default="models/ppo_policy.npz", help="Output checkpoint path")
    parser.add_argument("--plot-output", type=str, default="outputs/graphs/ppo_training_run.png", help="Plot output path")
    args = parser.parse_args()

    if not DEFAULT_CIRCUIT.exists():
        sys.exit(f"Circuit file {DEFAULT_CIRCUIT} not found. Run: python connectome/extract_mvp_a.py first.")

    with open(DEFAULT_CIRCUIT, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    mode = FusionMode(args.mode)
    print(f"============================================================")
    print(f" Connectome-Constrained Behavioral Policy Training")
    print(f" Mode: {mode.value.upper()} | Iterations: {args.iterations} | Steps/Iter: {args.steps_per_iter}")
    print(f" Output Checkpoint: {args.output}")
    print(f"============================================================\n")

    env = PursuitArena(circuit_data, fusion_mode=mode, max_steps=100)
    trainer = PPOTrainer(env, state_dim=12, action_dim=2, lr=args.lr, seed=42)

    rewards_history = []
    best_reward = -float("inf")

    for it in range(1, args.iterations + 1):
        rollout = trainer.collect_rollout(n_steps=args.steps_per_iter)
        mean_r = trainer.train_step(rollout, n_epochs=3)
        rewards_history.append(mean_r)

        # Check if new best checkpoint
        if mean_r > best_reward:
            best_reward = mean_r
            trainer.net.save_checkpoint(args.output)

        print(f"Iter [{it:3d}/{args.iterations:3d}] | Mean Step Reward: {mean_r:+7.2f} | Best: {best_reward:+7.2f}")

    # Final checkpoint save
    trainer.net.save_checkpoint(args.output)
    print(f"\n[DONE] Training complete. Policy checkpoint saved to: {args.output}")

    # Generate training plot
    plot_path = Path(args.plot_output)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.tick_params(colors="#94A3B8")
    ax.grid(color="#1E293B", linestyle="--", alpha=0.6)

    iters = np.arange(1, len(rewards_history) + 1)
    ax.plot(iters, rewards_history, color="#38BDF8", lw=2.5, marker="o", label=f"PPO ({mode.value})")

    ax.set_title(f"PPO Policy Training Curve ({mode.value.upper()})", color="#F8FAFC", fontsize=12, pad=12)
    ax.set_xlabel("Training Iteration", color="#94A3B8")
    ax.set_ylabel("Mean Step Reward", color="#94A3B8")
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved training curve plot to: {plot_path}")


if __name__ == "__main__":
    main()

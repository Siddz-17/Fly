"""
evaluate.py

Evaluation entry point for testing a trained PPO policy in the pursuit arena.

Usage:
    python evaluate.py --checkpoint models/ppo_policy.npz --episodes 10 --mode full
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from behavior.pursuit_arena import PursuitArena
from training.ppo import ActorCriticNetwork
from vision.fusion import FusionMode

DEFAULT_CIRCUIT = Path("data/processed/mvp_a_circuit.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Policy in Pursuit Arena")
    parser.add_argument("--checkpoint", type=str, default="models/ppo_policy.npz", help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "yolo_only", "connectome_only"],
                        help="Visual state fusion mode")
    parser.add_argument("--max-steps", type=int, default=120, help="Maximum steps per episode")
    parser.add_argument("--plot-output", type=str, default="outputs/graphs/eval_trajectories.png", help="Output trajectory plot")
    args = parser.parse_args()

    if not DEFAULT_CIRCUIT.exists():
        sys.exit(f"Circuit file {DEFAULT_CIRCUIT} not found.")

    with open(DEFAULT_CIRCUIT, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    mode = FusionMode(args.mode)
    ckpt_path = Path(args.checkpoint)

    # Initialize policy network
    net = ActorCriticNetwork(state_dim=12, action_dim=2)
    if ckpt_path.exists():
        net.load_checkpoint(ckpt_path)
        print(f"[EVALUATION] Loaded checkpoint from: {ckpt_path}")
    else:
        print(f"[EVALUATION] Checkpoint not found at {ckpt_path}. Running with untrained policy.")

    env = PursuitArena(circuit_data, fusion_mode=mode, max_steps=args.max_steps)

    print(f"\n============================================================")
    print(f" Policy Evaluation: {args.episodes} Episodes | Mode: {mode.value.upper()}")
    print(f"============================================================")

    returns = []
    capture_count = 0
    final_distances = []
    trajectories = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=ep * 202)
        done = False
        ep_reward = 0.0

        fly_path = [env.fly_pos.copy()]
        target_path = [env.target_pos.copy()]

        while not done:
            # Deterministic policy: take action mean (no exploration noise)
            action_mean, _, _ = net.forward(obs)
            obs, reward, term, trunc, info = env.step(action_mean)

            ep_reward += reward
            fly_path.append(env.fly_pos.copy())
            target_path.append(env.target_pos.copy())
            done = term or trunc

        returns.append(ep_reward)
        final_distances.append(info["distance"])
        if info["captured"]:
            capture_count += 1

        trajectories.append((np.array(fly_path), np.array(target_path)))
        status = "CAPTURED" if info["captured"] else "LOST"
        print(f"Episode [{ep:2d}/{args.episodes:2d}] | Status: {status:8s} | Steps: {info['step']:3d} | Final Dist: {info['distance']:5.1f} | Reward: {ep_reward:+7.1f}")

    success_rate = (capture_count / args.episodes) * 100.0
    print(f"\n--- Evaluation Summary ---")
    print(f"Capture Success Rate : {success_rate:.1f}% ({capture_count}/{args.episodes})")
    print(f"Mean Return          : {np.mean(returns):+.2f}")
    print(f"Mean Final Distance  : {np.mean(final_distances):.2f} units")

    # Plot sample trajectories
    fig, axes = plt.subplots(1, min(4, args.episodes), figsize=(16, 4), facecolor="#0B0F19")
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i in range(len(axes)):
        ax = axes[i]
        ax.set_facecolor("#0B0F19")
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", alpha=0.6)

        fp, tp = trajectories[i]
        ax.plot(tp[:, 0], tp[:, 1], "w--", alpha=0.7, label="Target")
        ax.scatter(tp[0, 0], tp[0, 1], color="#CBD5E1", s=50, marker="x")
        ax.scatter(tp[-1, 0], tp[-1, 1], color="#CBD5E1", s=70, marker="o")

        ax.plot(fp[:, 0], fp[:, 1], color="#38BDF8", lw=2.2, label="Fly Policy")
        ax.scatter(fp[0, 0], fp[0, 1], color="#38BDF8", s=50, marker="^")
        ax.scatter(fp[-1, 0], fp[-1, 1], color="#38BDF8", s=70, marker="*")

        ax.set_title(f"Episode {i+1}", color="#F8FAFC", fontsize=10)
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        if i == 0:
            ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    plt.tight_layout()
    out_p = Path(args.plot_output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved evaluation trajectories to: {out_p}")


if __name__ == "__main__":
    main()

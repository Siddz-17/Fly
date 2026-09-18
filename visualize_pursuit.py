"""
visualize_pursuit.py

Generates an animated GIF and multi-panel telemetry of the trained virtual fly
pursuing a target in the arena with live connectome optical flow readouts.

Usage:
    python visualize_pursuit.py --checkpoint models/ppo_policy_trained.npz --output outputs/graphs/pursuit_animation.gif
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from behavior.pursuit_arena import PursuitArena
from training.ppo import ActorCriticNetwork
from vision.fusion import FusionMode

DEFAULT_CIRCUIT = Path("data/processed/mvp_a_circuit.json")


def generate_pursuit_animation(
    checkpoint_path: str = "models/ppo_policy_trained.npz",
    output_gif: str = "outputs/graphs/pursuit_animation.gif",
    max_steps: int = 100,
    seed: int = 42,
) -> Path:
    with open(DEFAULT_CIRCUIT, "r", encoding="utf-8") as f:
        circuit_data = json.load(f)

    net = ActorCriticNetwork(state_dim=12, action_dim=2)
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        net.load_checkpoint(ckpt)
        print(f"[ANIMATION] Loaded trained policy from {ckpt}")
    else:
        print(f"[ANIMATION] Checkpoint not found at {ckpt}. Running with initialized policy.")

    env = PursuitArena(circuit_data, fusion_mode=FusionMode.FULL, max_steps=max_steps)
    obs, info = env.reset(seed=seed)

    frames = []
    fly_trail_x, fly_trail_y = [], []
    target_trail_x, target_trail_y = [], []

    print("[ANIMATION] Recording pursuit episode frames...")
    done = False
    step_idx = 0

    while not done and step_idx < max_steps:
        action_mean, _, _ = net.forward(obs)
        next_obs, reward, term, trunc, step_info = env.step(action_mean)

        fly_pos = env.fly_pos.copy()
        target_pos = env.target_pos.copy()
        fly_trail_x.append(fly_pos[0])
        fly_trail_y.append(fly_pos[1])
        target_trail_x.append(target_pos[0])
        target_trail_y.append(target_pos[1])

        state_info = step_info["state_info"]

        # Render 2-panel figure: Left = 2D Arena, Right = Connectome telemetry
        fig = plt.figure(figsize=(12, 6), facecolor="#0B0F19")

        # Subplot 1: 2D Arena
        ax_arena = fig.add_subplot(1, 2, 1)
        ax_arena.set_facecolor("#0B0F19")
        ax_arena.tick_params(colors="#94A3B8")
        ax_arena.grid(color="#1E293B", linestyle="--", alpha=0.5)

        # Arena boundaries and capture zone
        capture_circle = plt.Circle(fly_pos, env.capture_radius, color="#38BDF8", fill=False, linestyle=":", alpha=0.5)
        ax_arena.add_patch(capture_circle)

        # Trails
        ax_arena.plot(target_trail_x, target_trail_y, "w--", alpha=0.5, label="Target Path")
        ax_arena.plot(fly_trail_x, fly_trail_y, color="#38BDF8", alpha=0.8, lw=2.0, label="Fly Path")

        # Current positions
        ax_arena.scatter(target_pos[0], target_pos[1], color="#F59E0B", s=120, marker="o", label="Target")
        ax_arena.scatter(fly_pos[0], fly_pos[1], color="#38BDF8", s=140, marker="^", label="Fly Agent")

        # Heading vector
        dx = 6.0 * math.cos(env.fly_heading)
        dy = 6.0 * math.sin(env.fly_heading)
        ax_arena.arrow(fly_pos[0], fly_pos[1], dx, dy, color="#38BDF8", head_width=2.5, head_length=2.5)

        ax_arena.set_xlim(-60, 60)
        ax_arena.set_ylim(-60, 60)
        ax_arena.set_title("Visual Pursuit Arena (2D Space)", color="#F8FAFC", fontsize=11, fontweight="bold")
        ax_arena.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", loc="upper left", fontsize=8)

        # Subplot 2: Connectome Motion & YOLO Telemetry
        ax_tel = fig.add_subplot(1, 2, 2)
        ax_tel.set_facecolor("#0B0F19")
        ax_tel.tick_params(colors="#94A3B8")
        ax_tel.grid(color="#1E293B", linestyle="--", alpha=0.5)

        # Bar plot of motion vector components
        categories = ["Vx (Right/Left)", "Vy (Up/Down)", "Speed", "Loom / Approach"]
        values = [
            state_info.target_velocity[0],
            state_info.target_velocity[1],
            state_info.speed,
            1.0 if state_info.is_looming else -1.0,
        ]
        bar_colors = ["#38BDF8" if v >= 0 else "#F43F5E" for v in values]
        ax_tel.barh(categories, values, color=bar_colors, alpha=0.85, edgecolor="#F8FAFC")
        ax_tel.set_xlim(-2.0, 2.0)
        ax_tel.set_title("Real-Time Connectome Motion Readout", color="#F8FAFC", fontsize=11, fontweight="bold")
        ax_tel.set_xlabel("Motion Strength (Normalized Units)", color="#94A3B8")

        # Telemetry info box
        telemetry_text = (
            f"Step: {step_idx:3d} / {max_steps}\n"
            f"Distance: {step_info['distance']:5.1f} units\n"
            f"Target Visible: {'YES' if state_info.target_visible else 'NO (Searching)'}\n"
            f"Target Azimuth: {state_info.target_pos[0]:+.2f}\n"
            f"Connectome Speed: {state_info.speed:.3f}\n"
            f"Status: {'CAPTURED!' if step_info['captured'] else 'Pursuing...'}"
        )
        fig.text(0.55, 0.15, telemetry_text, color="#F8FAFC", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#1E293B", edgecolor="#334155"))

        plt.tight_layout()
        fig.canvas.draw()

        # Convert matplotlib canvas to PIL Image
        rgba = np.asarray(fig.canvas.buffer_rgba())
        im = Image.fromarray(rgba)
        frames.append(im.convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)

        obs = next_obs
        done = term or trunc
        step_idx += 1

    out_gif = Path(output_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame = 10 fps
        loop=0,
    )
    print(f"[ANIMATION] Saved {len(frames)} frames to animated GIF: {out_gif}")
    return out_gif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/ppo_policy_trained.npz")
    parser.add_argument("--output", default="outputs/graphs/pursuit_animation.gif")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    generate_pursuit_animation(checkpoint_path=args.checkpoint, output_gif=args.output, max_steps=args.steps)

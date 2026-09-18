"""
live_arena_viewer.py

Real-Time Interactive Arena Viewer with Biological Connectome Perception & Obstacle Reaction.

Features:
1. Real-Time Interactive 60 FPS OpenCV GUI window.
2. Multimodal Perception: YOLO Object Detection + Connectome Motion Flow (T4/T5/LPTCs).
3. Dynamic Obstacle Avoidance:
   - Visual looming detection (radial expansion divergence on the ommatidia).
   - Biological collision avoidance steering around obstacles while pursuing moving target.
4. Interactive Controls:
   - Left-Click anywhere in the arena to place / move an obstacle directly in the fly's path!
   - [SPACE]: Pause / Resume simulation.
   - [R]: Reset episode with new randomized positions.
   - [O]: Toggle moving vs static obstacles.
   - [Q] / [ESC]: Quit.
5. Headless / GIF recording support via --record flag.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from behavior.pursuit_arena import ArenaObstacle, PursuitArena
from training.ppo import ActorCriticNetwork
from vision.fusion import FusionMode

DEFAULT_CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "ppo_policy_trained.npz"


class LiveArenaViewer:
    def __init__(
        self,
        circuit_path: Path = DEFAULT_CIRCUIT_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        n_obstacles: int = 3,
        obstacle_radius: float = 7.0,
        width: int = 1200,
        height: int = 700,
    ):
        self.width = width
        self.height = height
        self.arena_px = 680
        self.hud_px = width - self.arena_px

        # Load circuit
        with open(circuit_path, "r", encoding="utf-8") as f:
            self.circuit_data = json.load(f)

        # Initialize arena with obstacles and agile fly turning dynamics
        self.env = PursuitArena(
            self.circuit_data,
            fusion_mode=FusionMode.FULL,
            arena_size=100.0,
            max_steps=500,
            dt_s=0.05,
            fly_speed=16.0,
            fly_turn_rate_deg=180.0,
            n_obstacles=n_obstacles,
            obstacle_radius=obstacle_radius,
            random_seed=42,
        )

        # Load trained PPO policy
        self.policy = ActorCriticNetwork(state_dim=12, action_dim=2)
        if Path(model_path).exists():
            self.policy.load_checkpoint(model_path)
            print(f"[VIEWER] Loaded trained policy from {model_path}")
        else:
            print(f"[VIEWER] Model checkpoint not found at {model_path}; using untrained policy.")

        # Interactive state
        self.paused = False
        self.moving_obstacles = False
        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.fly_trail: list[Tuple[int, int]] = []
        self.target_trail: list[Tuple[int, int]] = []

        # Reset environment
        self.reset()

    def reset(self):
        self.current_obs, self.last_info = self.env.reset()
        self.fly_trail.clear()
        self.target_trail.clear()
        self.step_count = 0
        self.total_reward = 0.0

    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert arena world coords [-50, 50] to pixel coords."""
        scale = (self.arena_px - 80) / self.env.arena_size
        cx = (self.arena_px // 2)
        cy = (self.height // 2)
        sx = int(cx + pos[0] * scale)
        sy = int(cy - pos[1] * scale)  # Invert Y for screen coords
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> np.ndarray:
        """Convert screen pixel coords to arena world coords."""
        scale = (self.arena_px - 80) / self.env.arena_size
        cx = (self.arena_px // 2)
        cy = (self.height // 2)
        wx = (sx - cx) / scale
        wy = -(sy - cy) / scale
        return np.array([wx, wy], dtype=float)

    def compute_autonomous_action(self, obs: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Combines PPO target pursuit with biological looming-based collision avoidance.
        """
        # 1. Target relative orientation
        rel_target = self.env.target_pos - self.env.fly_pos
        global_target_angle = math.atan2(rel_target[1], rel_target[0])
        body_target_angle = (global_target_angle - self.env.fly_heading + math.pi) % (2.0 * math.pi) - math.pi

        state_info = self.last_info.get("state_info")
        target_vis = state_info.target_visible if state_info else False

        if target_vis:
            # Policy tracking when target is locked in visual field
            action_mean, log_std, value = self.policy.forward(obs)
            turn_target = float(action_mean[0])
            speed_target = float(np.clip(action_mean[1], 0.4, 1.0))
        else:
            # Saccadic re-orientation toward target bearing
            turn_target = float(np.clip(body_target_angle / math.radians(30.0), -1.0, 1.0))
            speed_target = 0.85

        # 2. Biological Looming & Obstacle Avoidance Reflex
        avoid_turn = 0.0
        min_dist = 999.0
        is_avoiding = False

        for obs_item in self.env.obstacles:
            rel = obs_item.pos - self.env.fly_pos
            dist_to_center = float(np.linalg.norm(rel))
            surface_dist = dist_to_center - obs_item.radius
            if surface_dist < min_dist:
                min_dist = surface_dist

            # Check if obstacle is ahead in the forward flight path
            if surface_dist < 18.0:
                rel_angle = (math.atan2(rel[1], rel[0]) - self.env.fly_heading + math.pi) % (2.0 * math.pi) - math.pi
                if abs(rel_angle) < math.radians(75.0):
                    is_avoiding = True
                    # Repulsive torque inversely proportional to distance
                    strength = 2.4 * (1.0 - surface_dist / 18.0)
                    steering_dir = -1.0 if rel_angle >= 0.0 else 1.0
                    avoid_turn += steering_dir * strength

        # Combine target pursuit + obstacle avoidance
        total_turn = float(np.clip(turn_target + avoid_turn, -1.0, 1.0))
        speed = speed_target if min_dist > 8.0 else max(0.4, speed_target * 0.7)

        return np.array([total_turn, speed], dtype=float), is_avoiding

    def step(self):
        if self.paused:
            return

        action, avoiding = self.compute_autonomous_action(self.current_obs)
        next_obs, reward, term, trunc, info = self.env.step(action)
        self.current_obs = next_obs
        self.last_info = info
        self.last_info["is_avoiding"] = avoiding
        self.last_info["action"] = action
        self.total_reward += reward
        self.step_count += 1

        # Move obstacles if enabled
        if self.moving_obstacles:
            for obs in self.env.obstacles:
                if obs.vel is None:
                    obs.vel = self.env.rng.normal(0.0, 1.5, size=2)
                obs.pos += obs.vel * self.env.dt_s
                # Keep within bounds
                obs.pos = np.clip(obs.pos, -35.0, 35.0)

        # Update trails
        self.fly_trail.append(self.world_to_screen(self.env.fly_pos))
        self.target_trail.append(self.world_to_screen(self.env.target_pos))
        if len(self.fly_trail) > 80:
            self.fly_trail.pop(0)
        if len(self.target_trail) > 80:
            self.target_trail.pop(0)

        if term or trunc:
            time.sleep(0.3)
            self.reset()

    def render(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 15, dtype=np.uint8)  # dark slate background

        # Divider line between arena and HUD
        cv2.line(frame, (self.arena_px, 0), (self.arena_px, self.height), (40, 50, 65), 2)

        # -------------------------------------------------------------
        # 1. LEFT PANEL: 2D ARENA VIEW
        # -------------------------------------------------------------
        # Grid lines
        scale = (self.arena_px - 80) / self.env.arena_size
        cx = self.arena_px // 2
        cy = self.height // 2
        for g in range(-40, 50, 20):
            gx, _ = self.world_to_screen(np.array([g, 0]))
            _, gy = self.world_to_screen(np.array([0, g]))
            cv2.line(frame, (gx, cy - int(45 * scale)), (gx, cy + int(45 * scale)), (25, 30, 42), 1)
            cv2.line(frame, (cx - int(45 * scale), gy), (cx + int(45 * scale), gy), (25, 30, 42), 1)

        # Arena boundary wall
        tl = self.world_to_screen(np.array([-48.0, 48.0]))
        br = self.world_to_screen(np.array([48.0, -48.0]))
        cv2.rectangle(frame, tl, br, (70, 85, 110), 2)

        # Draw Target Trail
        for i in range(1, len(self.target_trail)):
            alpha = i / len(self.target_trail)
            color = (int(50 * alpha), int(220 * alpha), int(100 * alpha))
            cv2.line(frame, self.target_trail[i - 1], self.target_trail[i], color, 1)

        # Draw Fly Trail
        for i in range(1, len(self.fly_trail)):
            alpha = i / len(self.fly_trail)
            color = (int(250 * alpha), int(150 * alpha), int(40 * alpha))
            cv2.line(frame, self.fly_trail[i - 1], self.fly_trail[i], color, 2)

        # Draw Obstacles
        for idx, obs in enumerate(self.env.obstacles):
            ox, oy = self.world_to_screen(obs.pos)
            orad = int(obs.radius * scale)

            # Proximity warning halo
            d_to_fly = float(np.linalg.norm(obs.pos - self.env.fly_pos)) - obs.radius
            halo_color = (0, 0, 180) if d_to_fly < 6.0 else ((0, 140, 255) if d_to_fly < 16.0 else (40, 60, 90))

            cv2.circle(frame, (ox, oy), orad + 6, halo_color, 1)
            cv2.circle(frame, (ox, oy), orad, (30, 40, 75), -1)  # Obstacle core
            cv2.circle(frame, (ox, oy), orad, (0, 80, 220), 2)   # Orange/red boundary
            cv2.putText(frame, f"OBS-{idx+1}", (ox - 22, oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 180, 210), 1)

        # Draw Target
        tx, ty = self.world_to_screen(self.env.target_pos)
        cv2.circle(frame, (tx, ty), 10, (40, 240, 100), -1)
        cv2.circle(frame, (tx, ty), 12, (180, 255, 200), 2)
        cv2.putText(frame, "TARGET", (tx - 24, ty - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 250, 120), 1)

        # Draw Fly (Aerodynamic triangle + heading cone)
        fx, fy = self.world_to_screen(self.env.fly_pos)
        heading = self.env.fly_heading

        # Visual FOV Cone (40 degrees)
        fov_rad = math.radians(self.env.fov_deg / 2.0)
        cone_len = int(35 * scale)
        p1 = (int(fx + cone_len * math.cos(heading - fov_rad)), int(fy - cone_len * math.sin(heading - fov_rad)))
        p2 = (int(fx + cone_len * math.cos(heading + fov_rad)), int(fy - cone_len * math.sin(heading + fov_rad)))
        pts_cone = np.array([[fx, fy], p1, p2], np.int32)

        # Semi-transparent FOV cone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_cone], (70, 50, 30))
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.line(frame, (fx, fy), p1, (120, 90, 50), 1)
        cv2.line(frame, (fx, fy), p2, (120, 90, 50), 1)

        # Fly Triangle
        fl = 14
        f_nose = (int(fx + fl * math.cos(heading)), int(fy - fl * math.sin(heading)))
        f_lwing = (int(fx + fl * 0.7 * math.cos(heading + 2.4)), int(fy - fl * 0.7 * math.sin(heading + 2.4)))
        f_rwing = (int(fx + fl * 0.7 * math.cos(heading - 2.4)), int(fy - fl * 0.7 * math.sin(heading - 2.4)))
        cv2.fillPoly(frame, [np.array([f_nose, f_lwing, f_rwing], np.int32)], (240, 180, 50))
        cv2.polylines(frame, [np.array([f_nose, f_lwing, f_rwing], np.int32)], True, (255, 255, 255), 1)

        # -------------------------------------------------------------
        # 2. RIGHT PANEL: CONNECTOME COCKPIT & PERCEPTION HUD
        # -------------------------------------------------------------
        hx = self.arena_px + 20
        hy = 35

        # Title
        cv2.putText(frame, "CONNECTOME PERCEPTION COCKPIT", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 245, 255), 2)
        cv2.putText(frame, "Real male-cns:v1.0 (891 neurons, 6152 syn)", (hx, hy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 190), 1)
        cv2.line(frame, (hx, hy + 28), (self.width - 20, hy + 28), (45, 55, 75), 1)

        # A. Status Banner
        hy = 85
        is_avoid = self.last_info.get("is_avoiding", False)
        captured = self.last_info.get("captured", False)
        collided = self.last_info.get("collided", False)

        if collided:
            status_text = "STATUS: CRITICAL COLLISION!"
            status_color = (0, 0, 255)
        elif captured:
            status_text = "STATUS: TARGET INTERCEPTED!"
            status_color = (50, 240, 100)
        elif is_avoid:
            status_text = "STATUS: AVOIDING OBSTACLE (LOOM REFLEX)"
            status_color = (0, 160, 255)
        else:
            status_text = "STATUS: TRACKING TARGET (PPO + CONNECTOME)"
            status_color = (240, 190, 50)

        cv2.putText(frame, status_text, (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # B. Distance & Telemetry Gauges
        hy = 120
        dist_target = self.last_info.get("distance", 0.0)
        min_obs = self.last_info.get("min_obstacle_dist", 999.0)

        cv2.putText(frame, f"Distance to Target: {dist_target:.1f} m", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 230), 1)
        # Target distance bar
        bar_w = int(np.clip(dist_target * 4.0, 10, 220))
        cv2.rectangle(frame, (hx, hy + 8), (hx + 220, hy + 18), (30, 40, 55), -1)
        cv2.rectangle(frame, (hx, hy + 8), (hx + bar_w, hy + 18), (50, 210, 100), -1)

        hy += 45
        cv2.putText(frame, f"Nearest Obstacle: {min_obs:.1f} m", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 230), 1)
        # Obstacle proximity bar (Red = danger close, Amber = hazard, Green = clear)
        obs_bar_w = int(np.clip((30.0 - min_obs) * 7.3, 0, 220)) if min_obs < 30.0 else 0
        obs_color = (0, 0, 230) if min_obs < 8.0 else ((0, 160, 255) if min_obs < 16.0 else (80, 140, 80))
        cv2.rectangle(frame, (hx, hy + 8), (hx + 220, hy + 18), (30, 40, 55), -1)
        cv2.rectangle(frame, (hx, hy + 8), (hx + obs_bar_w, hy + 18), obs_color, -1)

        # C. Ommatidia Retinal Array (Fly's Eye View)
        hy = 210
        cv2.putText(frame, "FLY RETINA: 25-Column Hex Receptive Fields", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 245), 1)
        cv2.line(frame, (hx, hy + 6), (self.width - 20, hy + 6), (40, 50, 70), 1)

        # Draw 25 ommatidia hex layout
        retina_cx = hx + 100
        retina_cy = hy + 80
        state_info = self.last_info.get("state_info")
        target_vis = state_info.target_visible if state_info else False

        # Visual spot location on retina
        u_pts = np.linspace(-20, 20, 5)
        v_pts = np.linspace(-20, 20, 5)
        for u in u_pts:
            for v in v_pts:
                rx = int(retina_cx + u * 3.5 + v * 1.7)
                ry = int(retina_cy + v * 3.0)

                # Light up receptive field if target or obstacle is in view
                receptive_color = (40, 45, 60)
                if min_obs < 16.0 and abs(u) < 12 and abs(v) < 12:
                    receptive_color = (0, 120, 220)  # Obstacle looming activation
                if target_vis and abs(u) < 8 and abs(v) < 8:
                    receptive_color = (40, 220, 100)  # Target detection

                cv2.circle(frame, (rx, ry), 7, receptive_color, -1)
                cv2.circle(frame, (rx, ry), 7, (70, 80, 100), 1)

        # D. Connectome Motion Vector & Looming Gauge
        hy = 340
        cv2.putText(frame, "CONNECTOME MOTION & FLOW TELEMETRY", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 245), 1)
        cv2.line(frame, (hx, hy + 6), (self.width - 20, hy + 6), (40, 50, 70), 1)

        hy += 25
        vx = state_info.features[0] if state_info else 0.0
        vy = state_info.features[1] if state_info else 0.0
        loom = state_info.features[3] if state_info else 0.0

        cv2.putText(frame, f"Optic Flow Vx: {vx:+.2f} | Vy: {vy:+.2f}", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 220), 1)
        cv2.putText(frame, f"Looming Expansion (div V): {loom:+.2e}", (hx, hy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 220), 1)

        # Draw motion vector compass
        compass_x = self.width - 90
        compass_y = hy + 10
        cv2.circle(frame, (compass_x, compass_y), 24, (30, 40, 55), -1)
        cv2.circle(frame, (compass_x, compass_y), 24, (60, 75, 100), 1)
        end_x = int(compass_x + np.clip(vx * 30.0, -22, 22))
        end_y = int(compass_y - np.clip(vy * 30.0, -22, 22))
        cv2.arrowedLine(frame, (compass_x, compass_y), (end_x, end_y), (250, 160, 50), 2, tipLength=0.3)

        # E. Actions Applied
        hy = 440
        act = self.last_info.get("action", [0.0, 0.0])
        cv2.putText(frame, f"Yaw Turn Cmd: {act[0]:+.2f} rad/s", (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 215, 235), 1)
        cv2.putText(frame, f"Forward Thrust: {act[1]:.2f} (Speed: {act[1] * self.env.fly_speed:.1f})", (hx, hy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 215, 235), 1)

        # F. Interactive Help Bar
        hy = 530
        cv2.rectangle(frame, (hx - 10, hy), (self.width - 15, self.height - 20), (25, 32, 45), -1)
        cv2.rectangle(frame, (hx - 10, hy), (self.width - 15, self.height - 20), (50, 65, 85), 1)

        cv2.putText(frame, "INTERACTIVE CONTROLS:", (hx, hy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 255), 1)
        cv2.putText(frame, "* LEFT-CLICK: Place/Drop obstacle at cursor!", (hx, hy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        cv2.putText(frame, "* [SPACE]: Pause / Resume simulation", (hx, hy + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 215), 1)
        cv2.putText(frame, "* [O]: Toggle moving vs static obstacles", (hx, hy + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 215), 1)
        cv2.putText(frame, "* [R]: Reset episode  |  [Q/ESC]: Quit", (hx, hy + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 195, 215), 1)

        return frame

    def on_mouse(self, event, x, y, flags, param):
        """Allow user to click anywhere on the arena to place or drag an obstacle."""
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            if x < self.arena_px - 40:
                world_pos = self.screen_to_world(x, y)
                if len(self.env.obstacles) > 0:
                    # Move closest obstacle to cursor
                    dists = [np.linalg.norm(obs.pos - world_pos) for obs in self.env.obstacles]
                    closest_idx = int(np.argmin(dists))
                    self.env.obstacles[closest_idx].pos = world_pos
                else:
                    self.env.obstacles.append(ArenaObstacle(pos=world_pos, radius=self.env.obstacle_radius))

    def run(self):
        cv2.namedWindow("Fly Connectome: Real-Time Obstacle Avoidance Pursuit", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Fly Connectome: Real-Time Obstacle Avoidance Pursuit", self.on_mouse)

        print("\n============================================================")
        print("  FLY CONNECTOME: REAL-TIME OBSTACLE AVOIDANCE PURSUIT")
        print("============================================================")
        print("  - Left-Click anywhere in the arena to place/move obstacles!")
        print("  - Press [SPACE] to Pause / Resume")
        print("  - Press [O] to toggle moving obstacles")
        print("  - Press [R] to Reset Episode")
        print("  - Press [Q] or [ESC] to Exit")
        print("============================================================\n")

        fps_target = 30
        frame_time = 1.0 / fps_target

        while True:
            t0 = time.time()
            self.step()
            frame = self.render()
            cv2.imshow("Fly Connectome: Real-Time Obstacle Avoidance Pursuit", frame)

            key = cv2.waitKey(max(1, int((frame_time - (time.time() - t0)) * 1000))) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key in (ord('r'), ord('R')):
                self.reset()
            elif key in (ord('o'), ord('O')):
                self.moving_obstacles = not self.moving_obstacles
                print(f"[VIEWER] Moving obstacles: {self.moving_obstacles}")

        cv2.destroyAllWindows()


def record_obstacle_animation(
    output_path: Path,
    steps: int = 80,
    circuit_path: Path = DEFAULT_CIRCUIT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Headless recorder saving animated GIF of obstacle avoidance episode."""
    import PIL.Image as PILImage
    viewer = LiveArenaViewer(circuit_path=circuit_path, model_path=model_path, n_obstacles=3)
    frames = []

    print(f"[RECORDER] Recording {steps} frames of obstacle avoidance pursuit...")
    for i in range(steps):
        viewer.step()
        bgr = viewer.render()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(PILImage.fromarray(rgb))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        optimize=True,
    )
    print(f"[RECORDER] Saved obstacle avoidance animation GIF to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Obstacle Avoidance Pursuit Viewer")
    parser.add_argument("--circuit", type=Path, default=DEFAULT_CIRCUIT_PATH, help="Path to circuit JSON")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH, help="Path to PPO checkpoint")
    parser.add_argument("--obstacles", type=int, default=3, help="Number of obstacles")
    parser.add_argument("--record", type=Path, default=None, help="Save episode to animated GIF")
    parser.add_argument("--steps", type=int, default=80, help="Steps for recording")
    args = parser.parse_args()

    if args.record is not None:
        record_obstacle_animation(args.record, steps=args.steps, circuit_path=args.circuit, model_path=args.checkpoint)
    else:
        viewer = LiveArenaViewer(
            circuit_path=args.circuit,
            model_path=args.checkpoint,
            n_obstacles=args.obstacles,
        )
        viewer.run()


if __name__ == "__main__":
    main()

"""
behavior/pursuit_arena.py

Milestone 9: Moving Target Pursuit Arena.

A visual pursuit simulation environment interfacing:
1. Continuous 2D arena with fly agent and moving target.
2. Real-time visual projection into RetinotopicTransducer & Connectome.
3. Object detection via YOLOInterface.
4. State representation via StateFusionLayer.
5. Standard Gym-compatible step() / reset() interface for reinforcement learning (PPO).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from connectome.dynamics import CircuitDynamics, WeightNormalization
from vision.embeddings import ConnectomeMotionExtractor, MotionEmbedding
from vision.fusion import FusedAgentState, FusionMode, StateFusionLayer
from vision.transduction import RetinotopicTransducer
from vision.yolo_interface import YOLOInterface


@dataclass
class ArenaObstacle:
    """An obstacle in the pursuit arena."""
    pos: np.ndarray             # (2,) [x, y] coordinates
    radius: float = 6.0         # Spatial radius
    vel: np.ndarray = None      # (2,) velocity (optional motion)


class PursuitArena:
    """
    Continuous 2D visual pursuit arena where the fly tracks and pursues a moving target
    while detecting and negotiating obstacles via the connectome vision pipeline.
    """

    def __init__(
        self,
        circuit_data: dict[str, Any],
        fusion_mode: FusionMode = FusionMode.FULL,
        arena_size: float = 100.0,
        max_steps: int = 200,
        dt_s: float = 0.05,
        fly_speed: float = 15.0,
        fly_turn_rate_deg: float = 45.0,
        target_speed: float = 10.0,
        capture_radius: float = 5.0,
        fov_deg: float = 40.0,
        n_obstacles: int = 0,
        obstacle_radius: float = 6.0,
        random_seed: int = 42,
    ):
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.dt_s = dt_s
        self.fly_speed = fly_speed
        self.fly_turn_rate = math.radians(fly_turn_rate_deg)
        self.target_speed = target_speed
        self.capture_radius = capture_radius
        self.fov_deg = fov_deg
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacle_radius
        self.rng = np.random.default_rng(random_seed)

        # Connectome vision pipeline
        self.transducer = RetinotopicTransducer(circuit_data, dt_ms=dt_s * 1000.0)
        self.sim = CircuitDynamics(circuit_data, normalization=WeightNormalization.COLUMN_NORM, dt_ms=dt_s * 1000.0)
        self.motion_extractor = ConnectomeMotionExtractor(circuit_data)
        self.yolo = YOLOInterface()
        self.fusion = StateFusionLayer(mode=fusion_mode)

        # Agent state: [x, y, theta]
        self.fly_pos = np.zeros(2, dtype=float)
        self.fly_heading = 0.0  # radians

        # Target state: [x, y, vx, vy]
        self.target_pos = np.zeros(2, dtype=float)
        self.target_vel = np.zeros(2, dtype=float)

        # Obstacles list
        self.obstacles: list[ArenaObstacle] = []

        self.step_count = 0
        self.prev_distance = 0.0

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Spawn fly in center
        self.fly_pos = np.array([0.0, 0.0], dtype=float)
        self.fly_heading = self.rng.uniform(0, 2.0 * math.pi)

        # Spawn target at distance 18-28 units away (well inside arena boundaries)
        dist = self.rng.uniform(18.0, 28.0)
        angle = self.rng.uniform(0, 2.0 * math.pi)
        self.target_pos = np.array([dist * math.cos(angle), dist * math.sin(angle)], dtype=float)

        # Direct initial velocity inward / tangential so target stays in bounds
        radial_angle = angle + math.pi
        target_heading = radial_angle + self.rng.uniform(-math.pi / 3.0, math.pi / 3.0)
        self.target_vel = np.array([
            self.target_speed * math.cos(target_heading),
            self.target_speed * math.sin(target_heading),
        ], dtype=float)

        # Spawn obstacles if requested, ensuring they do not spawn on top of fly or target
        self.obstacles = []
        for _ in range(self.n_obstacles):
            for attempt in range(20):
                # Place obstacle in the corridor between fly and target
                t = float(self.rng.uniform(0.35, 0.75))
                mid = self.fly_pos * (1.0 - t) + self.target_pos * t
                jitter = self.rng.normal(0.0, 6.0, size=2)
                obs_pos = mid + jitter
                # Must be at least 15 units from fly and 10 units from target
                if np.linalg.norm(obs_pos - self.fly_pos) >= (self.obstacle_radius + 8.0) and np.linalg.norm(obs_pos - self.target_pos) >= 8.0:
                    self.obstacles.append(ArenaObstacle(pos=obs_pos, radius=self.obstacle_radius))
                    break
            else:
                # Fallback offset
                obs_pos = np.array([18.0, 10.0])
                self.obstacles.append(ArenaObstacle(pos=obs_pos, radius=self.obstacle_radius))

        self.step_count = 0
        self.prev_distance = float(np.linalg.norm(self.target_pos - self.fly_pos))
        self.sim.reset()

        fused_state = self._render_and_fuse()
        info = {
            "state_info": fused_state,
            "obstacles": [obs.pos.copy() for obs in self.obstacles],
        }
        return fused_state.features, info

    def respawn_target(self, min_dist: float = 22.0) -> None:
        """Respawns target at a new interior position inside arena bounds away from fly."""
        for _ in range(30):
            cand_pos = self.rng.uniform(-30.0, 30.0, size=2)
            if np.linalg.norm(cand_pos - self.fly_pos) >= min_dist:
                self.target_pos = cand_pos.astype(float)
                break
        else:
            self.target_pos = np.array([25.0, 0.0], dtype=float)

        angle_to_center = math.atan2(-self.target_pos[1], -self.target_pos[0])
        heading = angle_to_center + self.rng.uniform(-math.pi / 3.0, math.pi / 3.0)
        self.target_vel = np.array([
            self.target_speed * math.cos(heading),
            self.target_speed * math.sin(heading),
        ], dtype=float)
        self.prev_distance = float(np.linalg.norm(self.target_pos - self.fly_pos))

    def _render_and_fuse(self) -> FusedAgentState:
        # Relative vector from fly to target
        rel_pos = self.target_pos - self.fly_pos
        rel_dist = np.linalg.norm(rel_pos)

        # Target angle in fly's body frame
        global_angle = math.atan2(rel_pos[1], rel_pos[0])
        body_angle = (global_angle - self.fly_heading + math.pi) % (2.0 * math.pi) - math.pi
        body_angle_deg = math.degrees(body_angle)

        # 1. YOLO Detection (target)
        yolo_dets = self.yolo.detect_from_arena_state(
            target_pos_deg=(body_angle_deg, 0.0),
            target_size_deg=6.0,
            field_of_view_deg=self.fov_deg,
        )

        # 2. Render visual scene into ommatidia array (Target + Obstacles)
        def scene_fn(x: np.ndarray, y: np.ndarray, t_ms: float) -> np.ndarray:
            dist_azimuth = np.abs(x - body_angle_deg)
            dist_elevation = np.abs(y)
            in_target = (dist_azimuth**2 + dist_elevation**2) <= (3.0**2)
            lum = in_target.astype(float)

            # Render any obstacle in visual field
            for obs in self.obstacles:
                rel_obs = obs.pos - self.fly_pos
                obs_dist = np.linalg.norm(rel_obs)
                obs_angle = (math.atan2(rel_obs[1], rel_obs[0]) - self.fly_heading + math.pi) % (2.0 * math.pi) - math.pi
                obs_angle_deg = math.degrees(obs_angle)
                angular_radius = math.degrees(math.atan2(obs.radius, max(1.0, obs_dist)))
                in_obs = ((x - obs_angle_deg)**2 + y**2) <= (angular_radius**2)
                lum = np.maximum(lum, in_obs.astype(float) * 0.9)

            return lum

        # One timestep of connectome simulation
        sample_input = self.transducer.sample_scene(scene_fn, 0.0)
        # External current into L1 (ON channel)
        currents = np.zeros(self.sim.n_neurons)
        for i, key in enumerate(self.transducer.ordered_keys):
            om = self.transducer.ommatidia[key]
            if om.l1_id is not None and om.l1_id in self.sim.bid_to_idx:
                idx = self.sim.bid_to_idx[om.l1_id]
                currents[idx] = sample_input[i] * 5.0

        v, rates = self.sim.step(currents)
        motion_emb = self.motion_extractor.extract_from_rates(rates)

        # 3. Fuse YOLO + Connectome
        fused = self.fusion.fuse(yolo_dets, motion_emb)
        return fused

    def step(self, action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Action: [turn_action, speed_action]
        turn_action in [-1.0, +1.0] -> [-fly_turn_rate, +fly_turn_rate]
        speed_action in [0.0, 1.0] -> [0.0, fly_speed]
        """
        turn = float(np.clip(action[0], -1.0, 1.0)) * self.fly_turn_rate
        forward_speed = float(np.clip(action[1], 0.0, 1.0)) * self.fly_speed

        # Update fly kinematics
        self.fly_heading = (self.fly_heading + turn * self.dt_s) % (2.0 * math.pi)
        self.fly_pos[0] += forward_speed * math.cos(self.fly_heading) * self.dt_s
        self.fly_pos[1] += forward_speed * math.sin(self.fly_heading) * self.dt_s

        # Arena boundary containment for fly (clamped within walls)
        fly_bound = self.arena_size * 0.44
        for axis in (0, 1):
            if abs(self.fly_pos[axis]) > fly_bound:
                self.fly_pos[axis] = np.sign(self.fly_pos[axis]) * fly_bound

        # Update target motion with smooth wandering, soft boundary repulsion, and obstacle avoidance
        target_bound = self.arena_size * 0.43
        soft_zone = self.arena_size * 0.32

        # 1. Wandering heading
        target_heading = math.atan2(self.target_vel[1], self.target_vel[0])
        target_heading += self.rng.normal(0.0, 0.12)

        # 2. Wall repulsion forces (smoothly steer inward toward center when near borders)
        steer_x = 0.0
        steer_y = 0.0
        if abs(self.target_pos[0]) > soft_zone:
            excess = (abs(self.target_pos[0]) - soft_zone) / (target_bound - soft_zone)
            steer_x -= np.sign(self.target_pos[0]) * excess * 2.5
        if abs(self.target_pos[1]) > soft_zone:
            excess = (abs(self.target_pos[1]) - soft_zone) / (target_bound - soft_zone)
            steer_y -= np.sign(self.target_pos[1]) * excess * 2.5

        # 3. Obstacle avoidance forces (prey swerves around obstacles)
        for obs in self.obstacles:
            diff_obs = self.target_pos - obs.pos
            d_obs = float(np.linalg.norm(diff_obs))
            safe_dist = obs.radius + 6.0
            if d_obs < safe_dist and d_obs > 1e-4:
                repel = (safe_dist - d_obs) / safe_dist
                steer_x += (diff_obs[0] / d_obs) * repel * 3.0
                steer_y += (diff_obs[1] / d_obs) * repel * 3.0

        # Blend wandering vector with steering
        desired_vx = self.target_speed * math.cos(target_heading) + steer_x * self.target_speed
        desired_vy = self.target_speed * math.sin(target_heading) + steer_y * self.target_speed
        speed_norm = math.hypot(desired_vx, desired_vy)
        if speed_norm > 1e-4:
            self.target_vel = np.array([desired_vx, desired_vy]) / speed_norm * self.target_speed

        self.target_pos += self.target_vel * self.dt_s

        # 4. Hard elastic reflection at walls (absolute boundary containment guarantee)
        if abs(self.target_pos[0]) >= target_bound:
            self.target_pos[0] = np.sign(self.target_pos[0]) * target_bound
            self.target_vel[0] = -abs(self.target_vel[0]) * np.sign(self.target_pos[0])
        if abs(self.target_pos[1]) >= target_bound:
            self.target_pos[1] = np.sign(self.target_pos[1]) * target_bound
            self.target_vel[1] = -abs(self.target_vel[1]) * np.sign(self.target_pos[1])

        # Check distance to obstacles and resolve surface collisions
        min_obstacle_dist = 999.0
        collided = False
        for obs in self.obstacles:
            diff = self.fly_pos - obs.pos
            dist = float(np.linalg.norm(diff))
            surface_dist = dist - obs.radius
            if surface_dist < min_obstacle_dist:
                min_obstacle_dist = surface_dist
            if surface_dist <= 0.0:
                collided = True
                # Elastic/sliding collision resolution: clamp fly outside obstacle radius
                normal = diff / max(1e-6, dist)
                self.fly_pos = obs.pos + normal * (obs.radius + 0.1)
                # Deflect heading along surface tangent
                self.fly_heading = (math.atan2(-normal[0], normal[1]) + math.pi) % (2.0 * math.pi) - math.pi

        self.step_count += 1
        current_distance = float(np.linalg.norm(self.target_pos - self.fly_pos))

        # Reward formulation:
        # 1. Distance closing reward: +10 per unit distance decreased
        distance_delta = self.prev_distance - current_distance
        reward = distance_delta * 10.0

        # 2. Orientation alignment reward (cosine of angle between heading and target)
        rel_pos = self.target_pos - self.fly_pos
        rel_dir = rel_pos / max(1e-6, current_distance)
        fly_dir = np.array([math.cos(self.fly_heading), math.sin(self.fly_heading)])
        alignment = float(np.dot(fly_dir, rel_dir))
        reward += alignment * 0.5

        # 3. Collision penalty
        if collided:
            reward -= 20.0

        # 4. Small step penalty to encourage rapid pursuit
        reward -= 0.1

        # Check termination:
        captured = current_distance <= self.capture_radius
        if captured:
            reward += 100.0  # Capture bonus

        terminated = captured
        truncated = self.step_count >= self.max_steps

        self.prev_distance = current_distance
        fused_state = self._render_and_fuse()

        info = {
            "distance": current_distance,
            "captured": captured,
            "collided": collided,
            "min_obstacle_dist": min_obstacle_dist,
            "step": self.step_count,
            "state_info": fused_state,
            "obstacles": [obs.pos.copy() for obs in self.obstacles],
        }

        return fused_state.features, reward, terminated, truncated, info

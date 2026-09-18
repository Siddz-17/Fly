"""
tests/test_fusion_and_ppo.py

Automated test suite verifying the visual embeddings, YOLO interface,
multimodal fusion layer, and PPO training components.
"""

import json
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"

from behavior.pursuit_arena import PursuitArena
from training.ppo import ActorCriticNetwork, PPOTrainer
from vision.embeddings import ConnectomeMotionExtractor, MotionEmbedding
from vision.fusion import FusionMode, StateFusionLayer
from vision.yolo_interface import YOLODetection, YOLOInterface


@pytest.fixture
def circuit_data():
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_motion_extractor(circuit_data):
    extractor = ConnectomeMotionExtractor(circuit_data)
    dummy_rates = np.zeros(len(extractor.ordered_bids))

    # Activate T4a (Right-selective)
    t4a_idx = extractor.subtype_indices["T4a"]
    dummy_rates[t4a_idx] = 2.0

    emb = extractor.extract_from_rates(dummy_rates)
    assert emb.vx > 0.0  # Horizontal motion to the right
    assert emb.speed > 0.0
    vec = emb.to_feature_vector()
    assert len(vec) == 6


def test_yolo_interface():
    yolo = YOLOInterface()
    dets = yolo.detect_from_arena_state(target_pos_deg=(10.0, -5.0), field_of_view_deg=40.0)
    assert len(dets) == 1
    d = dets[0]
    assert -1.0 <= d.center_x <= 1.0
    assert -1.0 <= d.center_y <= 1.0
    assert d.confidence > 0.9


def test_state_fusion():
    fusion = StateFusionLayer(mode=FusionMode.FULL)
    det = YOLODetection(class_id=0, class_name="target", confidence=0.9, center_x=0.5, center_y=0.2, width=0.1, height=0.1)
    emb = MotionEmbedding(
        time_ms=0.0, vx=1.2, vy=0.0, speed=1.2, loom=0.1, on_power=2.0, off_power=0.0,
        spatial_field_vx=np.zeros(5), spatial_field_vy=np.zeros(5)
    )

    fused = fusion.fuse([det], emb)
    assert fused.target_visible is True
    assert len(fused.features) == 12  # 5 YOLO + 6 Connectome + 1 Alignment


def test_pursuit_arena_step(circuit_data):
    env = PursuitArena(circuit_data, max_steps=10)
    obs, info = env.reset()
    assert len(obs) == 12

    action = [0.2, 0.8]  # Turn, speed
    next_obs, reward, term, trunc, step_info = env.step(action)
    assert len(next_obs) == 12
    assert isinstance(reward, float)
    assert "distance" in step_info


def test_ppo_network_and_trainer(circuit_data):
    net = ActorCriticNetwork(state_dim=12, action_dim=2, seed=42)
    s = np.zeros(12)
    action, log_p, val = net.sample_action(s)
    assert action.shape == (2,)
    assert -1.0 <= action[0] <= 1.0
    assert isinstance(log_p, float)
    assert isinstance(val, float)

    env = PursuitArena(circuit_data, max_steps=20)
    trainer = PPOTrainer(env, state_dim=12, action_dim=2)
    rollout = trainer.collect_rollout(n_steps=20)
    assert len(rollout["states"]) == 20

    mean_r = trainer.train_step(rollout, n_epochs=1)
    assert isinstance(mean_r, float)


def test_pursuit_arena_obstacles(circuit_data):
    """Verify obstacle initialization, collision detection, and distance metrics."""
    env = PursuitArena(circuit_data, n_obstacles=3, obstacle_radius=6.0, max_steps=10)
    obs, info = env.reset()
    assert len(env.obstacles) == 3
    assert "obstacles" in info
    assert len(info["obstacles"]) == 3

    # Step with forward movement
    action = [0.0, 1.0]
    next_obs, reward, term, trunc, step_info = env.step(action)
    assert "min_obstacle_dist" in step_info
    assert isinstance(step_info["min_obstacle_dist"], float)
    assert "collided" in step_info
    assert isinstance(step_info["collided"], bool)


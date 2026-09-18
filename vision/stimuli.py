"""
vision/stimuli.py

Milestone 4: Controlled Synthetic Visual Stimulus Suite.

Provides mathematical scene luminance functions S(x, y, t) for:
1. Moving bars (4 cardinal directions: right, left, up, down; ON and OFF polarity).
2. Drifting sinusoidal and square-wave gratings.
3. Static bar flashes (control for onset response).
4. Full-field luminance flicker (control for non-directional brightness changes).

Conforms to [SYNTHETIC TEST STIMULI] taxonomy tier.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def create_moving_bar_stimulus(
    direction: str = "right",
    velocity_deg_s: float = 50.0,
    bar_width_deg: float = 6.0,
    contrast: float = 1.0,
    background: float = 0.0,
    initial_offset_deg: float = -20.0,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Creates a moving bar stimulus function S(x, y, t).

    direction: 'right' (+x), 'left' (-x), 'up' (+y), 'down' (-y)
    contrast: +1.0 for bright bar (ON edge), -1.0 for dark bar (OFF edge)
    """
    v_deg_ms = velocity_deg_s / 1000.0

    def scene_fn(x: np.ndarray, y: np.ndarray, t_ms: float) -> np.ndarray:
        # Compute center position of the bar at time t
        pos = initial_offset_deg + v_deg_ms * t_ms

        if direction == "right":
            # Bar oriented vertically, sweeping along +x
            dist = np.abs(x - pos)
        elif direction == "left":
            # Bar oriented vertically, sweeping along -x
            pos_left = -initial_offset_deg - v_deg_ms * t_ms
            dist = np.abs(x - pos_left)
        elif direction == "up":
            # Bar oriented horizontally, sweeping along +y
            dist = np.abs(y - pos)
        elif direction == "down":
            # Bar oriented horizontally, sweeping along -y
            pos_down = -initial_offset_deg - v_deg_ms * t_ms
            dist = np.abs(y - pos_down)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Square bar profile
        in_bar = dist <= (bar_width_deg / 2.0)
        luminance = np.full_like(x, background, dtype=float)
        luminance[in_bar] = background + contrast
        return np.clip(luminance, 0.0, 1.0)

    return scene_fn


def create_static_flash_stimulus(
    start_ms: float = 100.0,
    duration_ms: float = 200.0,
    bar_width_deg: float = 6.0,
    contrast: float = 1.0,
    background: float = 0.0,
    center_x_deg: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Creates a static bar flash (control for non-directional onset response).
    """
    def scene_fn(x: np.ndarray, y: np.ndarray, t_ms: float) -> np.ndarray:
        luminance = np.full_like(x, background, dtype=float)
        if start_ms <= t_ms <= (start_ms + duration_ms):
            in_bar = np.abs(x - center_x_deg) <= (bar_width_deg / 2.0)
            luminance[in_bar] = background + contrast
        return np.clip(luminance, 0.0, 1.0)

    return scene_fn


def create_fullfield_flicker_stimulus(
    frequency_hz: float = 2.0,
    contrast: float = 0.5,
    mean_luminance: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Full-field temporal sinusoidal flicker (control for motion vs luminance change).
    """
    def scene_fn(x: np.ndarray, y: np.ndarray, t_ms: float) -> np.ndarray:
        omega = 2.0 * np.pi * frequency_hz / 1000.0
        val = mean_luminance + contrast * np.sin(omega * t_ms)
        return np.full_like(x, np.clip(val, 0.0, 1.0), dtype=float)

    return scene_fn

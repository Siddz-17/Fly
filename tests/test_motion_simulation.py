"""
tests/test_motion_simulation.py

Automated test suite verifying the visual transduction and temporal
circuit dynamics of the MVP-A core motion detector.
"""

import json
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"

from connectome.dynamics import CircuitDynamics, WeightNormalization
from vision.stimuli import create_moving_bar_stimulus, create_static_flash_stimulus
from vision.transduction import RetinotopicTransducer


@pytest.fixture
def circuit_data():
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_retinotopic_transducer_ommatidia(circuit_data):
    transducer = RetinotopicTransducer(circuit_data)
    assert transducer.n_ommatidia == 25
    assert len(transducer.positions_deg) == 25

    # Center ommatidium should be near (0, 0) in visual angles
    dists = np.sqrt(transducer.positions_deg[:, 0]**2 + transducer.positions_deg[:, 1]**2)
    assert np.min(dists) < 1.0


def test_on_off_rectification(circuit_data):
    transducer = RetinotopicTransducer(circuit_data, dt_ms=1.0)

    # Moving bright bar (ON stimulus)
    scene_fn = create_moving_bar_stimulus(direction="right", contrast=1.0)
    out = transducer.transduce_timeseries(scene_fn, duration_ms=200.0)

    # ON channel (L1) must have strong positive activation
    assert np.max(out["L1"]) > 0.0
    # Both channels non-negative
    assert np.all(out["L1"] >= 0.0)
    assert np.all(out["L2"] >= 0.0)


def test_circuit_simulation_runs(circuit_data):
    transducer = RetinotopicTransducer(circuit_data, dt_ms=1.0)
    sim = CircuitDynamics(circuit_data, normalization=WeightNormalization.COLUMN_NORM, dt_ms=1.0)

    scene_fn = create_moving_bar_stimulus(direction="right", contrast=1.0)
    out = transducer.transduce_timeseries(scene_fn, duration_ms=100.0)
    n_steps = len(out["time_ms"])

    sim_out = sim.simulate(out["body_currents"], n_steps)
    rates = sim_out["rates"]

    assert rates.shape == (n_steps, sim.n_neurons)
    assert np.all(rates >= 0.0)
    assert np.max(rates) > 0.0


def test_normalization_schemes(circuit_data):
    for norm in [
        WeightNormalization.RAW,
        WeightNormalization.COLUMN_NORM,
        WeightNormalization.ROW_NORM,
        WeightNormalization.LOG,
        WeightNormalization.DEGREE_NORM,
    ]:
        sim = CircuitDynamics(circuit_data, normalization=norm)
        assert sim.weights.shape == (sim.n_neurons, sim.n_neurons)
        assert not np.all(sim.weights == 0.0)

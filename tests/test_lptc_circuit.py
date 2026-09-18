"""
tests/test_lptc_circuit.py

Unit tests for Milestone MVP-B: Lobula Plate Tangential Cell (LPTC) circuit,
neurotransmitter assignments, temporal dynamics, and biological motion opponency.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from connectome.lptc_dynamics import LPTCDynamics
from vision.embeddings import LPTCMotionExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUIT_B_PATH = REPO_ROOT / "data" / "processed" / "mvp_b_circuit.json"


@pytest.fixture(scope="module")
def mvp_b_circuit():
    if not CIRCUIT_B_PATH.exists():
        pytest.fail(f"MVP-B circuit not found at {CIRCUIT_B_PATH}. Run extract_mvp_b.py first.")
    with open(CIRCUIT_B_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_mvp_b_circuit_integrity(mvp_b_circuit):
    """Verify 0 synthetic neurons and valid male-cns:v1.0 metadata."""
    meta = mvp_b_circuit["metadata"]
    assert meta["dataset"] == "male-cns:v1.0"
    assert meta["total_neurons"] > 850
    assert meta["total_connections"] > 5000
    assert meta["total_synapses"] > 80000

    neurons = mvp_b_circuit["neurons"]
    for bid_str, n in neurons.items():
        assert n["tier"] == "REAL_CONNECTOME", f"Neuron {bid_str} contains synthetic fallback tier!"
        assert n["soma_side"] == "R", f"Neuron {bid_str} is not in right hemisphere!"

    # Verify LPTC and LPi types
    cell_types = set(n["cell_type"] for n in neurons.values())
    assert "HSN" in cell_types
    assert "HSE" in cell_types
    assert "HSS" in cell_types
    assert "VS" in cell_types
    assert "LPi21" in cell_types


def test_lptc_neurotransmitter_assignments(mvp_b_circuit):
    """Verify consensus neurotransmitter labels."""
    neurons = mvp_b_circuit["neurons"]
    hs_bodies = [n for n in neurons.values() if n["cell_type"] in {"HSN", "HSE", "HSS"}]
    for n in hs_bodies:
        assert n["neurotransmitter"] == "acetylcholine"

    lpi21_bodies = [n for n in neurons.values() if n["cell_type"] == "LPi21"]
    assert len(lpi21_bodies) > 0
    for n in lpi21_bodies:
        assert n["neurotransmitter"] == "gaba"


def test_lptc_dynamics_push_pull_opponency(mvp_b_circuit):
    """
    Test biological motion opponency:
    1. Progressive (T4a/T5a) excitation depolarizes HS cells (delta_V > 0).
    2. Regressive (T4b/T5b) excitation activates LPi21 and hyperpolarizes HS cells (delta_V < 0).
    """
    sim = LPTCDynamics(mvp_b_circuit, dt_ms=0.5)

    neurons = mvp_b_circuit["neurons"]
    t4a_bodies = [int(b) for b, n in neurons.items() if n["cell_type"] == "T4a"]
    t4b_bodies = [int(b) for b, n in neurons.items() if n["cell_type"] == "T4b"]

    assert len(t4a_bodies) > 0
    assert len(t4b_bodies) > 0

    # Progressive test (Preferred direction)
    sim.reset()
    pref_rates = {b: 1.0 for b in t4a_bodies}
    for _ in range(50):
        out_pref = sim.step_from_t4_t5_rates(pref_rates)

    assert out_pref["v_hs_mean"] > 0.0, f"Preferred stimulation failed to depolarize HS cells: {out_pref['v_hs_mean']}"

    # Regressive test (Null direction)
    sim.reset()
    null_rates = {b: 1.0 for b in t4b_bodies}
    for _ in range(50):
        out_null = sim.step_from_t4_t5_rates(null_rates)

    assert out_null["v_lpi21"] > 0.0, "Regressive stimulation failed to activate LPi21 interneurons!"
    assert out_null["v_hs_mean"] < 0.0, f"Null stimulation failed to hyperpolarize HS cells: {out_null['v_hs_mean']}"


def test_lptc_motion_extractor(mvp_b_circuit):
    """Verify LPTCMotionExtractor produces valid 6D motion embeddings."""
    extractor = LPTCMotionExtractor(mvp_b_circuit)
    neurons = mvp_b_circuit["neurons"]
    t4a_bodies = [int(b) for b, n in neurons.items() if n["cell_type"] == "T4a"]

    rates = {b: 1.0 for b in t4a_bodies}
    emb = extractor.extract_from_t4_t5_rates(rates, time_ms=10.0)

    vec = emb.to_feature_vector()
    assert len(vec) == 6
    assert emb.vx > 0.0  # Progressive motion gives positive horizontal velocity
    assert emb.speed > 0.0
    assert emb.on_power > 0.0

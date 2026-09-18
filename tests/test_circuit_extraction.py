"""
tests/test_circuit_extraction.py

Automated test suite verifying the integrity of the MVP-A extracted circuit
and Milestone 1.5 Biological Verification Audit.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUIT_PATH = REPO_ROOT / "data" / "processed" / "mvp_a_circuit.json"
AUDIT_PATH = REPO_ROOT / "data" / "processed" / "biological_audit.json"


def test_circuit_artifact_exists():
    assert CIRCUIT_PATH.exists(), f"Circuit artifact missing at {CIRCUIT_PATH}"
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    assert meta.get("dataset") == "male-cns:v1.0"
    assert meta.get("total_neurons", 0) > 100
    assert meta.get("total_connections", 0) > 1000
    assert meta.get("total_synapses", 0) > 10000


def test_biological_integrity_no_synthetic_fallbacks():
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    neurons = data.get("neurons", {})
    connections = data.get("connections", [])

    # Rule 1 & Rule 2: Every neuron has a real integer bodyId and tier is REAL_CONNECTOME
    for bid_str, n in neurons.items():
        assert int(bid_str) == n["body_id"]
        assert n["tier"] == "REAL_CONNECTOME"
        assert n["cell_type"] != "unknown"
        assert n["cell_type"] != ""

    # Rule 3: Every connection has positive synapse count and real pre/post IDs
    for c in connections:
        assert c["synapse_count"] >= 3
        assert str(c["pre_id"]) in neurons
        assert str(c["post_id"]) in neurons
        assert c["tier"] == "REAL_CONNECTOME"


def test_t4_t5_presence():
    with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    neurons = data.get("neurons", {})
    t4_types = {n["cell_type"] for n in neurons.values() if n["cell_type"].startswith("T4")}
    t5_types = {n["cell_type"] for n in neurons.values() if n["cell_type"].startswith("T5")}

    # Must contain all 4 cardinal subtypes for both T4 and T5
    assert {"T4a", "T4b", "T4c", "T4d"}.issubset(t4_types)
    assert {"T5a", "T5b", "T5c", "T5d"}.issubset(t5_types)


def test_biological_audit_report():
    assert AUDIT_PATH.exists(), f"Audit registry missing at {AUDIT_PATH}"
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)

    table = audit.get("audit_table", [])
    assert len(table) > 10

    # Verify critical EMD pathways are present in the audit
    pathways = {row["pathway"] for row in table}
    assert any("Mi1 -> T4a" in p for p in pathways)
    assert any("Mi9 -> T4a" in p for p in pathways)
    assert any("Mi4 -> T4a" in p for p in pathways)
    assert any("Tm1 -> T5a" in p for p in pathways)
    assert any("Tm9 -> T5a" in p for p in pathways)

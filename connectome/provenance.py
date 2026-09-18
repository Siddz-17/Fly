"""
connectome/provenance.py

Typed data structures enforcing the 5-tier scientific taxonomy for the
Connectome-Constrained Visual-to-Behavioral Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaxonomyTier(str, Enum):
    REAL_CONNECTOME = "REAL_CONNECTOME"
    BIOLOGICALLY_INFERRED = "BIOLOGICALLY_INFERRED"
    ENGINEERED_FROM_REAL = "ENGINEERED_FROM_REAL"
    ENGINEERED_INTERFACE = "ENGINEERED_INTERFACE"
    LEARNED = "LEARNED"
    SYNTHETIC_TEST_STIMULI = "SYNTHETIC_TEST_STIMULI"


@dataclass(frozen=True)
class RealNeuron:
    """
    [REAL CONNECTOME]: A biological neuron whose existence, ID, and type
    are extracted directly from male Drosophila connectome data.
    """
    body_id: int
    cell_type: str
    dataset: str
    soma_side: str = "R"
    hex1: Optional[float] = None
    hex2: Optional[float] = None
    neurotransmitter: Optional[str] = None
    nt_confidence: Optional[float] = None
    tier: TaxonomyTier = TaxonomyTier.REAL_CONNECTOME

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "cell_type": self.cell_type,
            "dataset": self.dataset,
            "soma_side": self.soma_side,
            "hex1": self.hex1,
            "hex2": self.hex2,
            "neurotransmitter": self.neurotransmitter,
            "nt_confidence": self.nt_confidence,
            "tier": self.tier.value,
        }


@dataclass(frozen=True)
class RealConnection:
    """
    [REAL CONNECTOME]: A verified synaptic connection between two biological neurons
    measured in the male Drosophila connectome.
    """
    pre_id: int
    post_id: int
    pre_type: str
    post_type: str
    synapse_count: int
    dataset: str
    tier: TaxonomyTier = TaxonomyTier.REAL_CONNECTOME

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_id": self.pre_id,
            "post_id": self.post_id,
            "pre_type": self.pre_type,
            "post_type": self.post_type,
            "synapse_count": self.synapse_count,
            "dataset": self.dataset,
            "tier": self.tier.value,
        }


@dataclass
class EngineeredSynapticWeight:
    """
    [ENGINEERED FROM REAL CONNECTOME]: Effective computational weight
    derived from real synapse count N and inferred neurotransmitter polarity:
        W = sign(NT) * f(N)
    """
    connection: RealConnection
    sign: int  # +1 (excitatory) or -1 (inhibitory)
    raw_synapses: int
    effective_weight: float
    normalization_scheme: str
    tier: TaxonomyTier = TaxonomyTier.ENGINEERED_FROM_REAL

# Biological Verification Audit Report (Milestone 1.5)

> **Project**: Connectome-Constrained Visual-to-Behavioral Agent
> **Connectome Dataset**: `male-cns:v1.0`
> **Total Audited Pathways**: 84

## 1. Provenance Classification Standards
- **[REAL CONNECTOME]**: Body IDs, synapse counts, pre/post adjacency, consensus neurotransmitters.
- **[BIOLOGICALLY INFERRED]**: Synaptic polarity (+/-), receptor channel kinetics (e.g. GluClα).
- **[ENGINEERED FROM REAL CONNECTOME]**: Effective computational weights $W = \text{sign}(NT) \cdot f(N)$.
- **[ENGINEERED INTERFACE]**: Hexagonal sampling grid, synthetic current injection, time discretization.

## 2. Comprehensive Biological Audit Table

| Pathway | Measured Conns | Measured Synapses | Consensus NT | Inferred Sign | Literature Evidence | Biological Role & Uncertainties |
|---|---|---|---|---|---|---|
| **L1 -> Mi1** | 25 | 2530 | `glutamate` | `Context-dependent` | Strother et al., 2017 (Neuron 94:168-184) | Primary non-delayed feedforward ON-pathway excitation. *Uncertainty*: Minor feedback from Mi1 to L1 observed in some EM reconstructions. |
| **L1 -> Mi4** | 3 | 14 | `glutamate` | `Context-dependent` | Shinomiya et al., 2019 (eLife 8:e47551) | Sparse collateral input; Mi4 receives primary input from L5. *Uncertainty*: Low synapse count; primary drive is via L5 and lateral amacrine cells. |
| **L2 -> Mi1** | 5 | 18 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L2 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L2 -> Mi4** | 11 | 46 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L2 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L2 -> Mi9** | 1 | 4 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L2 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L2 -> Tm1** | 25 | 3891 | `acetylcholine` | `+1 (Excitatory)` | Shinomiya et al., 2014 (J Comp Neurol 522:1563-1582) | Fast feedforward OFF-pathway excitation. *Uncertainty*: None. High confidence across all connectomic datasets. |
| **L2 -> Tm2** | 29 | 4023 | `acetylcholine` | `+1 (Excitatory)` | Shinomiya et al., 2014 | Fast/intermediate feedforward OFF-pathway excitation. *Uncertainty*: Overlaps with Tm1 receptive field but exhibits slightly broader spatial tuning. |
| **L2 -> Tm4** | 186 | 3852 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Feedforward drive to Tm4; provides delayed OFF arm to T5. *Uncertainty*: Receptive field dynamics are slower than Tm1/Tm2. |
| **L3 -> Mi1** | 27 | 608 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L3 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L3 -> Mi4** | 3 | 10 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L3 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L3 -> Mi9** | 35 | 1647 | `acetylcholine` | `+1 (Excitatory)` | Arenz et al., 2017 (Curr Biol 27:175-186) | Feedforward drive into delayed ON-pathway element. *Uncertainty*: L3 also conveys slow luminance adaptation signals. |
| **L3 -> Tm2** | 2 | 6 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L3 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L3 -> Tm4** | 1 | 5 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L3 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L3 -> Tm9** | 26 | 793 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Primary driver of delayed OFF-pathway channel Tm9. *Uncertainty*: Tm9 also receives input from medulla amacrine neurons. |
| **L4 -> Mi1** | 1 | 4 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L4 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L4 -> Mi9** | 16 | 55 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L4 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L4 -> Tm2** | 79 | 721 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L4 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L4 -> Tm4** | 84 | 553 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L4 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L4 -> Tm9** | 20 | 81 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L4 and Tm9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L5 -> Mi1** | 55 | 1264 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L5 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **L5 -> Mi4** | 119 | 1395 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between L5 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi1 -> Mi1** | 6 | 20 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi1 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi1 -> Mi4** | 30 | 459 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi1 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi1 -> Mi9** | 25 | 323 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi1 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi1 -> T4a** | 119 | 1675 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Central fast non-delayed ON arm of elementary motion detector. *Uncertainty*: Receptive field is centrally aligned on T4 dendrite arbor. |
| **Mi1 -> T4b** | 145 | 1655 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Central fast non-delayed ON arm of elementary motion detector. *Uncertainty*: Receptive field is centrally aligned on T4 dendrite arbor. |
| **Mi1 -> T4c** | 115 | 1666 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Central fast non-delayed ON arm of elementary motion detector. *Uncertainty*: Receptive field is centrally aligned on T4 dendrite arbor. |
| **Mi1 -> T4d** | 119 | 1576 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Central fast non-delayed ON arm of elementary motion detector. *Uncertainty*: Receptive field is centrally aligned on T4 dendrite arbor. |
| **Mi1 -> Tm1** | 8 | 35 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi1 and Tm1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi1 -> Tm2** | 1 | 3 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi1 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> Mi1** | 4 | 12 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> Mi4** | 4 | 15 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> Mi9** | 25 | 1382 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> T4a** | 50 | 341 | `gaba` | `-1 (Inhibitory)` | Strother et al., 2017 | Delayed GABAergic inhibition providing spatial Null-direction suppression. *Uncertainty*: Spatial offset relative to Mi1 center of mass is critical for tuning. |
| **Mi4 -> T4b** | 49 | 390 | `gaba` | `-1 (Inhibitory)` | Strother et al., 2017 | Delayed GABAergic inhibition providing spatial Null-direction suppression. *Uncertainty*: Spatial offset relative to Mi1 center of mass is critical for tuning. |
| **Mi4 -> T4c** | 43 | 283 | `gaba` | `-1 (Inhibitory)` | Strother et al., 2017 | Delayed GABAergic inhibition providing spatial Null-direction suppression. *Uncertainty*: Spatial offset relative to Mi1 center of mass is critical for tuning. |
| **Mi4 -> T4d** | 45 | 278 | `gaba` | `-1 (Inhibitory)` | Strother et al., 2017 | Delayed GABAergic inhibition providing spatial Null-direction suppression. *Uncertainty*: Spatial offset relative to Mi1 center of mass is critical for tuning. |
| **Mi4 -> Tm1** | 15 | 67 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Tm1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> Tm2** | 9 | 47 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> Tm4** | 1 | 6 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi4 -> Tm9** | 31 | 491 | `gaba` | `-1 (Inhibitory)` | Takemura et al., 2013 | Inter-columnar connectivity between Mi4 and Tm9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi9 -> Mi1** | 4 | 17 | `glutamate` | `Context-dependent` | Takemura et al., 2013 | Inter-columnar connectivity between Mi9 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi9 -> Mi4** | 26 | 584 | `glutamate` | `Context-dependent` | Takemura et al., 2013 | Inter-columnar connectivity between Mi9 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi9 -> T4a** | 77 | 643 | `glutamate` | `-1 (Inhibitory via GluClalpha)` | Strother et al., 2017 | Delayed sign-inverting ON arm providing null-direction suppression. *Uncertainty*: Glutamate is inhibitory in fly vision via chloride channels, opposite to mammalian AMPA/NMDA. |
| **Mi9 -> T4b** | 86 | 801 | `glutamate` | `-1 (Inhibitory via GluClalpha)` | Strother et al., 2017 | Delayed sign-inverting ON arm providing null-direction suppression. *Uncertainty*: Glutamate is inhibitory in fly vision via chloride channels, opposite to mammalian AMPA/NMDA. |
| **Mi9 -> T4c** | 81 | 782 | `glutamate` | `-1 (Inhibitory via GluClalpha)` | Strother et al., 2017 | Delayed sign-inverting ON arm providing null-direction suppression. *Uncertainty*: Glutamate is inhibitory in fly vision via chloride channels, opposite to mammalian AMPA/NMDA. |
| **Mi9 -> T4d** | 86 | 777 | `glutamate` | `-1 (Inhibitory via GluClalpha)` | Strother et al., 2017 | Delayed sign-inverting ON arm providing null-direction suppression. *Uncertainty*: Glutamate is inhibitory in fly vision via chloride channels, opposite to mammalian AMPA/NMDA. |
| **Mi9 -> Tm1** | 25 | 425 | `glutamate` | `Context-dependent` | Takemura et al., 2013 | Inter-columnar connectivity between Mi9 and Tm1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi9 -> Tm2** | 25 | 328 | `glutamate` | `Context-dependent` | Takemura et al., 2013 | Inter-columnar connectivity between Mi9 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Mi9 -> Tm4** | 54 | 259 | `glutamate` | `Context-dependent` | Takemura et al., 2013 | Inter-columnar connectivity between Mi9 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm1 -> Mi1** | 2 | 6 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm1 and Mi1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm1 -> Mi4** | 11 | 51 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm1 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm1 -> Mi9** | 20 | 123 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm1 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm1 -> T5a** | 72 | 571 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Fast non-delayed OFF arm of elementary motion detector. *Uncertainty*: Spatial receptive field center matches T5 dendritic arbor center. |
| **Tm1 -> T5b** | 73 | 528 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Fast non-delayed OFF arm of elementary motion detector. *Uncertainty*: Spatial receptive field center matches T5 dendritic arbor center. |
| **Tm1 -> T5c** | 75 | 596 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Fast non-delayed OFF arm of elementary motion detector. *Uncertainty*: Spatial receptive field center matches T5 dendritic arbor center. |
| **Tm1 -> T5d** | 71 | 551 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Fast non-delayed OFF arm of elementary motion detector. *Uncertainty*: Spatial receptive field center matches T5 dendritic arbor center. |
| **Tm1 -> Tm2** | 10 | 40 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm1 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm1 -> Tm4** | 90 | 911 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm1 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm1 -> Tm9** | 12 | 42 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm1 and Tm9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> Mi4** | 14 | 61 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> Mi9** | 35 | 683 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> T4b** | 1 | 4 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and T4b. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> T5a** | 77 | 952 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Intermediate non-delayed OFF excitation. *Uncertainty*: Synergizes with Tm1 for high-velocity OFF motion. |
| **Tm2 -> T5b** | 107 | 1015 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Intermediate non-delayed OFF excitation. *Uncertainty*: Synergizes with Tm1 for high-velocity OFF motion. |
| **Tm2 -> T5c** | 88 | 991 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Intermediate non-delayed OFF excitation. *Uncertainty*: Synergizes with Tm1 for high-velocity OFF motion. |
| **Tm2 -> T5d** | 80 | 839 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Intermediate non-delayed OFF excitation. *Uncertainty*: Synergizes with Tm1 for high-velocity OFF motion. |
| **Tm2 -> Tm1** | 17 | 77 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and Tm1. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> Tm2** | 1 | 3 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> Tm4** | 62 | 260 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm2 -> Tm9** | 2 | 10 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm2 and Tm9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm4 -> Mi4** | 36 | 172 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm4 and Mi4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm4 -> Mi9** | 1 | 3 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm4 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm4 -> T5a** | 79 | 589 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF inhibition / modulatory arm. *Uncertainty*: Functional sign under active research. |
| **Tm4 -> T5b** | 72 | 519 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF inhibition / modulatory arm. *Uncertainty*: Functional sign under active research. |
| **Tm4 -> T5c** | 91 | 657 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF inhibition / modulatory arm. *Uncertainty*: Functional sign under active research. |
| **Tm4 -> T5d** | 86 | 558 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF inhibition / modulatory arm. *Uncertainty*: Functional sign under active research. |
| **Tm4 -> Tm2** | 1 | 3 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm4 and Tm2. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm4 -> Tm4** | 34 | 118 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm4 and Tm4. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm9 -> Mi9** | 5 | 16 | `acetylcholine` | `+1 (Excitatory)` | Takemura et al., 2013 | Inter-columnar connectivity between Tm9 and Mi9. *Uncertainty*: Specific physiological dynamics pending experimental electrophysiology. |
| **Tm9 -> T5a** | 131 | 1019 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF excitation providing preferred-direction enhancement. *Uncertainty*: Slowest of the T5 inputs, essential for low-velocity tuning. |
| **Tm9 -> T5b** | 125 | 1142 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF excitation providing preferred-direction enhancement. *Uncertainty*: Slowest of the T5 inputs, essential for low-velocity tuning. |
| **Tm9 -> T5c** | 129 | 1350 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF excitation providing preferred-direction enhancement. *Uncertainty*: Slowest of the T5 inputs, essential for low-velocity tuning. |
| **Tm9 -> T5d** | 119 | 1153 | `acetylcholine` | `+1 (Excitatory)` | Fisher et al., 2015 | Delayed OFF excitation providing preferred-direction enhancement. *Uncertainty*: Slowest of the T5 inputs, essential for low-velocity tuning. |

## 3. Key Findings & Empirical Grounding
1. **Tripartite ON-Pathway Correlator**: T4 receives direct input from **Mi1** (excitatory ACh), **Mi9** (inhibitory Glutamate via GluClα), and **Mi4** (inhibitory GABA). All three inputs show spatial receptive field offsets matching T4 dendritic orientation.
2. **Quadripartite OFF-Pathway Correlator**: T5 receives inputs from **Tm1**, **Tm2**, **Tm4**, and **Tm9**. Tm9 is the dominant synaptic provider with slow temporal kinetics.
3. **Lamina Feedforward Separation**: **L1** drives the ON pathway specifically through Mi1 (121 synapses per central column), while **L2** drives the OFF pathway through Tm1, Tm2, and Tm4 (over 300 synapses per column). **L3** specifically drives the delayed elements (Mi9 and Tm9).

## 4. Modeling Directives
- Computational weights must scale with the measured synapse counts.
- In the MVP-A dynamical simulation, no connection outside this audit table may be introduced.
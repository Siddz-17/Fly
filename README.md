# Connectome-Constrained Visual-to-Behavioral Agent

A research-grade computational neuroscience project using the **real male *Drosophila melanogaster* connectome** as the centerpiece for visual motion perception and visually guided behavior.

---

## Central Hypothesis
> **H1: A computational model constrained by real male Drosophila optic-lobe connectivity produces measurable direction-selective responses to visual motion, and these responses differ from topology/weight-shuffled controls.**

Direction selectivity is treated as an empirical readout rather than an assumed certainty. All findings are benchmarked against formal control baselines (Real Connectome vs. Weight-Shuffled vs. Topology-Scrambled).

---

## Core Architecture Progression

```
             REAL MALE DROSOPHILA DATA (optic-lobe:v1.1 / male-cns:v1.0)
                                      │
                                      ▼
                           Retinotopic visual patch
                                      │
                                      ▼
                        R1-R6 → L1/L2/L3 → Mi/Tm → T4/T5
                                      │
                                      ▼
                              Temporal dynamics
                                      │
                                      ▼
                            Motion representation
                                      │
                             ┌────────┴────────┐
                             │                 │
                             ▼                 ▼
                       Direction/Speed      YOLOv5 (Phase 2)
                             │            Object identity
                             │            + localization
                             └────────┬────────┘
                                      ▼
                                 Fused state
                                      │
                                      ▼
                                 PPO (Phase 3)
                                      │
                                      ▼
                             FlyGym / NeuroMechFly
                                      │
                                      ▼
                                  Behavior
```

- **YOLO** tells the agent *what* it sees (semantic identity & localization).
- The **Connectome** tells the agent *how visual information evolves through time* (direction, velocity, loom).

---

## Current Status: MVP-A & Biological Audit (Completed)

### 1. Extracted Circuit (`data/processed/mvp_a_circuit.json`)
- **Biological Source**: Janelia MaleCNS (`male-cns:v1.0`) & Optic Lobe (`optic-lobe:v1.1`).
- **Anatomical ROI**: Central 25-column retinotopic patch ($\text{Hex1} \in [17..21], \text{Hex2} \in [18..22]$) in the right optic lobe (`somaSide == 'R'`).
- **Scale**:
  - **835 real, traced biological neurons** (0 synthetic fallbacks).
  - **3,890 verified synaptic connections** ($\ge 3$ synapses each).
  - **52,473 measured biological synapses**.
- **Pathways**:
  - Lamina ($\text{L1}\text{--}\text{L5}$) $\xrightarrow{\text{423 conns, 20,443 syns}}$ Medulla ($\text{Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9}$)
  - Medulla $\xrightarrow{\text{2,491 conns, 24,781 syns}}$ Motion Detectors ($\text{T4a-d, T5a-d}$)

### 2. Biological Verification Audit (`outputs/audit/biological_audit_report.md`)
Every connection in the circuit is cataloged with:
- Measured synapse count in the male connectome.
- Consensus neurotransmitter:
  - **Mi1**: Acetylcholine ($+1$ Excitatory)
  - **Mi9**: Glutamate ($-1$ Inhibitory via GluClα on T4)
  - **Mi4**: GABA ($-1$ Inhibitory via GABA_A on T4)
  - **Tm1, Tm2, Tm4, Tm9**: Acetylcholine (with Tm9 providing slow/delayed excitation)
- Published neuroscience literature citations (e.g. Strother et al. 2017, Arenz et al. 2017, Fisher et al. 2015, Gruntman et al. 2018).

---

## Setup & Running

```powershell
# Clone the repository
git clone https://github.com/Siddz-17/Fly.git
cd Fly

# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest tests/ -v

# Re-run circuit extraction
python connectome/extract_mvp_a.py

# Re-generate biological audit
python connectome/biological_audit.py
```

---

## Scientific Provenance Taxonomy
Every parameter and entity belongs to one of five explicit classes:
1. `[REAL CONNECTOME]`: Body IDs, neuron types, measured synapse counts ($N$), spatial column assignments, consensus neurotransmitters.
2. `[BIOLOGICALLY INFERRED]`: Neurotransmitter polarity (ACh = $+$, GABA = $-$, Glu on T4 = $-$ via GluClα), membrane time constants ($\tau_m$), synaptic delays ($\tau_d$).
3. `[ENGINEERED FROM REAL CONNECTOME]`: Computational weights calculated from synapse counts: $W = \text{sign}(NT) \cdot f(N)$.
4. `[ENGINEERED INTERFACE]`: Ommatidial sampling grids, rendering frames, time discretization ($dt$).
5. `[LEARNED]`: PPO behavioral policy weights, YOLOv5 detection weights.
6. `[SYNTHETIC TEST STIMULI]`: Controlled drifting bars, gratings, and flashes used as validation inputs.

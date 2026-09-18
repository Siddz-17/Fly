# Connectome Engineering & Biological Grounding Rules

These rules govern all computational neuroscience, connectome modeling, and biological agent tasks in this repository.

## 1. Central Scientific Hypothesis & Falsifiability
- Never assume a connectome circuit will automatically produce a desired computation (e.g. direction selectivity).
- Frame all experiments around explicit, falsifiable hypotheses:
  > **H1: A computational model constrained by real male Drosophila optic-lobe connectivity produces measurable direction-selective responses to visual motion, and these responses differ from topology/weight-shuffled controls.**
- If a biological computation does not emerge, that is a legitimate, publishable finding: investigate and document why rather than artificially tuning weights to force a preconceived outcome.

## 2. No Arbitrary Metric Thresholds
- Empirical quantities (e.g., Direction Selectivity Index DSI, velocity tuning, firing rates) must be measured and reported as empirical readouts.
- Never hardcode arbitrary pass/fail gates (e.g., `DSI >= 0.3`).
- Always benchmark real connectome responses against formal control baselines:
  - Real Connectome
  - Weight-Shuffled Baseline (preserves topology, randomizes weights)
  - Topology-Scrambled Baseline (degree-matched configuration / random graph)
  - Lesioned Circuit (selective ablation of specific subtypes like Mi9 or Mi4)

## 3. Strict 5-Tier Provenance Taxonomy
Every variable, parameter, and component in the codebase must explicitly declare its provenance tier:
1. `[REAL CONNECTOME]`: Body IDs, neuron types, verified synapse counts, anatomical columnar ROIs, consensus neurotransmitter predictions.
2. `[BIOLOGICALLY INFERRED]`: Neurotransmitter polarity (ACh = $+$, GABA = $-$, Glutamate on T4 = $-$ via GluClα), membrane time constants ($\tau_m$), synaptic delay kinetics.
3. `[ENGINEERED FROM REAL CONNECTOME]`: Computational weights calculated from synapse counts: $W = \text{sign}(NT) \cdot f(N)$. Synapse count $N$ is real; the functional mapping $f(N)$ is an engineered model.
4. `[ENGINEERED INTERFACE]`: Ommatidial sampling grids, rendering frames, time discretization ($dt$), artificial current injection.
5. `[LEARNED]`: Parameters adjusted via machine learning (e.g., PPO behavioral policy, YOLO weights).
6. `[SYNTHETIC TEST STIMULI]`: Controlled drifting bars, gratings, flashes used strictly as validation inputs.

## 4. Weight-Normalization Ablation Invariant
Whenever evaluating circuit computation, test whether results hold across canonical normalization schemes:
- Raw synapse counts: $W_{ij} \propto N_{ij}$
- Postsynaptic fraction (column-normalized): $W_{ij} = N_{ij} / \sum_k N_{kj}$
- Presynaptic fraction (row-normalized): $W_{ij} = N_{ij} / \sum_k N_{ik}$
- Log-transformed: $W_{ij} \propto \log(1 + N_{ij})$
- Degree-normalized: $W_{ij} = N_{ij} / \sqrt{d_{\text{pre}} \cdot d_{\text{post}}}$

## 5. Staged Incremental MVP Progression
- **MVP-A (Core Motion)**: Lamina $\to$ Medulla $\to$ T4/T5. Validate here first before proceeding!
- **MVP-B (LPTC Downstream)**: Lamina $\to$ Medulla $\to$ T4/T5 $\to$ LPTCs (HS / VS).
- **Phase 2 (Visual World Model)**: Integrate YOLOv5 for semantic object localization.
- **Phase 3 (Embodied Behavior)**: Sensorimotor control in FlyGym / NeuroMechFly via PPO.
- **NEVER** skip to Phase 2 or 3 before MVP-A is verified.

## 6. Mandatory Biological Verification Audit (M1.5)
- Every modeled connection must be logged in a machine-readable audit report specifying:
  - Presynaptic cell type & body IDs
  - Postsynaptic cell type & body IDs
  - Synapse count in male connectome (`optic-lobe:v1.1` / `male-cns:v1.0`)
  - Consensus neurotransmitter
  - Primary literature citation supporting the functional connection
  - Explicit documentation of any biological uncertainties.

## 7. Explicit Physical Sensory Transduction
- Never bypass physical sensory transduction (e.g., never pipe video pixels directly into medulla or lobula interneurons).
- Transduction must follow:
  $$\text{Stimulus / Photons} \longrightarrow \text{Ommatidia} \longrightarrow \text{Photoreceptors (R1–R6)} \longrightarrow \text{Lamina Monopolar Cells (L1–L3)} \longrightarrow \text{Medulla (Mi/Tm)}$$

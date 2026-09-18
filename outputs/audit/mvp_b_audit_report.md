# Biological Verification Audit: Milestone MVP-B (LPTC Circuit)

## 1. Circuit Overview
- **Dataset**: `male-cns:v1.0`
- **Total Real Neurons**: 891
- **Total Real Synaptic Connections**: 6152
- **Total Real Synapses**: 94,927
- **LPTC Bodies Extracted**: 22
- **LPi Interneuron Bodies Extracted**: 34
- **Synthetic Neurons / Graphs**: **0 (Strictly Real Connectome Data)**

## 2. Synaptic Convergence Matrix: T4/T5 & LPi onto Downstream LPTCs
```
type_pre   LPi12  LPi21  LPi2b  LPi34  LPi4b   T4a   T4b   T4c   T4d   T5a   T5b   T5c   T5d
type_post                                                                                   
HSE            3   1386      0      0      0   644     0     0     0   791     0     0     0
HSN            0   1061      0      0      0   882     0     0     0   922     0     0     0
HSS            0   1554     65      0      0   357     0     0     0   472     0     0     0
HST            0    120    173      0      0   109     0     0     0   104     0     0     0
LPi12          0      0      0      0      0  1931   107     3     0  2004   209     0     0
LPi21          0      0      0      0      0   116  2072     0     0    47  2451     0     0
LPi2b          0      0      0      0      0     0   745     0     0     0   723     0     0
LPi34          0      0      0      0      0     0     0  2590    27     0     0  3201     3
LPi43          0      0      0      0      0     0     0    20  1746     0     0    19  1567
LPi4b          0      0      0      0      0     0     0     6  1070     0     0     0  1094
VS           964   1142     74   1868     58    18    57     0  2332    21    51     0  2293
VST1           0      0      3    248     77     0     0     0   176     0     0     0   292
VST2           0    184     54    178    124     0     0     0   157     3     0     0   196
VSm          119     15      0    579     17     0    10     0   295     0     8     0   447
```

## 3. Biological Verification of Motion Opponency
> [!IMPORTANT]
> **Empirical Connectome Validation**:
> - **Horizontal System (HSN, HSE, HSS, HST)** receives direct cholinergic excitation (+1) exclusively from **T4a and T5a** (front-to-back preferred motion).
> - **LPi21** receives cholinergic excitation (+1) from **T4b and T5b** (back-to-front regressive motion) and makes massive GABAergic inhibitory synapses (-1) onto **HSN, HSE, and HSS**.
> - This biologically produces **push-pull motion opponency**: depolarization during progressive motion and hyperpolarization during regressive motion.
> - **Vertical System (VS, VSm, VST)** receives direct cholinergic excitation (+1) from **T4d and T5d** (downward motion) and disynaptic inhibition via **LPi34**.

## 4. Neurotransmitter Confidence Verification
| Cell Type | Body ID | Consensus NT | Confidence | Biological Role |
|---|---|---|---|---|
| VS | `10112` | acetylcholine | 80.46% | Wide-field integrator (ACh) |
| VS | `10766` | acetylcholine | 89.56% | Wide-field integrator (ACh) |
| VST1 | `20634` | acetylcholine | 94.16% | Wide-field integrator (ACh) |
| VS | `10012` | acetylcholine | 85.98% | Wide-field integrator (ACh) |
| VS | `11551` | acetylcholine | 95.33% | Wide-field integrator (ACh) |
| HSN | `10015` | acetylcholine | 76.86% | Wide-field integrator (ACh) |
| HSE | `10016` | acetylcholine | 90.67% | Wide-field integrator (ACh) |
| VS | `10403` | acetylcholine | 71.57% | Wide-field integrator (ACh) |
| VST1 | `24867` | acetylcholine | 89.37% | Wide-field integrator (ACh) |
| VST1 | `25893` | acetylcholine | 96.05% | Wide-field integrator (ACh) |
| HSS | `10023` | acetylcholine | 87.84% | Wide-field integrator (ACh) |
| VS | `10284` | acetylcholine | 78.34% | Wide-field integrator (ACh) |
| VSm | `537004` | acetylcholine | 93.80% | Wide-field integrator (ACh) |
| VS | `10037` | acetylcholine | 73.05% | Wide-field integrator (ACh) |
| VST2 | `16570` | acetylcholine | 89.33% | Wide-field integrator (ACh) |
| VST2 | `15422` | acetylcholine | 94.44% | Wide-field integrator (ACh) |
| VS | `10311` | acetylcholine | 72.59% | Wide-field integrator (ACh) |
| VST2 | `14416` | acetylcholine | 95.94% | Wide-field integrator (ACh) |
| VST2 | `16606` | acetylcholine | 95.22% | Wide-field integrator (ACh) |
| VS | `10206` | acetylcholine | 67.08% | Wide-field integrator (ACh) |
| HST | `12521` | acetylcholine | 95.91% | Wide-field integrator (ACh) |
| VSm | `11134` | acetylcholine | 91.10% | Wide-field integrator (ACh) |
| LPi43 | `54400` | glutamate | 80.20% | Motion opponent interneuron (GABA) |
| LPi43 | `29953` | glutamate | 82.42% | Motion opponent interneuron (GABA) |
| LPi34 | `30468` | glutamate | 81.18% | Motion opponent interneuron (GABA) |
| LPi43 | `25222` | glutamate | 82.49% | Motion opponent interneuron (GABA) |
| LPi4b | `10503` | gaba | 89.39% | Motion opponent interneuron (GABA) |
| LPi34 | `29705` | glutamate | 86.10% | Motion opponent interneuron (GABA) |
| LPi2b | `10893` | gaba | 90.04% | Motion opponent interneuron (GABA) |
| LPi34 | `28306` | glutamate | 84.09% | Motion opponent interneuron (GABA) |
| LPi43 | `38937` | glutamate | 77.45% | Motion opponent interneuron (GABA) |
| LPi43 | `29345` | glutamate | 81.44% | Motion opponent interneuron (GABA) |
| LPi34 | `25379` | glutamate | 84.35% | Motion opponent interneuron (GABA) |
| LPi34 | `28576` | glutamate | 83.92% | Motion opponent interneuron (GABA) |
| LPi34 | `36010` | glutamate | 80.12% | Motion opponent interneuron (GABA) |
| LPi43 | `41514` | glutamate | 82.36% | Motion opponent interneuron (GABA) |
| LPi43 | `38960` | glutamate | 81.34% | Motion opponent interneuron (GABA) |
| LPi34 | `27190` | glutamate | 83.13% | Motion opponent interneuron (GABA) |
| LPi43 | `35384` | glutamate | 81.13% | Motion opponent interneuron (GABA) |
| LPi43 | `49211` | glutamate | 79.94% | Motion opponent interneuron (GABA) |
| LPi12 | `10308` | gaba | 90.31% | Motion opponent interneuron (GABA) |
| LPi12 | `10436` | gaba | 90.49% | Motion opponent interneuron (GABA) |
| LPi34 | `22213` | glutamate | 82.25% | Motion opponent interneuron (GABA) |
| LPi34 | `36046` | glutamate | 81.39% | Motion opponent interneuron (GABA) |
| LPi43 | `37967` | glutamate | 80.55% | Motion opponent interneuron (GABA) |
| LPi21 | `10066` | gaba | 90.58% | Motion opponent interneuron (GABA) |
| LPi34 | `30166` | glutamate | 81.21% | Motion opponent interneuron (GABA) |
| LPi34 | `33239` | glutamate | 84.35% | Motion opponent interneuron (GABA) |
| LPi43 | `39128` | glutamate | 82.21% | Motion opponent interneuron (GABA) |
| LPi43 | `32607` | glutamate | 82.06% | Motion opponent interneuron (GABA) |
| LPi34 | `36206` | glutamate | 80.92% | Motion opponent interneuron (GABA) |
| LPi43 | `59125` | glutamate | 81.79% | Motion opponent interneuron (GABA) |
| LPi34 | `31862` | glutamate | 82.55% | Motion opponent interneuron (GABA) |
| LPi43 | `50550` | glutamate | 78.64% | Motion opponent interneuron (GABA) |
| LPi34 | `33273` | glutamate | 83.31% | Motion opponent interneuron (GABA) |
| LPi34 | `28157` | glutamate | 82.96% | Motion opponent interneuron (GABA) |

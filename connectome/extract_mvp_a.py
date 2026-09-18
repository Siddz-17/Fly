"""
connectome/extract_mvp_a.py

Extracts the MVP-A core retinotopic motion-processing circuit from the real
male Drosophila connectome (male-cns:v1.0).

Circuit Pathway:
    Lamina (L1, L2, L3, L4, L5)
        │
        ▼
    Medulla & Transmedulla (Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9)
        │
        ▼
    T4 Subtypes (T4a, T4b, T4c, T4d) & T5 Subtypes (T5a, T5b, T5c, T5d)

Scope:
    Central 25-column retinotopic patch (Hex1 in 17..21, Hex2 in 18..22)
    in the male right optic lobe (somaSide == 'R').

Artifact Produced:
    data/processed/mvp_a_circuit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.feather as feather

from connectome.loader import ConnectomeLoader
from connectome.provenance import RealConnection, RealNeuron, TaxonomyTier

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "mvp_a_circuit.json"


def extract_mvp_a_circuit(
    hex1_range: tuple[int, int] = (17, 21),
    hex2_range: tuple[int, int] = (18, 22),
    min_synapse_weight: int = 3,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    print("[M1 EXTRACTION] Initializing connectome loader...")
    loader = ConnectomeLoader()
    ann_df = loader.load_annotations()
    nt_df = loader.load_neurotransmitters()

    # Pre-build lookup dictionaries
    nt_map = nt_df.set_index("body")[["consensus_nt", "predicted_nt_confidence"]].to_dict("index")

    # Step 1: Select Lamina and Medulla neurons in the central 25-column patch
    lamina_types = ["L1", "L2", "L3", "L4", "L5"]
    medulla_types = ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9"]
    motion_types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

    column_cells = ann_df[
        (ann_df["somaSide"] == "R")
        & (ann_df["assignedOlHex1"].between(hex1_range[0], hex1_range[1]))
        & (ann_df["assignedOlHex2"].between(hex2_range[0], hex2_range[1]))
        & (ann_df["type"].isin(lamina_types + medulla_types))
    ].copy()

    patch_body_ids = set(column_cells["bodyId"].tolist())
    print(f"[M1 EXTRACTION] Found {len(column_cells)} column neurons across 25 columns in right optic lobe.")

    # Step 2: Query connections from Lamina -> Medulla and Medulla -> Medulla/T4/T5
    print("[M1 EXTRACTION] Querying synaptic connections from male connectome weights dataset...")
    weights_dataset = loader.get_weights_dataset()

    # Query connections where pre is in patch
    filter_pre = (ds.field("body_pre").isin(list(patch_body_ids))) & (ds.field("weight") >= min_synapse_weight)
    pre_table = weights_dataset.to_table(filter=filter_pre)
    pre_conns = pre_table.to_pandas()

    # Query connections where post is in patch (to capture intra-patch and feedback connections)
    filter_post = (ds.field("body_post").isin(list(patch_body_ids))) & (ds.field("weight") >= min_synapse_weight)
    post_table = weights_dataset.to_table(filter=filter_post)
    post_conns = post_table.to_pandas()

    all_conns = pd.concat([pre_conns, post_conns], ignore_index=True).drop_duplicates()

    # Map neuron metadata
    body_meta = ann_df.set_index("bodyId")[["type", "assignedOlHex1", "assignedOlHex2", "somaSide"]].to_dict("index")
    all_conns["type_pre"] = all_conns["body_pre"].map(lambda b: body_meta.get(b, {}).get("type"))
    all_conns["type_post"] = all_conns["body_post"].map(lambda b: body_meta.get(b, {}).get("type"))

    # Filter to valid pathways:
    # 1. Lamina -> Medulla
    # 2. Medulla -> Medulla
    # 3. Medulla -> T4/T5
    valid_pre_types = set(lamina_types + medulla_types)
    valid_post_types = set(medulla_types + motion_types)

    circuit_conns = all_conns[
        all_conns["type_pre"].isin(valid_pre_types) & all_conns["type_post"].isin(valid_post_types)
    ].copy()

    # Gather all unique neuron IDs in the extracted circuit
    all_circuit_bodies = set(circuit_conns["body_pre"].tolist()) | set(circuit_conns["body_post"].tolist())
    print(f"[M1 EXTRACTION] Extracted {len(circuit_conns)} verified connections across {len(all_circuit_bodies)} neurons.")

    # Build neuron records
    neuron_records: dict[str, dict[str, Any]] = {}
    for bid in all_circuit_bodies:
        m = body_meta.get(bid, {})
        nt_info = nt_map.get(bid, {})
        neuron_records[str(bid)] = {
            "body_id": int(bid),
            "cell_type": str(m.get("type", "unknown")),
            "soma_side": str(m.get("somaSide", "R")),
            "hex1": float(m["assignedOlHex1"]) if pd.notna(m.get("assignedOlHex1")) else None,
            "hex2": float(m["assignedOlHex2"]) if pd.notna(m.get("assignedOlHex2")) else None,
            "neurotransmitter": nt_info.get("consensus_nt"),
            "nt_confidence": float(nt_info["predicted_nt_confidence"]) if pd.notna(nt_info.get("predicted_nt_confidence")) else None,
            "tier": TaxonomyTier.REAL_CONNECTOME.value,
        }

    # Build connection records
    conn_records: list[dict[str, Any]] = []
    for _, row in circuit_conns.iterrows():
        conn_records.append({
            "pre_id": int(row["body_pre"]),
            "post_id": int(row["body_post"]),
            "pre_type": str(row["type_pre"]),
            "post_type": str(row["type_post"]),
            "synapse_count": int(row["weight"]),
            "dataset": "male-cns:v1.0",
            "tier": TaxonomyTier.REAL_CONNECTOME.value,
        })

    # Summary statistics
    by_pair = circuit_conns.groupby(["type_pre", "type_post"])["weight"].agg(["count", "sum"]).reset_index()
    pair_summary = [
        {
            "pre_type": r["type_pre"],
            "post_type": r["type_post"],
            "connections": int(r["count"]),
            "synapses": int(r["sum"]),
        }
        for _, r in by_pair.iterrows()
    ]

    circuit_data = {
        "metadata": {
            "dataset": "male-cns:v1.0",
            "circuit_name": "MVP-A Core Retinotopic Motion Patch",
            "patch_center": [19, 20],
            "hex1_range": list(hex1_range),
            "hex2_range": list(hex2_range),
            "min_synapse_weight": min_synapse_weight,
            "total_neurons": len(neuron_records),
            "total_connections": len(conn_records),
            "total_synapses": int(circuit_conns["weight"].sum()),
        },
        "pathway_summary": pair_summary,
        "neurons": neuron_records,
        "connections": conn_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(circuit_data, f, indent=2)

    print(f"[M1 EXTRACTION] Saved extracted circuit artifact to {output_path}")
    return circuit_data


if __name__ == "__main__":
    extract_mvp_a_circuit()

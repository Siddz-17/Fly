"""
Pull T4/T5 motion-detector connectivity from the male fly Optic Lobe connectome
via neuPrint, and dump synapse-count weights to CSV for use as correlator weights.

Setup:
    pip install neuprint-python pandas

    Get a token from https://neuprint.janelia.org (log in -> Account -> Auth Token)
    and either paste it below or set it as an environment variable:

        export NEUPRINT_TOKEN="your_token_here"

Usage:
    python fetch_t4_t5_connectivity.py
"""

import os
import sys
import pandas as pd

try:
    from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC
except ImportError:
    sys.exit("Missing dependency. Run: pip install neuprint-python pandas")

# ---------------------------------------------------------------------------
# 1. Connect to the male Optic Lobe dataset
# ---------------------------------------------------------------------------

TOKEN = os.environ.get("NEUPRINT_TOKEN", "PASTE_YOUR_TOKEN_HERE")
SERVER = "neuprint.janelia.org"
DATASET = "optic-lobe:v1.1"  # male fly right optic lobe, v1.1 (2025-03-10 release)

if TOKEN == "PASTE_YOUR_TOKEN_HERE":
    sys.exit(
        "Set your neuPrint token first: either edit TOKEN in this script, "
        "or export NEUPRINT_TOKEN=your_token_here"
    )

client = Client(SERVER, dataset=DATASET, token=TOKEN)
print(f"Connected. Dataset version: {client.fetch_version()}")

# ---------------------------------------------------------------------------
# 2. Fetch T4 and T5 subtype neurons
# ---------------------------------------------------------------------------

T4_TYPES = ["T4a", "T4b", "T4c", "T4d"]
T5_TYPES = ["T5a", "T5b", "T5c", "T5d"]
TARGET_TYPES = T4_TYPES + T5_TYPES

# Known upstream cell types driving T4 (ON edge motion) and T5 (OFF edge motion)
T4_INPUTS = ["Mi1", "Mi4", "Mi9"]
T5_INPUTS = ["Tm1", "Tm2", "Tm4", "Tm9"]
ALL_INPUTS = T4_INPUTS + T5_INPUTS

print(f"\nFetching neuron counts for: {TARGET_TYPES}")
target_neuron_df, _ = fetch_neurons(NC(type=TARGET_TYPES))
print(target_neuron_df.groupby("type").size().rename("count"))

if target_neuron_df.empty:
    sys.exit(
        "No neurons found for these types in this dataset. "
        "Double-check the dataset name/version and cell-type naming — "
        "it may differ slightly (e.g. suffixed with side/hemisphere)."
    )

# Also fetch neuron records for the upstream input types — needed so we can
# label connection sources by type below. (Bug fix: without this, every
# connection's source_type lookup comes back empty and gets filtered out.)
print(f"\nFetching neuron records for input types: {['Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']}")
input_neuron_df, _ = fetch_neurons(NC(type=["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9"]))
print(input_neuron_df.groupby("type").size().rename("count"))

neuron_df = pd.concat([target_neuron_df, input_neuron_df], ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Fetch upstream synaptic connectivity (input -> T4/T5) with weights
# ---------------------------------------------------------------------------

print(f"\nFetching synapse-weighted connections: {ALL_INPUTS} -> {TARGET_TYPES}")

neuron_criteria_src = NC(type=ALL_INPUTS)
neuron_criteria_tgt = NC(type=TARGET_TYPES)

# fetch_adjacencies returns (nodes_df, edges_df).
# nodes_df contains bodyId + type for BOTH the source and target neurons involved
# in the returned edges -- edges_df (bodyId_pre, bodyId_post, roi, weight) only
# has raw body IDs, so nodes_df is what we need for readable type labels.
nodes_df, conn_df = fetch_adjacencies(neuron_criteria_src, neuron_criteria_tgt)

if conn_df.empty:
    sys.exit("No connections found between the specified input types and T4/T5 subtypes.")

# ---------------------------------------------------------------------------
# 4. Aggregate synapse counts by (source_type, target_type) and save
# ---------------------------------------------------------------------------

# Build the type lookup from nodes_df (covers both Mi/Tm sources and T4/T5
# targets) rather than from neuron_df (which only ever held T4/T5 neurons).
type_lookup = nodes_df.set_index("bodyId")["type"].to_dict()
conn_df["source_type"] = conn_df["bodyId_pre"].map(type_lookup)
conn_df["target_type"] = conn_df["bodyId_post"].map(type_lookup)

# Some sources will be the T4/T5 targets themselves if fetch_adjacencies pulled
# both directions; keep only rows where source_type is one of our known inputs.
conn_df = conn_df[conn_df["source_type"].isin(ALL_INPUTS)]

if conn_df.empty:
    sys.exit(
        "Still no rows after type mapping. Print nodes_df['type'].unique() and "
        "conn_df['bodyId_pre'].head() to check whether cell-type names in this "
        "dataset differ from ['Mi1','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9']."
    )

summary = (
    conn_df.groupby(["source_type", "target_type"])["weight"]
    .sum()
    .reset_index()
    .sort_values(["target_type", "weight"], ascending=[True, False])
)

out_path = "t4_t5_synapse_weights.csv"
summary.to_csv(out_path, index=False)

print(f"\nSaved aggregated synapse-count table to {out_path}")
print("\nPreview:")
print(summary.to_string(index=False))

print(
    "\nNext step: normalize each target_type's incoming weights (e.g. divide by "
    "row-group sum) to turn these raw synapse counts into per-input correlator "
    "weights for the connectome-grounded Reichardt correlator."
)
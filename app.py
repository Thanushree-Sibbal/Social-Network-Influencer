"""Gradio app for node influence inference with the bundled GCN checkpoint."""

from pathlib import Path
import math

import gradio as gr
import pandas as pd
import torch
from torch_geometric.data import Data

from model import InfluenceGCN


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pth"
DEVICE = torch.device("cpu")
LABELS = ("Non-Influencer", "Influencer")


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    model = InfluenceGCN().to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:  # torch 2.2 compatibility
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()


def _parse_features(raw):
    """Parse one finite scalar node feature per line or comma-separated value."""
    if raw is None or not str(raw).strip():
        raise gr.Error("Enter at least one node feature.")
    tokens = str(raw).replace(",", "\n").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise gr.Error("Node features must be numbers separated by commas or new lines.") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise gr.Error("Node features must be finite numbers.")
    if len(values) > 5000:
        raise gr.Error("Please classify no more than 5,000 nodes at once.")
    return values


def _parse_edges(raw, node_count):
    """Parse optional zero-based `source,target` rows into an undirected graph."""
    if raw is None or not str(raw).strip():
        return torch.empty((2, 0), dtype=torch.long)
    edges = []
    for line_number, line in enumerate(str(raw).splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = [part.strip() for part in line.replace(",", " ").split()]
        if len(columns) != 2:
            raise gr.Error(f"Edge line {line_number} must contain two node IDs, e.g. 0,1.")
        try:
            source, target = (int(part) for part in columns)
        except ValueError as exc:
            raise gr.Error(f"Edge line {line_number} must use integer node IDs.") from exc
        if not (0 <= source < node_count and 0 <= target < node_count):
            raise gr.Error(f"Edge line {line_number} references an ID outside 0–{node_count - 1}.")
        if source == target:
            raise gr.Error(f"Edge line {line_number} is a self-link; enter links between distinct nodes.")
        edges.extend(((source, target), (target, source)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    # Remove duplicate undirected links while preserving order.
    edges = list(dict.fromkeys(edges))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def predict_influence(node_features, edge_list):
    values = _parse_features(node_features)
    edge_index = _parse_edges(edge_list, len(values))
    graph = Data(
        x=torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
        edge_index=edge_index,
    )
    with torch.inference_mode():
        probabilities = MODEL(graph).exp()
    rows = []
    for index, scores in enumerate(probabilities.tolist()):
        predicted = max(range(len(scores)), key=scores.__getitem__)
        rows.append({
            "Node": index,
            "Node feature": values[index],
            "Prediction": LABELS[predicted],
            "Confidence": f"{scores[predicted] * 100:.1f}%",
        })
    return pd.DataFrame(rows)


with gr.Blocks(title="Social Network Influence Prediction") as demo:
    gr.Markdown(
        "# Social Network Influence Prediction\n"
        "Classify nodes with the bundled Graph Convolutional Network (GCN). "
        "Enter one finite scalar feature per node (signed values are supported). Optionally provide graph links "
        "as zero-based `source,target` pairs, one per line."
    )
    with gr.Row():
        degrees = gr.Textbox(
            label="Node feature values (signed)",
            value="-8, 2, -9, 3",
            lines=3,
            placeholder="-8, 2, -9, 3",
        )
        edges = gr.Textbox(
            label="Optional edges (undirected)",
            value="0,1\n1,2\n1,3",
            lines=4,
            placeholder="0,1\n1,2",
        )
    run = gr.Button("Predict influence", variant="primary")
    results = gr.Dataframe(label="Node predictions", interactive=False, wrap=True)
    run.click(predict_influence, inputs=[degrees, edges], outputs=results)
    gr.Markdown(
        "**Interpretation:** confidence is the selected class probability from this checkpoint. "
        "This is a research/demo model; its training data, evaluation metrics, and calibration "
        "are not included in the supplied project. The shown features are model inputs, not raw graph degrees. "
        "Treat outputs as illustrative, not as "
        "validated claims about real users."
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

# Social Network Influence Prediction

A small Gradio application that runs a two-layer Graph Convolutional Network (GCN) on a user-supplied graph. It predicts one of two classes for each node: `Non-Influencer` or `Influencer`.

## Run locally

Use Python 3.10–3.13 (Python 3.11 is recommended). From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860> in a browser. The included checkpoint is loaded relative to `app.py`, so the app works when launched from any current directory.

## Use the app

- Enter one finite scalar node feature per node, separated by commas or new lines. Signed values are accepted because the bundled checkpoint's saved examples include negative inputs.
- Optionally enter edges as zero-based `source,target` pairs, one per line. Links are treated as undirected and duplicate links are ignored.
- Leave the edge field empty to classify disconnected nodes using the GCN's built-in self-loops.
- Select **Predict influence**. Results include the node index, input feature value, predicted class, and the model's class probability.

For example, the default inputs use node features `-8, 2, -9, 3` and links `0–1`, `1–2`, and `1–3`. With the supplied checkpoint, the sample yields both `Influencer` and `Non-Influencer` predictions.

## Project files

```text
app.py          Gradio interface, validation, and inference
model.py        GCN architecture
model.pth       Supplied PyTorch model weights
requirements.txt Runtime dependencies
```

## Model and data limitations

The archive supplied the model weights and architecture, but no training script, source training dataset, model card, or evaluation results. A saved Gradio feedback CSV contained two negative inputs labeled as `Influencer`; since raw graph degree cannot be negative, the app treats these values as general scalar model features rather than claiming they are literal degrees. The checkpoint loads and produces both classes for the included example, but its prediction quality and confidence calibration cannot be independently verified from the supplied material. Treat predictions as a demonstration only, not as validated assessments of real people.

To make this a research-ready model, provide the original labeled graph dataset and split, training procedure, and evaluation protocol; then retrain and report held-out metrics. Do not use this checkpoint for consequential decisions.

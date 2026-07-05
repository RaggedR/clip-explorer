# CLIP Image Explorer

Processes a directory of images through OpenCLIP (ViT-L/14) to produce embeddings, zero-shot aesthetic scores, and a 2D UMAP projection, then renders the collection as an interactive D3.js scatter plot. Each image gets scored across five dimensions (aesthetic quality, colorfulness, complexity, mood, nature) without any trained probe weights — scores come purely from text-image similarity. The web UI lets you zoom, hover to preview thumbnails, and recolor the plot by any scoring dimension.

## Tech Stack
- Python 3 (PyTorch, open-clip-torch, umap-learn, Pillow, NumPy, tqdm)
- D3.js (interactive scatter plot via `visualize.html`)
- Hardware acceleration: MPS on Apple Silicon, CUDA on NVIDIA

## Setup
```bash
pip install -r requirements.txt

# Embed and score images (first run downloads ~900 MB model)
python embed.py /path/to/images

# Serve the visualization (required — direct file open blocked by CORS)
python -m http.server 8080
# Open http://localhost:8080/visualize.html
```

## License
MIT

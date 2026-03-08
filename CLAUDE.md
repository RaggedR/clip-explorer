# CLIP Image Explorer

> Embed, score, and visually explore an image collection using CLIP zero-shot inference and UMAP projection.

## Overview

This tool processes a directory of images through a CLIP vision model (ViT-L/14) to produce:
1. **768-D embeddings** for each image
2. **Aesthetic scores** (1-10) via zero-shot text-image similarity
3. **Additional dimension scores** (colorfulness, complexity, mood, nature)
4. **2D UMAP projection** for interactive visualization
5. **Thumbnails** for the web UI

The interactive D3.js visualization lets you explore the image collection as a zoomable scatter plot, colored by any scoring dimension.

## Architecture

```
embed.py (Python)                    visualize.html (D3.js)
┌──────────────────────┐             ┌────────────────────────┐
│ Load images          │             │ Load output/data.json  │
│ CLIP encode (ViT-L/14)│            │ Render scatter plot     │
│ Zero-shot scoring    │ → output/ → │ Zoom, hover, filter    │
│ UMAP 2D projection   │   data.json │ Color by dimension     │
│ Generate thumbnails  │   thumbs/   │ Preview on hover       │
└──────────────────────┘             └────────────────────────┘
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Embed and score images (first run downloads ~900MB model)
python embed.py /path/to/images

# Serve visualization
python -m http.server 8080
# Open http://localhost:8080/visualize.html
```

## Key Files

| File | Purpose |
|------|---------|
| `embed.py` | Main pipeline: embedding, scoring, UMAP, thumbnails |
| `visualize.html` | Interactive D3.js scatter plot visualization |
| `requirements.txt` | Python dependencies |
| `output/data.json` | Generated: image metadata, scores, UMAP coordinates |
| `output/thumbs/` | Generated: 96x96 JPEG thumbnails for the web UI |

## Dependencies

- `torch` — PyTorch (uses MPS on Apple Silicon, CUDA on NVIDIA)
- `open-clip-torch` — LAION's maintained CLIP implementation
- `umap-learn` — UMAP dimensionality reduction
- `Pillow` — Image loading and thumbnail generation
- `numpy`, `tqdm` — Utilities

## Scoring Dimensions

All scoring is **zero-shot** — no trained weights beyond the base CLIP model. Each dimension is defined by positive/negative text prompts in `embed.py`. Scores are rescaled to 1-10.

| Dimension | Measures |
|-----------|----------|
| `aesthetic` | Visual quality, composition, appeal |
| `colorfulness` | Vivid/saturated vs monochrome/muted |
| `complexity` | Busy/detailed vs minimal/simple |
| `mood_happy` | Cheerful/warm vs somber/cold |
| `nature` | Outdoor/landscape vs indoor/urban |

Custom dimensions can be added by editing the `DIMENSIONS` dict in `embed.py`.

## Context

This project originated from a discussion about watermark detection for the NFTmarket project (`~/git/NFTmarket/`). The same image dataset (Facebook-scraped photos, external drive) may be used for both watermark robustness testing and CLIP exploration. The CLIP embeddings could also support duplicate/near-duplicate detection for stolen artwork identification.

## Known Limitations

- CLIP model download is ~900MB on first run
- ViT-L/14 needs ~2GB RAM for inference
- Zero-shot aesthetic scores are approximate — a trained linear probe (e.g., LAION aesthetic predictor V2) would be more accurate but requires separate weight files
- UMAP is non-deterministic across runs unless `random_state` is set (currently pinned to 42)
- The visualization requires serving via HTTP (`python -m http.server`) — opening the HTML file directly won't load the JSON data due to CORS

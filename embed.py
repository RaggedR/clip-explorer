#!/usr/bin/env python3
"""
CLIP Image Explorer — Embedding, Scoring & Projection Pipeline

Processes a directory of images:
1. Computes CLIP embeddings (ViT-L/14)
2. Scores aesthetics using zero-shot CLIP text-image similarity
3. Scores additional dimensions (colorfulness, mood, complexity, nature)
4. Computes UMAP 2D projection
5. Generates thumbnails
6. Outputs data.json for the interactive visualization

Usage:
    python embed.py /path/to/images
    python embed.py /path/to/images --output-dir output --batch-size 16
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Zero-shot scoring prompts
# ---------------------------------------------------------------------------

AESTHETIC_POSITIVE = [
    "a visually striking, compelling artwork with strong composition",
    "a masterpiece of visual art, bold and expressive",
    "excellent use of color, form, and technique",
    "a powerful image with artistic vision and originality",
    "skilled craftsmanship, intentional design, visual impact",
]

AESTHETIC_NEGATIVE = [
    "a poorly made, amateurish image lacking skill",
    "an uninteresting, generic image with no artistic merit",
    "sloppy execution, careless composition, no visual impact",
    "a bland, forgettable image with nothing compelling",
    "low effort, poorly composed, visually weak",
]

# Additional scoring dimensions — all zero-shot via CLIP text similarity
DIMENSIONS = {
    "colorfulness": {
        "positive": [
            "a very colorful, vibrant image with vivid saturated colors",
            "bright rainbow of colors, rich hues",
        ],
        "negative": [
            "a monochrome, desaturated, black and white image",
            "muted, dull, washed-out colors",
        ],
    },
    "complexity": {
        "positive": [
            "a complex, detailed image with many visual elements",
            "intricate patterns, dense composition, lots of detail",
        ],
        "negative": [
            "a simple, minimalist image with few elements",
            "plain, empty composition, blank space",
        ],
    },
    "mood_happy": {
        "positive": [
            "a happy, joyful, cheerful, uplifting scene",
            "bright and warm, smiling, celebration",
        ],
        "negative": [
            "a sad, melancholy, gloomy, depressing scene",
            "dark and cold, lonely, somber",
        ],
    },
    "nature": {
        "positive": [
            "a natural landscape, nature scene, outdoors",
            "trees, mountains, water, sky, wildlife, flowers",
        ],
        "negative": [
            "an indoor scene, urban environment, artificial",
            "buildings, streets, technology, furniture",
        ],
    },
    "abstract": {
        "positive": [
            "abstract art, non-representational, geometric shapes and forms",
            "cubism, expressionism, abstract expressionism, pure form and color",
        ],
        "negative": [
            "realistic, representational, photorealistic, figurative art",
            "lifelike depiction, accurate rendering of real objects and people",
        ],
    },
    "modern": {
        "positive": [
            "modern art, contemporary, avant-garde, bold and experimental",
            "pop art, street art, digital art, graphic design",
        ],
        "negative": [
            "classical art, traditional, old master painting, Renaissance",
            "academic art, Baroque, Rococo, Neoclassical",
        ],
    },
}


# ---------------------------------------------------------------------------
# CLIP model loading
# ---------------------------------------------------------------------------

def load_clip_model(model_name="ViT-L-14", pretrained="openai", device="cpu"):
    """Load CLIP model and preprocessing transforms."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def find_images(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all image files in a directory (optionally recursive)."""
    images = []

    if recursive:
        for root, _, files in os.walk(directory):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(p)
    else:
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)

    return sorted(set(images))


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings(
    model, preprocess, image_paths: List[Path], device: str, batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """Compute CLIP image embeddings in batches. Returns (embeddings, metadata)."""
    all_embeddings = []
    metadata = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        batch_meta = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                batch_tensors.append(preprocess(img))
                batch_meta.append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "width": w,
                        "height": h,
                        "size_kb": path.stat().st_size // 1024,
                    }
                )
            except Exception as e:
                print(f"  Skipping {path.name}: {e}")
                continue

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize

        all_embeddings.append(emb.cpu().float().numpy())
        metadata.extend(batch_meta)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, metadata


# ---------------------------------------------------------------------------
# Zero-shot scoring
# ---------------------------------------------------------------------------

def score_zero_shot(
    model,
    tokenizer,
    embeddings: np.ndarray,
    positive_prompts: List[str],
    negative_prompts: List[str],
    device: str,
) -> np.ndarray:
    """
    Score images using zero-shot CLIP text-image similarity.

    For each image, computes:
        score = cosine_sim(image, mean(positive_prompts))
              - cosine_sim(image, mean(negative_prompts))

    Returns scores rescaled to 1–10 range.
    """
    pos_tokens = tokenizer(positive_prompts).to(device)
    neg_tokens = tokenizer(negative_prompts).to(device)

    with torch.no_grad():
        pos_emb = model.encode_text(pos_tokens)
        neg_emb = model.encode_text(neg_tokens)
        pos_emb = pos_emb / pos_emb.norm(dim=-1, keepdim=True)
        neg_emb = neg_emb / neg_emb.norm(dim=-1, keepdim=True)

    pos_mean = pos_emb.mean(dim=0, keepdim=True).cpu().float().numpy()
    neg_mean = neg_emb.mean(dim=0, keepdim=True).cpu().float().numpy()

    # Cosine similarities (embeddings are already L2-normalized)
    pos_sim = (embeddings @ pos_mean.T).squeeze()
    neg_sim = (embeddings @ neg_mean.T).squeeze()

    raw = pos_sim - neg_sim

    # Rescale to 1–10 using percentile-based normalization
    p5 = np.percentile(raw, 5)
    p95 = np.percentile(raw, 95)
    scores = 1 + 9 * (raw - p5) / (p95 - p5 + 1e-8)
    scores = np.clip(scores, 1.0, 10.0)

    return scores


# ---------------------------------------------------------------------------
# UMAP projection
# ---------------------------------------------------------------------------

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Compute 2D UMAP projection from CLIP embeddings."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        n_components=2,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def generate_thumbnails(
    metadata: List[Dict], output_dir: Path, thumb_size: int = 96
) -> List[Dict]:
    """Generate small JPEG thumbnails for the web visualization."""
    thumbs_dir = output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    for i, meta in enumerate(
        tqdm(metadata, desc="Generating thumbnails")
    ):
        try:
            img = Image.open(meta["path"]).convert("RGB")
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            thumb_name = f"{i:05d}.jpg"
            img.save(thumbs_dir / thumb_name, "JPEG", quality=80)
            meta["thumb"] = f"thumbs/{thumb_name}"
        except Exception as e:
            print(f"  Thumbnail failed for {meta['filename']}: {e}")
            meta["thumb"] = None

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLIP Image Explorer — embed, score, and project an image collection"
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP inference")
    parser.add_argument("--thumb-size", type=int, default=96, help="Thumbnail size in pixels")
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14",
        help="CLIP model (default: ViT-L-14)",
    )
    parser.add_argument(
        "--no-extra-dimensions",
        action="store_true",
        help="Skip extra scoring dimensions (only compute aesthetics)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories for images",
    )
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Find images ──────────────────────────────────────────────────────
    print(f"\nScanning {args.image_dir}...")
    image_paths = find_images(args.image_dir, recursive=not args.no_recursive)
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("No images found!")
        return

    # ── Load CLIP ────────────────────────────────────────────────────────
    print(f"\nLoading CLIP model ({args.model})...")
    model, preprocess, tokenizer = load_clip_model(args.model, device=device)
    print("Model loaded.")

    # ── Compute embeddings (with caching) ────────────────────────────────
    cache_path = args.output_dir / "embeddings.npy"
    meta_cache_path = args.output_dir / "metadata_cache.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and meta_cache_path.exists():
        print(f"\nLoading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
        with open(meta_cache_path) as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} cached embeddings → shape {embeddings.shape}")
    else:
        print(f"\nComputing CLIP embeddings for {len(image_paths)} images...")
        embeddings, metadata = compute_embeddings(
            model, preprocess, image_paths, device, args.batch_size
        )
        np.save(cache_path, embeddings)
        with open(meta_cache_path, "w") as f:
            json.dump(metadata, f)
        print(f"Embedded {len(metadata)} images → shape {embeddings.shape}")
        print(f"Cached to {cache_path}")

    # ── Aesthetic scoring ────────────────────────────────────────────────
    print("\nScoring aesthetics (zero-shot)...")
    aesthetic_scores = score_zero_shot(
        model, tokenizer, embeddings, AESTHETIC_POSITIVE, AESTHETIC_NEGATIVE, device
    )
    for i, meta in enumerate(metadata):
        meta["aesthetic"] = round(float(aesthetic_scores[i]), 2)

    # ── Extra dimensions ─────────────────────────────────────────────────
    dimension_names = ["aesthetic"]

    if not args.no_extra_dimensions:
        print("\nScoring extra dimensions...")
        for dim_name, prompts in DIMENSIONS.items():
            scores = score_zero_shot(
                model, tokenizer, embeddings,
                prompts["positive"], prompts["negative"], device,
            )
            for i, meta in enumerate(metadata):
                meta[dim_name] = round(float(scores[i]), 2)
            dimension_names.append(dim_name)
            print(f"  {dim_name}: done")

    # ── UMAP ─────────────────────────────────────────────────────────────
    print("\nComputing UMAP 2D projection...")
    projection = compute_umap(embeddings)
    for i, meta in enumerate(metadata):
        meta["x"] = round(float(projection[i, 0]), 4)
        meta["y"] = round(float(projection[i, 1]), 4)
    print("UMAP done.")

    # ── Clustering ────────────────────────────────────────────────────────
    print("\nClustering embeddings (HDBSCAN)...")
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(min_cluster_size=30, min_samples=10)
    cluster_labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels) - {-1})
    n_noise = (cluster_labels == -1).sum()
    print(f"  {n_clusters} clusters found, {n_noise} noise points")

    for i, meta in enumerate(metadata):
        meta["cluster"] = int(cluster_labels[i])
    dimension_names.append("cluster")

    # ── Thumbnails ───────────────────────────────────────────────────────
    print("\nGenerating thumbnails...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = generate_thumbnails(metadata, args.output_dir, args.thumb_size)

    # ── Save results ─────────────────────────────────────────────────────
    output_path = args.output_dir / "data.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "images": metadata,
                "model": args.model,
                "n_images": len(metadata),
                "dimensions": dimension_names,
            },
            f,
            indent=2,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  {len(metadata)} images processed.")
    print(f"  Output: {output_path}")
    print(f"  Dimensions: {', '.join(dimension_names)}")
    print(f"{'=' * 55}")

    ranked = sorted(metadata, key=lambda m: m["aesthetic"], reverse=True)
    print("\nTop 5 aesthetic scores:")
    for m in ranked[:5]:
        print(f"  {m['aesthetic']:5.1f}  {m['filename']}")
    print("\nBottom 5:")
    for m in ranked[-5:]:
        print(f"  {m['aesthetic']:5.1f}  {m['filename']}")

    print(f"\nNext: open visualize.html in a browser (serve with a local server).")
    print(f"  cd {args.output_dir.parent} && python -m http.server 8080")


if __name__ == "__main__":
    main()

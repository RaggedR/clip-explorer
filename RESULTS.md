# Experiment Results: Human vs CLIP Aesthetic Agreement

## Methodology

### Image Dataset
- Source: Art images from the Tindart collection (user-submitted artwork via Firebase Storage)
- Pool: 3,200+ images processed through the CLIP pipeline

### CLIP Scoring
- Model: OpenCLIP ViT-L/14 (pretrained on OpenAI's CLIP dataset)
- Scoring method: Zero-shot text-image similarity using 5 positive and 5 negative aesthetic prompts
- Positive prompts emphasize composition, expressiveness, color use, originality, and visual impact
- Negative prompts describe amateurish, generic, sloppy, bland, or low-effort imagery
- Scores rescaled to 1-10 range via min-max normalization across the dataset

### Human Evaluation Protocol
- Tool: `compare.html` — a pairwise comparison interface ("Which image is more beautiful?")
- Image selection: Top 5% (score = 10) vs bottom 5% (score = 1) by CLIP aesthetic score
- Pool sizes: 160 images in each pool (top and bottom)
- Pair construction: One image randomly drawn from each pool per round
- Bias control: Left/right placement randomized per pair (coin flip)
- Input: Arrow keys or click to select; spacebar to skip
- Persistence: State saved to browser localStorage, resumable across sessions

### What This Tests
Whether CLIP's zero-shot aesthetic scoring (using generic text prompts about visual quality) aligns with a single human's subjective taste when comparing extreme cases — images CLIP is most and least confident about.

## Results

| Metric | Value |
|--------|-------|
| Total comparisons | 160 |
| Human agreed with CLIP | 114 |
| Human disagreed | 46 |
| **Agreement rate** | **71.2%** |
| Skipped | 0 |

### Interpretation

- **Baseline (chance):** 50% — random guessing between two images
- **Observed:** 71.2% — CLIP captures a real aesthetic signal, but diverges from personal taste ~29% of the time
- **Note:** This test was maximally generous to CLIP — it compared only the extremes (top 5% vs bottom 5%). Agreement would likely be lower for images closer to the median aesthetic score

### Recovered Data

The raw comparison data (160 rounds with filenames, scores, and human choices) was recovered from Chrome's localStorage LevelDB database and saved to `/tmp/clip-comparison-recovered.json`. The experiment was originally run via `python -m http.server 8080` serving the local `compare.html` interface.

## Date

Experiment conducted circa March 2025.

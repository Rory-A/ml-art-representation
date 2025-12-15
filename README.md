# ML Art Representation

This repository explores how a vision–language model’s image encoder (CLIP) captures artistic style by building embeddings for paintings, running k-NN style classification and visualizing results.

## Reports
- `reports/Do CLIP image embeddings reflect artistic style in an unsupervised setting.pdf`
- `reports/(Follow up) Do CLIP style predictions extend to unseen artists.pdf`

## What you can find
- `notebooks/paintings-by-style-plot.ipynb`: main analysis. Loads the style dataset, computes CLIP embeddings, runs k-NN baselines (full vs unseen-artist splits), and plots/exports comparisons.
- `scripts/generate_paintings_csv.py`: cleans filenames in `data/paintings/` to produce `data/paintings.csv`, moving unusable files to `data/unused-paintings/`.
- `scripts/generate_paintings_by_style_csv.py`: builds `data/paintings-by-style.csv` from a `style/artist/image` directory tree.
- `outputs/`: saved figures and CSV results from the notebook runs (e.g., `style_generalisation_comparison.csv`, `style_accuracy_overall.png`). Regenerate by re-running the notebook cells.
- `requirements.txt`: minimal environment for running the scripts and notebook.

## How to run locally
- Install deps: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Prepare metadata: run `python scripts/generate_paintings_csv.py` or `python scripts/generate_paintings_by_style_csv.py` depending on which dataset you have under `data/`.
- Open `notebooks/paintings-by-style-plot.ipynb` and run all cells (GPU optional; CPU works but is slower).
- Outputs write to `outputs/` by default; adjust paths in the notebook if you prefer a different directory.

## Notes for reviewers
- Results demonstrate simple, interpretable baselines (k-NN over CLIP embeddings) with both in-distribution and unseen-artist evaluations.
- Filenames are parsed to recover titles/years; edge cases are handled conservatively, with unparseable files quarantined in `data/unused-paintings/`.
- Visualizations include thumbnail scatter plots and per-style accuracy comparisons to make qualitative inspection easy.

# Surya Embedding Regression

Predicting solar physical quantities from patch-level embeddings of the **Surya Foundation Model** using MLP, Vanilla CNN, and Spatial ResNet architectures.

## Overview

The [Surya Foundation Model](https://arxiv.org/abs/2407.12816) produces 1280-dimensional embedding vectors for each 16×16-pixel patch of a solar image. This notebook investigates how much physical information about the Sun is encoded in those embeddings, and trains neural networks to decode it.

Each patch is described by:
- A **1280-dim embedding** from the Surya Foundation Model
- **AIA 193 Å intensity** — coronal EUV emission
- **AIA 304 Å intensity** — chromospheric EUV emission
- **HMI line-of-sight magnetic flux** (signed and unsigned)
- **Filament covering fraction** — fraction of patch pixels occupied by a filament
- **Helioprojective coordinates** (patch centre, arcsec)

The dataset contains **~32,000 on-disk patches** from a single solar observation (2012-01-05).

---

## Notebook: `embedding_modality_correlation.ipynb`

### Structure

| Section | Description |
|---|---|
| **1. Data loading** | Load `patch_data_all_modalities.npz`; inspect shapes, NaN counts, value ranges |
| **2. Exploratory distributions** | Log-scale histograms of all physical quantities |
| **3. PCA on embeddings** | Scree plot; top-20 principal components |
| **4. Correlation analysis** | Pearson and Spearman *r* between each PCA component and each physical quantity; heatmaps and ranking bar chart |
| **5. MLP baseline** | 4-target MLP (1280 → 512 → 256 → 128 → 4); trained to predict AIA 193, AIA 304, Mag flux, \|Mag flux\|; loss curves, scatter plots, residual plots, spatial maps |
| **6. Stratified MLP evaluation** | Metrics and scatter plots split by filament covering fraction: frac = 0 / frac > 0 / frac > 0.8 |
| **7. Spatial grid reconstruction** | Recover the 202×202 patch grid from helioprojective coordinates (nearest-neighbour step ≈ 9.6 arcsec) |
| **8. Vanilla CNN** | Three 3×3 conv layers over a 5×5 patch neighbourhood → global average pool → 4 outputs; exploits spatial correlations in the embedding field |
| **9. Spatial ResNet** | 1×1 stem (1280 → 128 channels) + 6 residual blocks (3×3 convs with skip connections) → global pool → 4 outputs |
| **10. CNN/ResNet evaluation** | Loss curves, scatter plots, stratified metrics, comparison table vs MLP |
| **11. CNN spatial maps** | True vs predicted maps for all patches (including training) and test-only, for all 4 targets |

### Key results

| Target | MLP R² | Vanilla CNN R² | ResNet R² |
|---|---|---|---|
| AIA 193 Å | 0.679 | **0.943** | 0.925 |
| AIA 304 Å | 0.408 | **0.738** | 0.743 |
| Mag flux (signed) | 0.031 | 0.171 | 0.088 |
| \|Mag flux\| | 0.123 | 0.260 | 0.200 |

Exploiting the spatial structure of the embedding field (5×5 patch neighbourhood) gives a **+0.26 R² gain on AIA 193** and **+0.33 on AIA 304** over the patch-level MLP. Magnetic flux prediction remains difficult — the Surya embeddings encode little signed magnetic field information — but still improves with spatial context.

**Filament patches** (frac > 0) are harder to predict with the MLP (AIA 193 R² drops to 0.34) but the CNN maintains R² ≈ 0.93 even for high-frac patches, showing that spatial context mitigates the class-imbalance problem.

### Residual structure

The linear negative-slope structure visible in residual plots (predicted − true vs true) is a signature of **regression toward the mean**: the slope equals *r* − 1, where *r* is the Pearson correlation. A steeper slope (Mag flux, AIA 304) signals lower predictability from the embeddings, not a model defect.

---

## Data

The input file `patch_data_all_modalities.npz` (not included in this repo) contains the following arrays:

| Key | Shape | Description |
|---|---|---|
| `embeddings` | (32149, 1280) | Surya Foundation Model patch embeddings |
| `aia193_intensity` | (32149,) | Mean AIA 193 Å intensity per patch [DN/s] |
| `aia304_intensity` | (32149,) | Mean AIA 304 Å intensity per patch [DN/s] |
| `magnetic_flux` | (32149,) | Mean HMI LOS magnetic flux per patch [Mx] |
| `filament_covering` | (32149,) | Filament covering fraction [0–1] |
| `disk_fraction` | (32149,) | Fraction of patch on the solar disk [0–1] |
| `patch_x` | (32149,) | Helioprojective longitude of patch centre [arcsec] |
| `patch_y` | (32149,) | Helioprojective latitude of patch centre [arcsec] |
| `patch_size` | scalar | Patch side length in pixels (16) |
| `H_patch` | scalar | Patch grid height (256) |
| `W_patch` | scalar | Patch grid width (256) |

---

## Environment

```bash
conda create -n ml python=3.12
conda activate ml
pip install numpy scipy pandas matplotlib scikit-learn torch astropy
pip install jupyter nbformat nbclient
```

---

## Reference

Surya Foundation Model: [Yi et al. 2024, arXiv:2407.12816](https://arxiv.org/abs/2407.12816)

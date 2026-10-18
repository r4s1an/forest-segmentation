# Forest segmentation on Sentinel-2 (Lankaran region, Azerbaijan)

A U-Net (ResNet34 encoder, 4 spectral bands) that labels every 10 m pixel of a Sentinel-2 scene as
forest or non-forest, trained against ESA WorldCover 2021 "tree cover". The predicted mask is
vectorized into polygons with areas in hectares. It is a time-boxed portfolio experiment built only
on free tools (Kaggle T4 GPU, Microsoft Planetary Computer, MLflow, Optuna).

The emphasis of this repository is **honest evaluation**: a three-way geographic split with buffer
zones, a test block that was evaluated exactly once, an all-forest baseline reported next to every
score, a seed check on the hyperparameter search, and a list of everything that limits the result.

## Headline results

| Split | Evaluation method | Forest IoU | Non-forest IoU | mIoU | All-forest baseline mIoU |
|---|---|---|---|---|---|
| Validation (used for checkpoint and hyperparameter selection, so optimistic) | 256 px tiles, 30 overlapping tiles | 0.851 | 0.721 | 0.786 | 0.350 |
| **Test (evaluated once, never used for any decision)** | block-wise sliding window | **0.937** | **0.594** | **0.766** | 0.446 |

- The model clearly beats the trivial "everything is forest" baseline on both splits (test mIoU
  0.766 vs 0.446).
- The test block is 89% forest, so its forest IoU is inflated by class imbalance: a constant
  "forest" prediction already scores 0.892. The informative numbers there are **non-forest IoU and
  mIoU**, and non-forest IoU rests on only about 960 ha of non-forest area (10.8% of the block).
- The validation-to-test drop (mIoU 0.786 to 0.766) is within the seed-to-seed spread measured below
  (std 0.024), and non-forest IoU falls partly because that class shrinks from 30% of the validation
  block to 11% of the test block.
- Hyperparameter tuning gave **no measurable gain**: tuned and default settings both average 0.78
  mIoU over three fresh seeds.

## Key findings

1. **Dense forest is segmented almost perfectly; the mixed farmland-and-village strip is hard.**
   Validation error is about 4-5% in forest-interior tile rows and 22% in the northernmost strip,
   where only 46% of the label is forest.
2. **The final model under-predicts forest** in held-out areas (validation false negatives 8.5% of
   pixels vs false positives 2.3%).
3. **The model is insensitive to the hyperparameters searched.** 20 Optuna trials spanned only
   0.761-0.808 validation mIoU, and 17 of them landed within 0.02 of each other. Data size and label
   quality are the more likely limits.
4. **Total forest area matches WorldCover to the hectare (39,513 ha) only because errors cancel**
   across regions. Area agreement is not accuracy.
5. **The best checkpoint came from epoch 1 of 15, and training dynamics were not diagnosed before
   the test was run.** See [Open issue](#open-issue-training-dynamics).

## Data and labels

| Item | Value |
|---|---|
| Imagery | Sentinel-2 L2A, June-August 2021, scene cloud cover < 20% (143 scenes), per-pixel median composite |
| Bands | B04 (red), B03 (green), B02 (blue), B08 (near-infrared), digital numbers divided by 10,000 |
| Grid | 10 m, EPSG:32639 (UTM 39N), 2232 x 2276 pixels (about 22 x 23 km) |
| Area of interest | bbox [48.55, 38.60, 48.80, 38.80], Lankaran lowlands and Talysh forest foothills |
| Labels | ESA WorldCover 2021, class 10 ("Tree cover") = forest, everything else = non-forest, resampled onto the Sentinel-2 grid with nearest neighbour |
| Forest share | 77.8% of the scene (3,951,300 forest pixels, 39,513 ha) |

The two rasters were verified to share transform, shape and CRS (`gdalinfo`, `rasterio`).

**Label caveat:** WorldCover is itself a model-derived product, partly built from Sentinel-2, so the
network learns to imitate it rather than to match ground truth, and agreement may be partly circular.

## Experimental design

```
            cols 0 ... 1100  |  256 px gap  |  cols 1356 ... 2232
rows 0 .. 1000    TRAIN      |              |   VALIDATION
rows 1000..1256   TRAIN      |              |   (256 px buffer, unused)
rows 1256..2276   TRAIN      |              |   TEST
```

| Block | Pixels (rows x cols) | Label forest share | Used for |
|---|---|---|---|
| Train | 2276 x 1100 | 0.775 | Fitting the weights (112 tiles of 256 x 256, stride 128) |
| Validation | 1000 x 876 | 0.700 | Per-epoch monitoring, checkpoint choice, hyperparameter search (30 tiles, stride 128) |
| Test | 1020 x 876 | 0.892 | One final evaluation |

- A **256 px (2.56 km) buffer** separates every pair of blocks. Tiles never cross a block edge. Both
  properties are enforced by unit tests.
- Input bands are normalized with **training-block statistics only**.
- The test block was **not even tiled** during development. It was predicted once, after the final
  model was chosen, and no model, threshold or split was changed afterwards.
- Buffers reduce but do not remove spatial autocorrelation, so this test is weaker than an
  independent region would be (see [Limitations](#limitations)).

## Model and training

| Item | Value |
|---|---|
| Architecture | U-Net, ResNet34 encoder with ImageNet weights, 4 input channels, 1 output channel (`segmentation_models_pytorch`) |
| Loss | Dice + `bce_w` x binary cross-entropy |
| Optimizer | AdamW, constant learning rate (no schedule) |
| Augmentation | random 90-degree rotations and horizontal flips |
| Final settings | lr 7.85e-4, weight decay 1.32e-4, `bce_w` 1.255, batch size 8, 15 epochs, seed 0 |
| Checkpoint | best validation mIoU, reached at epoch 1 (the second epoch, counting from 0) |
| Compute | one epoch takes about 3-6 s on a Kaggle T4 |
| Inference | threshold 0.5; block prediction uses 256 px windows at stride 128 with averaged probabilities |

## Results in detail

### Validation (final model, best epoch)

| Metric | Value |
|---|---|
| Forest IoU / non-forest IoU / mIoU | 0.851 / 0.721 / 0.786 |
| Forest Dice / pixel accuracy | 0.920 / 0.893 |
| False positive / false negative pixels | 2.26% / 8.48% |
| Non-forest recall / precision | 0.925 / 0.766 |
| All-forest baseline (accuracy = forest IoU) | 0.700 |

### Test (evaluated once)

| Metric | Model | All-forest baseline |
|---|---|---|
| Forest IoU | 0.937 | 0.892 |
| Non-forest IoU | 0.594 | 0.000 |
| mIoU | 0.766 | 0.446 |
| Non-forest share of block | 10.8% (about 960 ha) | |

Test inference covers the whole 1020 x 876 block (about 8,935 ha) with 42 overlapping windows and
scores every non-nodata pixel. The test set is therefore **one contiguous block, not many
independent samples**, and the non-forest IoU is a noisy estimate (a different 960 ha of non-forest
could move it by several points).

**Validation and test are not computed identically.** Validation during training uses discrete tiles
without overlap averaging, while the test uses block-wise sliding-window inference with averaged
overlaps (the same function that produces the forest map). Averaging overlapping predictions acts
like a small ensemble and can help slightly. As a rough check, evaluating the validation block with
the sliding-window function (windows allowed to read neighbouring pixels) gives forest IoU 0.830 and
non-forest IoU 0.755 (mIoU about 0.79), against 0.851 / 0.721 (0.786) from the tile-based
evaluation, so the method moves per-class scores by 0.02-0.03 but barely moves mIoU.

### Hyperparameter search (Optuna, tracked in MLflow)

TPE sampler (seed 0), 20 trials of 8 epochs each, optimizing validation mIoU. Search space: encoder
{resnet18, resnet34}, learning rate 1e-4 to 1e-3 (log), weight decay 1e-5 to 1e-2 (log), `bce_w`
0.25 to 2.0, batch size {8, 16}. Trial 0 was the default configuration (resnet34, lr 3e-4,
weight decay 1e-2, `bce_w` 1.0, batch size 16).

| | Validation mIoU |
|---|---|
| Default configuration (trial 0) | 0.798 |
| Best trial (trial 15) | 0.808 |
| Median / min / max of 20 trials | 0.795 / 0.761 / 0.808 |

The best of 20 noisy scores is optimistically biased (the "winner's curse"), which is why the seed
check below matters.

### Seed check

Default and tuned settings, 8 epochs, three fresh seeds each (validation mIoU, population std):

| | Seed 1 | Seed 2 | Seed 3 | Mean | Std |
|---|---|---|---|---|---|
| Default | 0.803 | 0.748 | 0.794 | 0.781 | 0.024 |
| Tuned | 0.781 | 0.772 | 0.788 | 0.780 | 0.007 |

The means are identical, so **the tuning gain is zero within seed noise**. The tuned runs vary less
(std 0.007 vs 0.024), but with three runs that spread is itself uncertain. Differences below roughly
0.04 mIoU should be treated as noise on this validation set.

### Run-to-run variation across sessions

The whole pipeline ran twice on the same split and data (the first session's model was lost when
that session ended, and the test was evaluated only on the second). Same configuration and seed
gave different results:

| | Session 1 | Session 2 (reported) |
|---|---|---|
| Trial 1 (lr 4.0e-4, batch 8) validation mIoU | 0.811 | 0.798 |
| Best trial in the search | 0.811 | 0.808 |
| Seed check, tuned config (mean / std) | 0.798 / 0.009 | 0.780 / 0.007 |
| Seed check, default config (mean / std) | 0.781 / 0.034 | 0.781 / 0.024 |
| Final model validation mIoU (best epoch) | 0.815 (epoch 4) | 0.786 (epoch 1) |

GPU training is not bit-reproducible even with fixed seeds, and on 30 overlapping validation tiles
the same configuration can move by 0.01-0.03 mIoU. Only the second session's numbers are used in this
README, and no result was picked from the better of the two sessions.

### Error analysis (validation block)

Error by position, from the 30 validation tiles (6 tile rows x 5 tile columns, stride 128):

| Tile row (scene rows) | Label forest | Predicted forest | Error | False positive | False negative |
|---|---|---|---|---|---|
| 0 (0-256) | 0.46 | 0.33 | 0.218 | 0.043 | 0.175 |
| 1 (128-384) | 0.60 | 0.51 | 0.162 | 0.037 | 0.125 |
| 2 (256-512) | 0.72 | 0.68 | 0.081 | 0.019 | 0.061 |
| 3 (384-640) | 0.78 | 0.76 | 0.045 | 0.013 | 0.032 |
| 4 (512-768) | 0.83 | 0.79 | 0.049 | 0.010 | 0.040 |
| 5 (640-896) | 0.82 | 0.76 | 0.090 | 0.015 | 0.076 |

- **Error falls as the forest share rises**, from 22% in the northern mixed strip to about 4-5% in
  the forest interior.
- **Missed forest dominates everywhere** (the false-negative column), and the predicted forest share
  is lower than the label in every row, so this model under-predicts forest.
- The worst tiles by error rate are 0 (0.273), 1 (0.266), 29 (0.236) and 2 (0.211); the best are 15
  (0.001) and 16 (0.004). Tiles 0, 1 and 2 are neighbours in the top row and overlap by 50%, so
  they are one failure area seen three times, not three independent failures. Tile 29 is the
  bottom-right tile of the block.

Failure modes, as observed in the **earlier session's** error figure (same data and split, worst
tiles 0, 1, 2 and 4, all in the village-and-farmland strip). The causes are plausible hypotheses and
were not tested:

1. Green irrigated crop fields predicted as forest. In a June-August composite, crops can resemble
   canopy in both visible and near-infrared bands.
2. Sparse or dry vegetation that the label counts as forest and the model misses. Whether the label
   or the model is right here cannot be decided from 10 m imagery.
3. Thin boundary errors around settlements, from mixed pixels, very fragmented labels and smoother
   predicted masks.

Figures: `results/error_analysis.png` (best and worst tiles: image, label, prediction, errors).

![error analysis](results/error_analysis.png)

### Forest map and area

![forest map](results/forest_map.png)

Whole-scene prediction (train, validation and test windows all included; the frames in the figure
mark the three blocks):

| Quantity | Value |
|---|---|
| Predicted forest, raster | 39,513 ha |
| Predicted forest, polygons >= 1 ha | 39,381 ha (146 polygons) |
| WorldCover forest | 39,513 ha |
| Pixels where prediction and label disagree | 413,243 of 5,080,032 (8.1%) |

The raster total equals the label total to the hectare (3,951,333 vs 3,951,300 pixels), but the
regional breakdown shows cancelling errors:

| Region | Predicted | Label | Difference |
|---|---|---|---|
| Train block (in-sample) | 20,005 ha | 19,274 ha | +731 ha (+3.8%) |
| Validation block | 5,040 ha | 5,493 ha | -453 ha (-8.2%) |
| Rest (gaps and test block) | about 14,468 ha | about 14,745 ha | -277 ha (-1.9%) |

Block-wise whole-scene scores (windows may read pixels across block edges): train block forest IoU
0.905 / non-forest IoU 0.692 (in-sample, and only 0.9 forest IoU, so the model is not memorizing the
training area); validation block 0.830 / 0.755.

### Earlier experiment: NIR ablation (superseded split)

Before the final three-way split, a two-way split (west train, east validation) was used for one
ablation: RGB vs RGB+NIR, 8 epochs, 2 seeds each.

| | mIoU | Non-forest IoU |
|---|---|---|
| RGB | 0.759 | 0.648 |
| RGB+NIR | 0.783 | 0.692 |

NIR helped non-forest IoU on both seeds (about +0.04) while the mIoU gain (+0.024) was comparable
to the seed spread. **This ablation was not repeated on the final split**, and given the seed noise
measured later (std about 0.02-0.03), it should be treated as suggestive only. The final model uses
all four bands.

## How to read these numbers

- **IoU** = overlap / union for one class. Pixel accuracy is misleading when one class dominates, so
  forest and non-forest IoU are reported separately, with mIoU as their mean.
- **All-forest baseline**: predicting "forest" everywhere scores forest IoU equal to the forest share
  of the block (0.700 validation, 0.892 test) and non-forest IoU 0. Every score above is read
  against it.
- **Minority-class IoU drops when the class gets rarer**, even at the same per-pixel error rate.
  Non-forest is 30% of validation but 11% of test, so the lower test non-forest IoU is not by itself
  evidence of poor generalization.

## Limitations

1. **Labels are model-derived** and noisy in mixed areas, and may be partly circular with the
   Sentinel-2 inputs.
2. **The best checkpoint came from epoch 1 of 15** with a high constant learning rate, and the
   training curve was not examined before the test was run (see below).
3. **Small and noisy validation set**: 30 tiles with 50% overlap (about 8 independent tiles'
   worth). Checkpoint selection and the hyperparameter search both used it, so validation scores are
   optimistic and the best-of-20 search result is biased upward.
4. **The test is one contiguous block** of about 8,935 ha, with about 960 ha of non-forest, from the
   same scene and season as the training data. The buffers reduce, but do not remove, spatial
   autocorrelation. It is a weaker claim than an independent region.
5. **Validation and test use different inference protocols** (tiles vs block-wise sliding windows).
6. **One scene, one season, one region.** No claim is made about other regions or years.
7. **GPU training is not bit-reproducible**, and identical configurations varied by 0.01-0.03 mIoU
   between sessions.
8. The failure-mode explanations are hypotheses, and the final model's worst tiles (including
   tile 29) were not all inspected.

## Open issue: training dynamics

A best epoch of 1 out of 15, an in-sample forest IoU of only 0.905, and a flat hyperparameter search
suggest the model may be undertrained or training unstably (a learning rate of 7.85e-4 with no
schedule is high for fine-tuning a pretrained encoder at batch size 8), rather than data-limited.
This was **not verified**. The planned diagnostic logs training loss, training mIoU and validation
mIoU per epoch for the current settings and for a gentler learning rate with cosine decay.

Because the test block has already been scored once, any model retrained after this point must be
reported as a new model, and the current test block then counts as a second validation block. A
clean final claim would be evaluated once on an independent region.

## Next steps

1. Run the training-dynamics diagnostic and retrain with a learning-rate schedule if it shows
   instability.
2. Evaluate once on an independent region (for example the Gabala-Shaki foothills of the Greater
   Caucasus), which would be a stronger test than a block of the same scene.
3. Evaluate validation and test with the same inference method, and add more seeds.
4. Add a second-season composite (spring) and vegetation indices to help separate crops from forest.
5. Inspect label errors against high-resolution imagery in the northern strip.
6. Multi-temporal change detection, and the same pipeline with field-boundary labels for field
   delineation.

## MLOps

- **Experiment tracking: MLflow** (local SQLite backend). 30 runs: a smoke test, the Optuna parent
  run with 20 nested trials, the seed-check parent run with 6 nested runs, and the final training
  run. The final run holds the parameters, per-epoch validation metrics, final validation and test
  metrics, and the artifacts `best.pt`, `norm.npz` (band means and standard deviations) and the
  error-analysis figure.
- **Hyperparameter search: Optuna** (TPE), with trial 0 pinned to the defaults so the "before
  tuning" reference comes for free.
- `mlflow.db` and `mlruns/` are not committed (see `.gitignore`); a screenshot of the run comparison
  is in `results/mlflow_runs.png`.
- Unit tests (7, all passing) cover the properties where a silent bug would invalidate the results:
  blocks stay inside the scene and are separated by at least 256 px, tiles never leave their block,
  nodata tiles are skipped, and IoU and polygon-to-hectare conversion are correct.

## Repository layout and reproduction

```
forest_seg/          tested helper code
  tiling.py          window starts, three-way block split, tile cutting
  metrics.py         confusion counts, IoU / Dice / accuracy
  vectorize.py       mask -> polygons with area in hectares
tests/test_core.py   7 unit tests
notebooks/           the full experiment notebook (data download, training, search, evaluation)
results/             error_analysis.png, forest_map.png, mlflow_runs.png, forest.geojson
requirements.txt
```

To reproduce: open the notebook on a GPU runtime with internet access, run it top to bottom (the
data download uses the free Planetary Computer API, which rate-limits shared machines; a second
public source, AWS Earth Search plus the open WorldCover tile, works as a fallback), and run
`python -m pytest -q` for the tests. Expect the numbers to vary by the amounts documented above.

Contains modified Copernicus Sentinel data (2021). ESA WorldCover 2021 (c) ESA, CC BY 4.0.
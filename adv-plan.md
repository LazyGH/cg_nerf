# adv-plan.md — Edge-Aware Ray Sampling for TensorFlow NeRF

## Goal

Implement a minimal edge-aware ray sampler for the existing modified TensorFlow NeRF codebase. The method should sample more training camera rays from high-gradient image regions while retaining some uniform sampling coverage. Use the normal biased NeRF RGB reconstruction loss; do **not** add inverse-probability loss weighting.

This is a course-project improvement over original random ray sampling. Keep the change isolated to the ray-selection stage and preserve the existing NeRF architecture, rendering pipeline, hierarchical depth sampling, optimizer, logging, and evaluation scripts as much as possible.

## Scope constraints

1. Implement only the biased training variant:
   - Keep the original MSE RGB loss exactly as before.
   - Do not divide loss terms by ray sampling probability.
2. Evaluate only two scenes:
   - `lego`
   - `fern`
3. Use one fixed alpha per scene:
   - `lego`: `edge_sampling_alpha = 0.6`
   - `fern`: `edge_sampling_alpha = 0.4`
4. Implement in two ray-batching modes: `use_batching` and `no_batching`
5. Do not run alpha ablations.
6. Do not retrain original NeRF baselines. The user already has original trained checkpoints and specific baseline config files.
7. New edge-aware configs must reference/copy the user's existing scene configs and change only the experiment name plus edge-sampling flags.
8. The final run will be done manually on Colab with the existing conda environment. Local `uv`/Python 3.8 is only for lightweight syntax and logic checks.

## Current codebase

The repository is a modified version of the original TensorFlow NeRF codebase (./nerf), only some adaptation of running environment. 

- Main training script: `run_nerf.py`
- Dataset loaders: `load_blender.py`, `load_llff.py`
- independent evaluation scripts (./project_tools) and customized training configs (./nerf/course_configs).You can re-use them.

- The original code has two ray-batching modes:

  1. `use_batching = not args.no_batching`
     - Precomputes `rays_rgb` over all training images.
     - Shuffles and slices random ray batches.
  2. `no_batching=True`
     - Selects one training image per iteration.
     - Randomly samples pixel coordinates from that image.

  The implementation should support both paths if practical. This avoids silently changing Fern's training mode if its config currently uses global batching.

## High-level method

For each training image, compute an image-space edge probability map:

```text
gray image -> Sobel gradient magnitude -> optional light smoothing -> normalize
```

Then mix the edge distribution with uniform sampling:

```text
P_mix(pixel) = alpha * P_edge(pixel) + (1 - alpha) * P_uniform(pixel)
```

Training ray selection then samples pixel indices according to `P_mix`.

Important: this changes only **which rays are selected**. It must not change:

- volumetric rendering
- near/far bounds
- positional encoding
- coarse/fine model
- hierarchical samples along each ray
- RGB loss formula

## Implementation tasks

### 1. Add command-line/config flags

Add to `config_parser()` in `run_nerf.py`:

```python
parser.add_argument("--edge_ray_sampling", action="store_true",
                    help="Enable edge-aware image-space ray sampling.")

parser.add_argument("--edge_sampling_alpha", type=float, default=0.0,
                    help="Mixture weight for edge probability map. 0.0 means uniform; 1.0 means edge-only.")

parser.add_argument("--edge_sampling_eps", type=float, default=1e-6,
                    help="Small positive constant added before edge-probability normalization.")

parser.add_argument("--edge_sampling_smooth", action="store_true",
                    help="Apply a lightweight 3x3 smoothing filter to gradient magnitude before normalization.")
```

Validation rules:

```python
if args.edge_ray_sampling:
    assert 0.0 <= args.edge_sampling_alpha <= 1.0
```

Print one clear startup message:

```text
EDGE RAY SAMPLING enabled: alpha=<value>, eps=<value>, smooth=<bool>
```

### 2. Add edge-map utility functions

Prefer a small utility module, for example:

```text
edge_ray_sampler.py
```

If the repository is very small, adding these functions directly to `run_nerf.py` is acceptable, but a separate module is cleaner.

Required functions:

```python
def rgb_to_luma_np(img):
    """img: [H, W, 3] float image in [0,1]. Return [H, W] luma."""
```

Use standard luma weights:

```python
0.299 * R + 0.587 * G + 0.114 * B
```

```python
def sobel_grad_mag_np(gray):
    """Compute Sobel gradient magnitude with numpy only. No OpenCV, scipy, or skimage."""
```

Use reflect padding and Sobel kernels:

```python
Kx = [[ 1, 0, -1],
      [ 2, 0, -2],
      [ 1, 0, -1]]

Ky = [[ 1,  2,  1],
      [ 0,  0,  0],
      [-1, -2, -1]]
```

```python
def smooth3x3_np(x):
    """Optional dependency-free 3x3 smoothing."""
```

Use a simple weighted filter or box filter. Keep it dependency-free.

```python
def build_edge_prob_map(img, alpha, eps=1e-6, smooth=False):
    """
    Return a flattened [H*W] probability vector.
    P_mix = alpha * normalized_edge + (1-alpha) * uniform.
    """
```

```python
def build_edge_cdfs(images, i_train, alpha, eps=1e-6, smooth=False):
    """
    Return:
      per_image_cdfs: dict or list mapping image index -> CDF over H*W pixels
      global_cdf: CDF over concatenated train images in the same order used by rays_rgb
    """
```

```python
def sample_from_cdf(cdf, n):
    """Fast weighted sampling with replacement using np.searchsorted."""
```

Clamp returned indices to `len(cdf)-1` for numerical safety.

### 3. Compute edge CDFs after images are finalized

In `train()` inside `run_nerf.py`, compute edge CDFs after:

- dataset loading
- Blender alpha compositing / white background handling
- train/test/val split creation
- `H`, `W`, `focal` casting

Do not compute edge maps before Blender alpha compositing, because the sampler should see the same RGB targets used by training.

Pseudo-placement:

```python
# after images are finalized and i_train is known
edge_cdfs = None
edge_global_cdf = None

if args.edge_ray_sampling:
    edge_cdfs, edge_global_cdf = build_edge_cdfs(
        images=images,
        i_train=i_train,
        alpha=args.edge_sampling_alpha,
        eps=args.edge_sampling_eps,
        smooth=args.edge_sampling_smooth,
    )
```

Log basic sanity statistics:

```text
Built edge-aware CDFs for <num_train> train images.
Global CDF length = <num_train * H * W>.
```

### 4. Modify the `use_batching` path

Original behavior:

```python
batch = rays_rgb[i_batch:i_batch+N_rand]
np.random.shuffle(rays_rgb) when exhausted
```

New behavior when `args.edge_ray_sampling` is enabled:

1. Keep constructing `rays_rgb` in the original order:
   - stack train images
   - flatten to `[N_train * H * W, 3, 3]`
2. Do **not** shuffle `rays_rgb` for edge-aware sampling.
3. Build `edge_global_cdf` in exactly the same image/pixel flattening order:

```python
global_probs = np.concatenate([prob_map_for_image_i for i in i_train], axis=0)
edge_global_cdf = np.cumsum(global_probs)
edge_global_cdf[-1] = 1.0
```

4. In the training loop:

```python
if use_batching:
    if args.edge_ray_sampling:
        select_inds = sample_from_cdf(edge_global_cdf, N_rand)
        batch = rays_rgb[select_inds]
    else:
        batch = rays_rgb[i_batch:i_batch+N_rand]
        ...
```

5. Then keep the existing code:

```python
batch = tf.transpose(batch, [1, 0, 2])
batch_rays, target_s = batch[:2], batch[2]
```

Notes:

- Sampling with replacement is acceptable and much faster than weighted sampling without replacement over millions of rays.
- Duplicate rays within a batch should be rare for Fern because the global ray pool is large.
- This path preserves Fern compatibility if the existing config uses global batching.

### 5. Modify the `no_batching` path

Original behavior:

```python
img_i = np.random.choice(i_train)
coords = full image or precrop coords
select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
```

New behavior when `args.edge_ray_sampling` is enabled:

```python
if args.edge_ray_sampling:
    if i < args.precrop_iters:
        # Respect the existing center-crop warmup.
        # Restrict probabilities to the crop coordinates and renormalize.
        flat_coords = coords_np[:, 0] * W + coords_np[:, 1]
        crop_probs = image_prob_flat[flat_coords]
        crop_probs = crop_probs / np.sum(crop_probs)
        crop_cdf = np.cumsum(crop_probs)
        local_ids = sample_from_cdf(crop_cdf, N_rand)
        select_inds_np = coords_np[local_ids]
    else:
        flat_ids = sample_from_cdf(edge_cdfs[img_i], N_rand)
        ys = flat_ids // W
        xs = flat_ids % W
        select_inds_np = np.stack([ys, xs], axis=-1)

    select_inds = tf.convert_to_tensor(select_inds_np, dtype=tf.int32)
else:
    # keep original uniform selection
```

If implementing crop support is too intrusive, acceptable fallback:

- During `precrop_iters`, use original uniform crop sampling.
- After `precrop_iters`, switch to edge-aware sampling.

But prefer crop-aware renormalization because it is simple and cleaner.

### 6. Preserve biased loss

Do not change this block except comments if necessary:

```python
img_loss = img2mse(rgb, target_s)
loss = img_loss

if 'rgb0' in extras:
    img_loss0 = img2mse(extras['rgb0'], target_s)
    loss += img_loss0
```

Do not add sample-probability weights.

Add a comment:

```python
# Edge-aware sampling intentionally uses biased ray selection.
# Keep the original unweighted RGB MSE objective.
```

### 7. Add edge-aware config files

existing baseline configs:

```
lego_original: nerf/course_configs/lego_t4_20min.txt
fern_original: nerf/course_configs/fern_t4_20min.txt
```

Create two new config files by copying the user's existing baseline configs.

Do not overwrite the baseline configs.

Suggested names:

```text
configs/lego_edge_alpha06.txt
configs/fern_edge_alpha04.txt
```

If the repository keeps configs at the root, use root-level names instead:

```text
config_lego_edge_alpha06.txt
config_fern_edge_alpha04.txt
```

The new files should preserve every baseline setting except `expname` and added edge flags.

#### Lego config change

Starting from the user's previous Lego config:

```text
expname = lego_edge_alpha06

edge_ray_sampling = True
edge_sampling_alpha = 0.6
edge_sampling_eps = 1e-6
edge_sampling_smooth = True
```

Do not change:

- `datadir`
- `dataset_type`
- `half_res`
- `white_bkgd`
- `N_rand`
- `N_samples`
- `N_importance`
- `use_viewdirs`
- learning rate / decay if present
- render/test intervals if present

#### Fern config change

Starting from the user's previous Fern config:

```text
expname = fern_edge_alpha04

edge_ray_sampling = True
edge_sampling_alpha = 0.4
edge_sampling_eps = 1e-6
edge_sampling_smooth = True
```

Do not change:

- `datadir`
- `dataset_type`
- `factor`
- `llffhold`
- `N_rand`
- `N_samples`
- `N_importance`
- `use_viewdirs`
- `raw_noise_std`
- learning rate / decay if present
- render/test intervals if present

### 8. Suggested training commands for Colab

Use the user's conda setup exactly as before.

```
cd /content/cg_nerf
conda env create -f nerf/environment_colab_tf115.yml
conda run -n nerf-tf115 python -V
```

previous train script usage Examples:

```bash
cd /content/cg_nerf/nerf
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/lego_t4_20min.txt
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/fern_t4_20min.txt
```

Train each edge-aware model to the same checkpoint iteration used by the user's saved original NeRF baseline for that scene.

Do not compare a 200k-step baseline against a shorter edge-aware run unless clearly reported as a convergence-speed comparison.

## Evaluation plan

### 1. Baselines

Use the user's already-trained original NeRF checkpoints (will load from google drive).

Do not retrain these.

### 2. Edge-aware models

Train only:

```text
lego_edge_alpha06
fern_edge_alpha04
```

### 3. Rendering

previous  Render Evaluate script usage Examples:

```bash
cd /content/cg_nerf
conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_t4_20min.txt \
  --output_dir result/nerf/lego_trained_20min \
  --source trained \
  --train_time_minutes 20
```

Render the test set for each original and edge-aware model using the existing render/evaluation workflow.


Use the same checkpoint iteration for original and edge-aware models where possible.

### 4. Metrics

Use the repository's existing independent evaluation scripts if they already compute standard novel-view metrics.

Minimum required metrics:

```text
PSNR
SSIM
LPIPS
```

Preferred metrics if already available:

If the existing scripts do not support edge-focused analysis, add a small optional script:

```text
scripts/eval_edge_regions.py
```

This script should compute, for each rendered test image:

1. Load ground truth RGB and predicted RGB.
2. Compute ground-truth Sobel gradient magnitude.
3. Define edge mask as top 20% gradient pixels.
4. Define flat mask as bottom 50% gradient pixels.
5. Report:
   - full-image PSNR
   - edge-region PSNR
   - flat-region PSNR

This script can be dependency-free except for `numpy` and `imageio`.

Do not block the main implementation if SSIM/LPIPS dependencies are unavailable locally. The Colab conda environment or existing evaluation scripts may already handle them.

### 5. Expected result table

Produce one CSV or markdown table with this structure:

```text
scene, method, alpha, checkpoint_step, psnr, ssim, lpips, edge_psnr, flat_psnr, notes
lego, original, -, <step>, ...
lego, edge-aware, 0.6, <step>, ...
fern, original, -, <step>, ...
fern, edge-aware, 0.4, <step>, ...
```

If SSIM/LPIPS are unavailable, leave them blank or write `NA`.

### 6. Qualitative outputs

For the final report/demo, save images:

```text
outputs/edge_sampling_comparison/lego/
outputs/edge_sampling_comparison/fern/
```

For each scene, include:

1. Ground truth test view.
2. Original NeRF rendering.
3. Edge-aware NeRF rendering.
4. Absolute error map.
5. Cropped comparison around high-gradient regions.

Suggested crops:

- Lego: object silhouette, wheels/treads, thin black/gray boundaries, high-contrast object-background edges.
- Fern: leaf boundaries, fine stems, high-frequency foliage regions.

## Minimal testing plan for Codex/local environment

Because local `uv` Python 3.8 is only for lightweight checks, do not require TensorFlow locally.

### 1. Syntax check

Run:

```bash
python -m py_compile edge_ray_sampler.py
python -m py_compile run_nerf.py
```

If the TensorFlow codebase uses syntax incompatible with Python 3.8 or imports fail, at least run syntax checks on the new utility module.

### 2. Utility smoke test

Add or run a small script:

```bash
python scripts/test_edge_ray_sampler.py
```

Suggested test cases:

1. Constant image:
   - Edge map should be nearly uniform.
   - Probability sum should be 1.
2. Image with a sharp vertical boundary:
   - Pixels near the boundary should have higher sampling probability than flat pixels.
3. CDF sampling:
   - Returned indices must be within `[0, H*W - 1]`.
   - No NaN or inf values.

Example assertions:

```python
assert np.isclose(prob.sum(), 1.0)
assert np.all(np.isfinite(prob))
assert np.all(prob > 0)
assert sampled.min() >= 0
assert sampled.max() < prob.size
```

## Acceptance criteria

Implementation is complete when:

1. `edge_ray_sampling=False` reproduces the original sampling path with no behavioral change.
2. `edge_ray_sampling=True` trains without changing the NeRF model architecture or loss formula.
3. Both global batching and per-image no-batching paths either work or have a clearly documented fallback.
4. New configs exist for:
   - Lego alpha 0.6
   - Fern alpha 0.4
5. The original baseline configs/checkpoints are not overwritten.
6. The code can render test views for both original and edge-aware models.
7. The evaluation output contains at least full-image PSNR for all four model-scene pairs.
8. Optional but preferred: edge-region PSNR and qualitative crops are generated for the final report.

## Notes for final write-up

Claim the contribution carefully:

```text
We introduce an edge-aware image-space ray sampler for NeRF training. Instead of uniformly sampling training pixels, the sampler constructs a Sobel-gradient-based probability map for each training image and mixes it with a uniform distribution. This biases the fixed ray budget toward high-frequency image regions such as silhouettes and thin structures while preserving coverage of low-gradient regions. The rendering loss remains the standard unweighted NeRF RGB reconstruction loss.
```

Expected interpretation:

- Lego may show sharper object boundaries and high-contrast details because alpha is higher (`0.6`).
- Fern uses lower alpha (`0.4`) because natural images contain many texture gradients; too much edge bias could undersample smooth geometry and background regions.
- If global PSNR changes only slightly, emphasize edge-region PSNR and qualitative crops.
- If flat-region PSNR drops, explain it as the expected trade-off from biased sampling and discuss the uniform-mixture component as mitigation.

## Write running instruction file

Write a running instruction md file for codebase on Colab.

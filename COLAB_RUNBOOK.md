# NeRF and DVGO Colab Runbook

This workspace now contains:

- Minimal TensorFlow 2 compatibility for the official `nerf` code.
- A configurable `N_iters` argument for `nerf/run_nerf.py`.
- Colab-oriented 30 minute configs for NeRF and DVGO.
- Standalone evaluation scripts that save metrics, RGB renders, disparity or depth maps, and error maps.

## Recommended experiment matrix

Run these 6 evaluation cases:

1. NeRF trained from scratch on `lego`
2. NeRF trained from scratch on `fern`
3. NeRF official pretrained checkpoint on `lego`
4. NeRF official pretrained checkpoint on `fern`
5. DVGO trained from scratch on `lego`
6. DVGO trained from scratch on `fern`

Use the same test split for all runs:

- `lego`: official Blender test split
- `fern`: `llffhold = 8`

Recommended comparison protocol:

- Primary: same wall-clock budget, about 30 minutes per scene
- Secondary: same configured training steps from the provided configs

For DVGO, the first run on a fresh Colab runtime compiles custom CUDA extensions. Do one short warm-up import before timing the first DVGO training run, or note that the first scene includes one-time compile overhead.

## Suggested 30 minute configs

NeRF:

- `nerf/course_configs/lego_t4_30min.txt`
- `nerf/course_configs/fern_t4_30min.txt`

DVGO:

- `DirectVoxGO/configs/course/lego_t4_30min.py`
- `DirectVoxGO/configs/course/fern_t4_30min.py`

These are tuned for a Colab T4 and low-resolution training. Actual runtime still varies with Colab load, first-run package install time, LLFF image minification, and DVGO CUDA compilation.

## Colab setup

Use a GPU runtime with a T4 if available.

Upload or clone this project into Colab, then work from the project root:

```bash
cd /content/pj
```

### Common utilities

```bash
pip install -U imageio imageio-ffmpeg scipy matplotlib lpips configargparse
apt-get update
apt-get install -y imagemagick
```

## Part A: NeRF

### Install

The official TensorFlow install guide recommends `tensorflow[and-cuda]` on Linux. In Colab, the following works with the patched NeRF code:

```bash
cd /content/pj
pip install -U "tensorflow[and-cuda]" configargparse imageio imageio-ffmpeg matplotlib scipy lpips
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Download data

```bash
cd /content/pj/nerf
bash download_example_data.sh
```

This downloads both:

- `data/nerf_synthetic/lego`
- `data/nerf_llff_data/fern`

### Train baseline NeRF

Lego:

```bash
cd /content/pj/nerf
%%time
python run_nerf.py --config course_configs/lego_t4_30min.txt
```

Fern:

```bash
cd /content/pj/nerf
%%time
python run_nerf.py --config course_configs/fern_t4_30min.txt
```

Record the wall-clock time reported by `%%time` and keep it for the comparison table.

### Download and evaluate official NeRF pretrained checkpoints

```bash
cd /content/pj/nerf
bash download_example_weights.sh
find logs -name "model_*.npy"
```

Use the returned checkpoint paths with the evaluation script.

### Evaluate NeRF runs

Trained Lego:

```bash
cd /content/pj
python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_t4_30min.txt \
  --output_dir result/nerf/lego_trained \
  --source trained \
  --train_time_minutes 30
```

Trained Fern:

```bash
cd /content/pj
python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/fern_t4_30min.txt \
  --output_dir result/nerf/fern_trained \
  --source trained \
  --train_time_minutes 30
```

Official pretrained Lego:

```bash
cd /content/pj
python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_t4_30min.txt \
  --ckpt <CKPT_PATH_FROM_FIND> \
  --output_dir result/nerf/lego_pretrained \
  --source pretrained
```

Official pretrained Fern:

```bash
cd /content/pj
python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/fern_t4_30min.txt \
  --ckpt <CKPT_PATH_FROM_FIND> \
  --output_dir result/nerf/fern_pretrained \
  --source pretrained
```

Replace `<CKPT_PATH_FROM_FIND>` with the concrete file returned by `find`.

## Part B: DVGO

### Install

The official PyG installation guide recommends using the wheel index that matches the installed PyTorch and CUDA version for `torch_scatter`. The official MMCV docs recommend `openmim` for installation. DVGO only uses `mmcv.Config`, so `mmcv-lite` is sufficient here and installs faster. That package choice is an inference from the codebase.

```bash
cd /content/pj
pip install -U ninja tqdm opencv-python-headless imageio imageio-ffmpeg scipy lpips torch_efficient_distloss einops openmim mmcv-lite
export TORCH=$(python - <<'PY'
import torch
print(torch.__version__.split('+')[0])
PY
)
export CUDA=$(python - <<'PY'
import torch
print('cpu' if torch.version.cuda is None else f"cu{torch.version.cuda.replace('.', '')}")
PY
)
echo $TORCH
echo $CUDA
```

Then install `torch_scatter` using those values:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Example only:

```bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### Reuse the same datasets

DVGO expects:

- `DirectVoxGO/data/nerf_synthetic/lego`
- `DirectVoxGO/data/nerf_llff_data/fern`

If you already downloaded the NeRF example data, copy or symlink it:

```bash
cd /content/pj
mkdir -p DirectVoxGO/data
cp -r nerf/data/nerf_synthetic DirectVoxGO/data/
cp -r nerf/data/nerf_llff_data DirectVoxGO/data/
```

### Optional warm-up for DVGO extension build

```bash
cd /content/pj/DirectVoxGO
python -c "from lib import dvgo; print('DVGO CUDA extension import ok')"
```

### Train DVGO

Lego:

```bash
cd /content/pj/DirectVoxGO
%%time
python run.py --config configs/course/lego_t4_30min.py --render_test --dump_images --eval_ssim --eval_lpips_alex
```

Fern:

```bash
cd /content/pj/DirectVoxGO
%%time
python run.py --config configs/course/fern_t4_30min.py --render_test --dump_images --eval_ssim --eval_lpips_alex
```

Record the wall-clock time for each scene.

### Evaluate DVGO runs with the unified exporter

Trained Lego:

```bash
cd /content/pj
python project_tools/render_eval_dvgo.py \
  --config DirectVoxGO/configs/course/lego_t4_30min.py \
  --output_dir result/dvgo/lego_trained \
  --source trained \
  --train_time_minutes 30
```

Trained Fern:

```bash
cd /content/pj
python project_tools/render_eval_dvgo.py \
  --config DirectVoxGO/configs/course/fern_t4_30min.py \
  --output_dir result/dvgo/fern_trained \
  --source trained \
  --train_time_minutes 30
```
## for 20 min train time:

```
python nerf/run_nerf.py --config nerf/course_configs/lego_t4_20min.txt
python nerf/run_nerf.py --config nerf/course_configs/fern_t4_20min.txt

python DirectVoxGO/run.py --config DirectVoxGO/configs/course/
lego_t4_20min.py --render_test --dump_images --eval_ssim
--eval_lpips_alex
python DirectVoxGO/run.py --config DirectVoxGO/configs/course/
fern_t4_20min.py --render_test --dump_images --eval_ssim
--eval_lpips_ale
```

## Aggregate the quantitative results

```bash
cd /content/pj
python project_tools/aggregate_metrics.py \
  result/nerf/lego_trained/metrics.json \
  result/nerf/fern_trained/metrics.json \
  result/nerf/lego_pretrained/metrics.json \
  result/nerf/fern_pretrained/metrics.json \
  result/dvgo/lego_trained/metrics.json \
  result/dvgo/fern_trained/metrics.json \
  --out_csv result/summary_metrics.csv
```

For paper-reported values, fill in `project_tools/paper_reference_template.csv` manually from the paper tables or supplementary material, then combine that CSV with `result/summary_metrics.csv` in your report spreadsheet.

## What each evaluation script saves

Each output directory contains:

- `metrics.json`: aggregate and per-image metrics
- `metrics.csv`: per-image metrics in spreadsheet form
- `rgb_pred/`: rendered RGB test images
- `rgb_gt/`: ground-truth test images
- `error_map/`: per-image error heatmaps and raw error arrays

NeRF outputs also contain:

- `disp_vis/`: disparity visualization PNGs
- `disp_raw/`: raw disparity arrays
- `acc_raw/`: accumulated opacity arrays

DVGO outputs also contain:

- `depth_vis/`: depth visualization PNGs
- `depth_raw/`: raw depth arrays
- `bgmap_raw/`: raw background alpha-inverse arrays

Both exporters also save stacked `.npy` tensors for fast reuse in the report:

- `pred_rgb_stack.npy`
- `gt_rgb_stack.npy`
- scene depth or disparity stacks

## What you should save after running

Keep these directories and files:

1. Training logs and checkpoints
   - `nerf/logs/<expname>`
   - `DirectVoxGO/logs/.../<expname>`
2. Evaluation outputs
   - `result/nerf/...`
   - `result/dvgo/...`
3. The exact configs you used
   - NeRF config txt
   - DVGO config py
4. Timing records
   - the notebook cell output with `%%time`
   - or a small table with minutes per scene
5. The final combined metrics file
   - `result/summary_metrics.csv`
6. The notebook itself
   - this preserves exact commands, package versions, and runtime state

## Minimal report structure

Recommended quantitative table columns:

- method
- scene
- source
- train_time_minutes
- PSNR
- SSIM
- LPIPS

Recommended qualitative figures:

- RGB render vs ground truth
- NeRF disparity map
- DVGO depth map
- error map
- one short novel-view video or GIF per method and scene

## Notes

- NeRF quick configs are low-resolution baselines for the course project, not paper-quality reproduction settings.
- If you have extra time, compare them against `nerf/paper_configs/blender_config.txt` and `nerf/paper_configs/llff_config.txt` as a higher-quality reference.
- For a fair same-time comparison, exclude one-time dependency installation from all timing and preferably exclude the first DVGO CUDA extension compile as well.


## run tensorboard on colab

```
%load_ext tensorboard
%tensorboard --logdir /content/pj/nerf/logs/summaries
```

## Save Craft

```
from google.colab import drive
drive.mount('/content/drive/My Drive/Colab Notebooks/CG-pj')
!cp example.txt /content/drive/My Drive/Colab Notebooks/CG-pj
```

## Training Time Setting**

Based on the configuration you provided (note: your config specifies the **fern** dataset from LLFF, not the synthetic **lego** dataset), **yes, this setup is borderline suitable for a ~30-minute training time on a Colab T4 GPU, but it will likely spill over slightly to 35–45 minutes.**

Here is a quick breakdown of why:
* **The Math:** On a T4 GPU, standard TensorFlow NeRF processes a batch (`N_rand = 1024` rays with `64+64` samples) at roughly 6 to 8 iterations per second. 15,000 iterations will take about **30 to 35 minutes of pure training time**.
* **The Bottleneck:** You have `i_img = 500`. This means NeRF will pause training to render a full test image every 500 steps (30 times total). Rendering a full image is computationally expensive and will easily add 5 to 10 minutes of overhead to your run.

---

### How to modify it to guarantee < 30 minutes

If you strictly need the training to finish under 30 minutes, you should make the following adjustments. I have categorized them from **"No Impact on Quality"** to **"Slight Drop in Quality"**.

#### 1. Reduce Validation Overhead (Highly Recommended - No quality loss)
The easiest way to save time is to stop the model from rendering so many test images during training.
* Change `i_img = 5000` (Instead of 500. This alone saves ~5 minutes).

#### 2. Reduce the Number of Iterations (Guaranteed time reduction)
If 15,000 iterations takes 35 minutes, 10,000 will comfortably finish in ~20-25 minutes. The model will converge slightly less, but the `fern` dataset looks decent even at 10k steps.
* Change `N_iters = 10000` 

#### 3. Reduce Batch Size or Samples per Ray (If you must keep 15,000 iters)
If you want to keep `N_iters = 15000` but speed up the individual steps, you can lower the workload per iteration. *Note: This will make the final render slightly noisier/blurrier.*
* **Option A (Fewer rays):** Change `N_rand = 512` (Down from 1024. Halves the rays processed per step).
* **Option B (Fewer samples along the ray):** Change `N_samples = 32` and `N_importance = 32` (Down from 64. Halves the number of times the neural network is evaluated per ray).

---

### Suggested Modified Config (Targeting ~20-25 mins)

If you want a safe, optimized config for a Colab T4 that definitely runs in under 30 minutes while preserving as much quality as possible, use this:

```text
expname = fern_t4_under_30min
basedir = ./logs
datadir = ./data/nerf_llff_data/fern
dataset_type = llff

factor = 8
llffhold = 8
use_viewdirs = True
raw_noise_std = 1e0

# Keeping samples high for quality, but dropping batch size slightly
N_samples = 64
N_importance = 64
N_rand = 512       # Reduced from 1024 to speed up each iteration
N_iters = 15000    
random_seed = 0

i_print = 100
i_img = 5000       # Reduced rendering frequency to save overhead time!
i_weights = 5000
i_testset = 15000
i_video = 15000
```



**Response**:

• Implemented a Colab-oriented reproduction/evaluation workflow with
  minimal source changes. The main entry point is COLAB_RUNBOOK.md, which
  now gives the full NeRF/DVGO setup, training, evaluation, comparison
  flow, and a clear list of what to save after running.

  The only official-source patch is in nerf/run_nerf.py and nerf/
  run_nerf_helpers.py: TensorFlow 2 compatible summaries plus a new N_iters
  config argument so your NeRF runs can be capped cleanly for a 30 minute
  budget. I added 30-minute configs at nerf/course_configs/
  lego_t4_30min.txt, nerf/course_configs/fern_t4_30min.txt, DirectVoxGO/
  configs/course/lego_t4_30min.py, and DirectVoxGO/configs/course/
  fern_t4_30min.py.

  For evaluation and comparison, I added unified exporters at
  project_tools/render_eval_nerf.py and project_tools/render_eval_dvgo.py.
  They save metrics.json, metrics.csv, rendered RGB, GT RGB, disparity/
  depth visualizations, raw arrays, and error maps. Aggregation is in
  project_tools/aggregate_metrics.py, and a manual paper-baseline template
  is in project_tools/paper_reference_template.csv.

  I could not run a local Python compile check here because this Windows e
  nvironment’s python.exe and python3.exe point to inaccessible WindowsApps
  stubs. The runbook’s install notes were aligned to official docs for Ten
  sorFlow, PyG torch_scatter, and MMCV:
  https://www.tensorflow.org/install/pip,
  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html,
  https://mmcv.readthedocs.io/en/latest/get_started/installation.html.


• 1. The evaluation scripts do not train again.

  project_tools/render_eval_nerf.py only does this:

  - parses the NeRF config
  - finds a checkpoint you already trained, or uses the checkpoint path you
    pass with --ckpt
  - rebuilds the NeRF network and loads weights via ft_path
  - renders the test images
  - computes PSNR, SSIM, LPIPS
  - saves RGB predictions, GT, disparity, opacity, error maps, and
    metrics.json / metrics.csv

  You can see that flow in project_tools/render_eval_nerf.py:53,
  project_tools/render_eval_nerf.py:137, project_tools/
  render_eval_nerf.py:140, project_tools/render_eval_nerf.py:166,
  project_tools/render_eval_nerf.py:233.

  project_tools/render_eval_dvgo.py is the same idea for DVGO:

  - loads fine_last.tar or coarse_last.tar, or your explicit --ckpt
  - loads the trained model with utils.load_model(...)
  - renders the test images
  - computes metrics and saves artifacts

  See project_tools/render_eval_dvgo.py:51, project_tools/
  render_eval_dvgo.py:105, project_tools/render_eval_dvgo.py:109,
  project_tools/render_eval_dvgo.py:147, project_tools/
  render_eval_dvgo.py:219.

  So:

  - if checkpoints already exist, they use them
  - if checkpoints do not exist, they fail
  - they do not continue training

  2. TensorBoard on Colab

  For NeRF, TensorBoard logs are written to nerf/run_nerf.py:756 under:

  - nerf/logs/summaries/<expname>

  Use this in Colab:

  %load_ext tensorboard
  %tensorboard --logdir /content/pj/nerf/logs/summaries

  If you want one run only:

  %tensorboard --logdir /content/pj/nerf/logs/summaries/lego_t4_30min

  Also useful:

  - validation images are saved in nerf/logs/<expname>/tboard_val_imgs
  - checkpoints are in nerf/logs/<expname>/

  If you want to view locally later, download:

  - nerf/logs/summaries/<expname>/ for TensorBoard event files
  - optionally nerf/logs/<expname>/tboard_val_imgs/

  Then locally run:

  tensorboard --logdir /path/to/summaries

  For DVGO, the official code does not write TensorBoard logs. It prints
  progress to console and saves checkpoints; see DirectVoxGO/run.py:399 and
  DirectVoxGO/run.py:508. So for DVGO, unless we patch it, there is no
  TensorBoard. You mainly keep:

  - console cell output
  - DirectVoxGO/logs/.../<expname>/fine_last.tar
  - rendered evaluation outputs in result/...

  3. To limit training to about 20 minutes each

  You usually do not need to modify source code further. Just change config
  values.

  For NeRF, the main controls are:

  - N_iters
  - i_img
  - N_rand
  - N_samples
  - N_importance

  The most important hidden overhead is i_img, because every time it
  triggers, NeRF renders a full validation image during training; see nerf/
  run_nerf.py:887. For a 20 minute target, raise i_img a lot.

  Recommended 20-minute NeRF settings:


  If it is still too slow, reduce one more level:

  - N_rand = 512 for lego too
  - or N_samples = 32, N_importance = 32

  For DVGO, the main controls are:

  - coarse_train.N_iters
  - fine_train.N_iters
  - N_rand
  - num_voxels
  - pg_scale

  Recommended 20-minute DVGO settings:

  lego


  fern

  So the answer to “what other code needs to modify?” is:

  - none, if you are fine with iteration-based control
  - just create new config files and run with them

  Only if you want a strict wall-clock stop at exactly 20 minutes, then you
  would need extra control code outside the official loop, because both
  NeRF and DVGO are iteration-driven, not time-driven.

  If you want, I can add four ready-to-use *_t4_20min config files into the
  repo now.
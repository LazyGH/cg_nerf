# Separate Runtime Guide

This guide replaces the earlier "one mixed environment for everything" approach.

Recommended workflow:

1. Use one Colab runtime for NeRF only.
2. Use a different fresh Colab runtime for DVGO only.
3. Keep the outputs on Drive or download them after each runtime finishes.

This avoids TensorFlow and PyTorch/CUDA package conflicts and keeps the NeRF environment close to the original official repo.

## A. NeRF runtime

### Why this setup

The official NeRF repo in this project was originally written for:

- Python 3.7
- `tensorflow-gpu==1.15`

That original stack is preserved in [environment.yml](C:/Users/Jerry/Documents/Code/CityU/CS6493_CG/pj/nerf/environment.yml). For Colab, use the extended environment file [environment_colab_tf115.yml](C:/Users/Jerry/Documents/Code/CityU/CS6493_CG/pj/nerf/environment_colab_tf115.yml), which keeps TensorFlow 1.15 for training and adds:

- `tensorboard`
- `scipy`
- CPU PyTorch + `lpips` for `project_tools/render_eval_nerf.py`

### Colab setup for NeRF

Start a fresh Colab notebook and clone or upload the repo.

Install Conda inside Colab:

```python
!pip install -q condacolab
import condacolab
condacolab.install()
```

After the runtime restarts, create the NeRF environment:

```bash
cd /content/cg_nerf
conda env create -f nerf/environment_colab_tf115.yml
conda run -n nerf-tf115 python -V
```

Expected Python version:

```text
Python 3.7.x
```

### Download data

```bash
cd /content/cg_nerf/nerf
conda run -n nerf-tf115 bash download_example_data.sh
```

This creates:

- `nerf/data/nerf_synthetic/lego`
- `nerf/data/nerf_llff_data/fern`

### Train NeRF

20 minute configs:

```bash
cd /content/cg_nerf/nerf
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/lego_t4_20min.txt
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/fern_t4_20min.txt
```

30 minute configs:

```bash
cd /content/cg_nerf/nerf
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/lego_t4_30min.txt
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/fern_t4_30min.txt
```

### TensorBoard for NeRF

NeRF logs are written to:

- `nerf/logs/summaries/<expname>`

In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/cg_nerf/nerf/logs/summaries
```

Also keep:

- `nerf/logs/<expname>/tboard_val_imgs`
- `nerf/logs/<expname>/model_*.npy`
- `nerf/logs/<expname>/model_fine_*.npy`

### Evaluate trained NeRF

Because `environment_colab_tf115.yml` already includes the extra packages needed by the exporter, you can run evaluation inside the same env:

```bash
cd /content/cg_nerf
conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_t4_20min.txt \
  --output_dir result/nerf/lego_trained_20min \
  --source trained \
  --train_time_minutes 20
```

For pretrained checkpoints:

```bash
cd /content/cg_nerf/nerf
conda run -n nerf-tf115 bash download_example_weights.sh
find /content/cg_nerf/nerf/logs -name "model_*.npy"
```

Then pass the exact checkpoint path:

```bash
cd /content/cg_nerf
conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_t4_20min.txt \
  --ckpt <EXACT_CHECKPOINT_PATH> \
  --output_dir result/nerf/lego_pretrained \
  --source pretrained
```

## B. DVGO runtime

### Why use a separate runtime

DVGO is PyTorch-based and builds CUDA extensions on first use. A clean runtime is simpler and more reliable than trying to co-exist with TensorFlow 1.15.

Use a new fresh Colab runtime for DVGO.

### Recommended DVGO environment

Do not rely on the default modern Colab stack for DVGO.

Your reported failing stack was:

- Python 3.11
- PyTorch 2.6.0
- CUDA 12.4

That is not the target era for this 2022 codebase and its custom CUDA ops. Use the dedicated legacy env:

- [environment_colab_legacy.yml](C:/Users/Jerry/Documents/Code/CityU/CS6493_CG/pj/DirectVoxGO/environment_colab_legacy.yml)

### Install DVGO dependencies

In a fresh Colab notebook:

```python
!pip install -q condacolab
import condacolab
condacolab.install()
```

After the runtime restarts:

```bash
cd /content/cg_nerf
conda env remove -n dvgo-legacy -y || true
conda env create -f DirectVoxGO/environment_colab_legacy.yml
conda run -n dvgo-legacy python -V
```

Install the official PyTorch CUDA wheel from the PyTorch previous-versions page:

```bash
conda run --no-capture-output -n dvgo-legacy pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda run --no-capture-output -n dvgo-legacy python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda)"
```

Expected:

- Python 3.8
- PyTorch 1.12.1+cu113
- CUDA 11.3

Then install `torch_scatter` using the matching PyG wheel:

```bash
conda run --no-capture-output -n dvgo-legacy pip install --no-index torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

If that still fails in your runtime, use the conda package instead:

```bash
conda run --no-capture-output -n dvgo-legacy conda install -y pyg::pytorch-scatter
```

### Prepare DVGO data

Reuse the same NeRF-downloaded data:

```bash
cd /content/cg_nerf
mkdir -p DirectVoxGO/data
cp -r nerf/data/nerf_synthetic DirectVoxGO/data/
cp -r nerf/data/nerf_llff_data DirectVoxGO/data/
```

### Train DVGO

First verify the CUDA extension import:

```bash
cd /content/cg_nerf/DirectVoxGO
conda run --no-capture-output -n dvgo-legacy python -c "from lib import dvgo; print('DVGO CUDA extension loaded successfully.')"
```

20 minute configs:

```bash
cd /content/cg_nerf/DirectVoxGO
time conda run --no-capture-output -n dvgo-legacy python run.py --config configs/course/lego_t4_20min.py --i_tb 100
time conda run --no-capture-output -n dvgo-legacy python run.py --config configs/course/fern_t4_20min.py --i_tb 100
```

30 minute configs:

```bash
cd /content/cg_nerf/DirectVoxGO
time conda run --no-capture-output -n dvgo-legacy python run.py --config configs/course/lego_t4_30min.py --i_tb 100
time conda run --no-capture-output -n dvgo-legacy python run.py --config configs/course/fern_t4_30min.py --i_tb 100
```

### TensorBoard for DVGO

DVGO now writes TensorBoard summaries to:

- `DirectVoxGO/logs/.../summaries/<expname>`

In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/cg_nerf/DirectVoxGO/logs
```

### Evaluate trained DVGO

```bash
cd /content/cg_nerf
conda run --no-capture-output -n dvgo-legacy python project_tools/render_eval_dvgo.py \
  --config DirectVoxGO/configs/course/lego_t4_20min.py \
  --output_dir result/dvgo/lego_trained_20min \
  --source trained \
  --train_time_minutes 20
```

## C. What to save

For NeRF:

- `nerf/logs/<expname>`
- `nerf/logs/summaries/<expname>`
- `result/nerf/...`

For DVGO:

- `DirectVoxGO/logs/.../<expname>`
- `DirectVoxGO/logs/.../summaries/<expname>`
- `result/dvgo/...`

For both:

- the notebook itself
- the exact config files used
- recorded wall-clock times
- `result/summary_metrics.csv`

## D. Notes on code compatibility

NeRF was updated in a minimal way so the same codebase can run in either:

- the original TF1.15-style environment, or
- a newer TF2 eager environment as a fallback

Specific compatibility fixes:

- [load_blender.py](C:/Users/Jerry/Documents/Code/CityU/CS6493_CG/pj/nerf/load_blender.py) now falls back from `tf.image.resize_area` to `tf.image.resize(..., method='area')` when needed.
- [run_nerf.py](C:/Users/Jerry/Documents/Code/CityU/CS6493_CG/pj/nerf/run_nerf.py) now supports both TF1.15 `tf.contrib.summary` style logging and TF2 `tf.summary` style logging.
- DVGO CUDA JIT loading is now centralized in [cuda_utils.py](C:/Users/Jerry/Documents/Code/CityU/CS6493_CG/pj/DirectVoxGO/lib/cuda_utils.py), and the C++ wrappers were updated from `tensor.type().is_cuda()` to `tensor.is_cuda()` for better compatibility with newer PyTorch APIs.

Sources:

- PyTorch previous versions: https://docs.pytorch.org/get-started/previous-versions/
- PyG installation guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

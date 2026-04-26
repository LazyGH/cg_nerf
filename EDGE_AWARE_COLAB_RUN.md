# Edge-Aware NeRF Colab Run Guide

## Environment

```bash
cd /content
git clone <your-remote-repo-url> cg_nerf
cd /content/cg_nerf
conda env create -f nerf/environment_colab_tf115.yml
conda run -n nerf-tf115 python -V
```

Add datasets and any existing baseline checkpoints under the same paths already used by the original configs:

- `nerf/data/nerf_synthetic/lego`
- `nerf/data/nerf_llff_data/fern`
- `nerf/logs/<baseline_expname>/...`

## Train Edge-Aware Models

```bash
cd /content/cg_nerf/nerf
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/lego_edge_alpha06.txt
time conda run -n nerf-tf115 python run_nerf.py --config course_configs/fern_edge_alpha04.txt
```

Train to the same checkpoint step used by the matching baseline if you want a direct quality comparison instead of a convergence-speed comparison.

## Render And Evaluate

Baseline examples:

```bash
cd /content/cg_nerf
conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_t4_20min.txt \
  --output_dir result/nerf/lego_original \
  --source trained \
  --train_time_minutes 20

conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/fern_t4_20min.txt \
  --output_dir result/nerf/fern_original \
  --source trained \
  --train_time_minutes 20
```

Edge-aware examples:

```bash
conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/lego_edge_alpha06.txt \
  --output_dir result/nerf/lego_edge_alpha06 \
  --source trained \
  --train_time_minutes 20

conda run -n nerf-tf115 python project_tools/render_eval_nerf.py \
  --config nerf/course_configs/fern_edge_alpha04.txt \
  --output_dir result/nerf/fern_edge_alpha04 \
  --source trained \
  --train_time_minutes 20
```

## Optional Edge-Region Metrics

```bash
conda run -n nerf-tf115 python project_tools/eval_edge_regions.py --eval_dir result/nerf/lego_original
conda run -n nerf-tf115 python project_tools/eval_edge_regions.py --eval_dir result/nerf/lego_edge_alpha06
conda run -n nerf-tf115 python project_tools/eval_edge_regions.py --eval_dir result/nerf/fern_original
conda run -n nerf-tf115 python project_tools/eval_edge_regions.py --eval_dir result/nerf/fern_edge_alpha04
```

## Aggregate Metrics

```bash
conda run -n nerf-tf115 python project_tools/aggregate_metrics.py \
  result/nerf/lego_original/metrics.json \
  result/nerf/lego_edge_alpha06/metrics.json \
  result/nerf/fern_original/metrics.json \
  result/nerf/fern_edge_alpha04/metrics.json \
  --out_csv result/nerf/summary_metrics.csv
```

If you also run `eval_edge_regions.py`, merge those CSV or JSON outputs into the final report table with columns:

`scene, method, alpha, checkpoint_step, psnr, ssim, lpips, edge_psnr, flat_psnr, notes`

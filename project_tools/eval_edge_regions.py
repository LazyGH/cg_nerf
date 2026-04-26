import argparse
import csv
import json
import os
import sys
from glob import glob

import imageio.v2 as imageio
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NERF_ROOT = os.path.join(PROJECT_ROOT, "nerf")
sys.path.insert(0, NERF_ROOT)

from edge_ray_sampler import rgb_to_luma_np, sobel_grad_mag_np


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval_dir", required=True, help="Directory produced by render_eval_nerf.py")
    parser.add_argument("--out_csv", default="", help="Optional explicit CSV path")
    parser.add_argument("--out_json", default="", help="Optional explicit JSON path")
    parser.add_argument("--edge_percentile", type=float, default=80.0, help="Top percentile threshold for edge mask")
    parser.add_argument("--flat_percentile", type=float, default=50.0, help="Bottom percentile threshold for flat mask")
    return parser.parse_args()


def load_rgb(path):
    image = imageio.imread(path)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        return image / 255.0
    return image


def masked_psnr(pred, gt, mask=None):
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    diff = diff * diff
    if mask is None:
        mse = diff.mean()
    else:
        weights = np.asarray(mask, dtype=np.float64)[..., None]
        denom = np.maximum(weights.sum() * pred.shape[-1], 1.0)
        mse = (diff * weights).sum() / denom
    mse = max(float(mse), 1e-12)
    return float(-10.0 * np.log10(mse))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    pred_paths = sorted(glob(os.path.join(args.eval_dir, "rgb_pred", "*.png")))
    gt_paths = sorted(glob(os.path.join(args.eval_dir, "rgb_gt", "*.png")))
    if not pred_paths or len(pred_paths) != len(gt_paths):
        raise ValueError("Expected matching rgb_pred/*.png and rgb_gt/*.png files.")

    rows = []
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)
        grad = sobel_grad_mag_np(rgb_to_luma_np(gt))
        edge_thresh = np.percentile(grad, args.edge_percentile)
        flat_thresh = np.percentile(grad, args.flat_percentile)
        edge_mask = grad >= edge_thresh
        flat_mask = grad <= flat_thresh
        rows.append(
            {
                "frame_name": os.path.splitext(os.path.basename(pred_path))[0],
                "psnr": masked_psnr(pred, gt),
                "edge_psnr": masked_psnr(pred, gt, edge_mask),
                "flat_psnr": masked_psnr(pred, gt, flat_mask),
                "edge_fraction": float(edge_mask.mean()),
                "flat_fraction": float(flat_mask.mean()),
            }
        )

    avg = {
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "edge_psnr": float(np.mean([row["edge_psnr"] for row in rows])),
        "flat_psnr": float(np.mean([row["flat_psnr"] for row in rows])),
    }
    out_csv = args.out_csv or os.path.join(args.eval_dir, "edge_region_metrics.csv")
    out_json = args.out_json or os.path.join(args.eval_dir, "edge_region_metrics.json")
    write_csv(out_csv, rows)
    write_json(out_json, {"avg": avg, "per_image": rows})
    print(json.dumps({"avg": avg, "out_csv": os.path.abspath(out_csv), "out_json": os.path.abspath(out_json)}, indent=2))


if __name__ == "__main__":
    main()

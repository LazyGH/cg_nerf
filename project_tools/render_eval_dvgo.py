import argparse
import os
import sys

import mmcv
import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DVGO_ROOT = os.path.join(PROJECT_ROOT, "DirectVoxGO")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, DVGO_ROOT)

from eval_utils import (  # noqa: E402
    LpipsComputer,
    colorize_scalar_map,
    compute_scalar_vis_range,
    ensure_dir,
    make_error_map,
    psnr,
    save_array,
    save_rgb,
    ssim,
    write_csv,
    write_json,
)
from lib import dcvgo, dmpigo, dvgo, utils  # noqa: E402
from lib.load_data import load_data  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="DVGO config file")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation artifacts")
    parser.add_argument("--ckpt", default="", help="Optional explicit checkpoint .tar path")
    parser.add_argument("--source", default="trained", help="Label such as trained")
    parser.add_argument("--train_time_minutes", type=float, default=None, help="Optional wall-clock training time")
    parser.add_argument("--lpips_nets", nargs="*", default=["alex", "vgg"], help="LPIPS backbones to evaluate")
    return parser.parse_args()


def resolve_repo_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(DVGO_ROOT, path))


def resolve_latest_ckpt(cfg):
    fine_path = os.path.join(cfg.basedir, cfg.expname, "fine_last.tar")
    coarse_path = os.path.join(cfg.basedir, cfg.expname, "coarse_last.tar")
    if os.path.isfile(fine_path):
        return fine_path
    if os.path.isfile(coarse_path):
        return coarse_path
    raise FileNotFoundError(f"No DVGO checkpoint found under {os.path.join(cfg.basedir, cfg.expname)}")


def select_model_class(cfg):
    if cfg.data.ndc:
        return dmpigo.DirectMPIGO
    if cfg.data.unbounded_inward:
        return dcvgo.DirectContractedVoxGO
    return dvgo.DirectVoxGO


@torch.no_grad()
def render_single_view(model, cfg, render_kwargs, H, W, K, c2w):
    rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
        H,
        W,
        K,
        torch.as_tensor(c2w, device=next(model.parameters()).device),
        cfg.data.ndc,
        inverse_y=cfg.data.inverse_y,
        flip_x=cfg.data.flip_x,
        flip_y=cfg.data.flip_y,
    )
    keys = ["rgb_marched", "depth", "alphainv_last"]
    rays_o = rays_o.flatten(0, -2)
    rays_d = rays_d.flatten(0, -2)
    viewdirs = viewdirs.flatten(0, -2)
    chunks = []
    for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0)):
        chunks.append({k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys})
    render_result = {
        key: torch.cat([chunk[key] for chunk in chunks]).reshape(H, W, -1)
        for key in chunks[0].keys()
    }
    rgb = render_result["rgb_marched"].cpu().numpy().astype(np.float32)
    depth = render_result["depth"].cpu().numpy().astype(np.float32)
    bgmap = render_result["alphainv_last"].cpu().numpy().astype(np.float32)
    return rgb, depth, bgmap


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    cfg = mmcv.Config.fromfile(config_path)
    cfg.basedir = resolve_repo_path(cfg.basedir)
    cfg.data.datadir = resolve_repo_path(cfg.data.datadir)
    ckpt_path = resolve_repo_path(args.ckpt) if args.ckpt else resolve_latest_ckpt(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = select_model_class(cfg)
    model = utils.load_model(model_class, ckpt_path).to(device)
    model.eval()

    data_dict = load_data(cfg.data)
    images = data_dict["images"]
    poses = data_dict["poses"]
    HW = data_dict["HW"]
    Ks = data_dict["Ks"]
    i_test = np.array(data_dict["i_test"])

    render_kwargs = {
        "near": data_dict["near"],
        "far": data_dict["far"],
        "bg": 1 if cfg.data.white_bkgd else 0,
        "stepsize": cfg.fine_model_and_render.stepsize,
        "inverse_y": cfg.data.inverse_y,
        "flip_x": cfg.data.flip_x,
        "flip_y": cfg.data.flip_y,
        "render_depth": True,
    }

    output_dir = os.path.abspath(args.output_dir)
    rgb_dir = os.path.join(output_dir, "rgb_pred")
    gt_dir = os.path.join(output_dir, "rgb_gt")
    depth_dir = os.path.join(output_dir, "depth_vis")
    depth_npy_dir = os.path.join(output_dir, "depth_raw")
    bgmap_npy_dir = os.path.join(output_dir, "bgmap_raw")
    error_dir = os.path.join(output_dir, "error_map")
    for path in [output_dir, rgb_dir, gt_dir, depth_dir, depth_npy_dir, bgmap_npy_dir, error_dir]:
        ensure_dir(path)

    lpips_fns = {name: LpipsComputer(name) for name in args.lpips_nets}
    metrics_rows = []
    pred_rgbs = []
    gt_rgbs = []
    depths = []
    bgmaps = []

    for out_idx, img_idx in enumerate(i_test.tolist()):
        gt = np.asarray(images[img_idx], dtype=np.float32)
        rgb, depth, bgmap = render_single_view(
            model,
            cfg,
            render_kwargs,
            int(HW[img_idx][0]),
            int(HW[img_idx][1]),
            Ks[img_idx],
            poses[img_idx],
        )

        pred_rgbs.append(rgb)
        gt_rgbs.append(gt)
        depths.append(depth)
        bgmaps.append(bgmap)

        save_rgb(os.path.join(rgb_dir, f"{out_idx:03d}.png"), rgb)
        save_rgb(os.path.join(gt_dir, f"{out_idx:03d}.png"), gt)
        save_array(os.path.join(depth_npy_dir, f"{out_idx:03d}.npy"), depth)
        save_array(os.path.join(bgmap_npy_dir, f"{out_idx:03d}.npy"), bgmap)

        row = {
            "image_index": int(img_idx),
            "frame_name": f"{out_idx:03d}",
            "psnr": psnr(rgb, gt),
            "ssim": ssim(rgb, gt),
        }
        for name, lpips_fn in lpips_fns.items():
            row[f"lpips_{name}"] = lpips_fn(rgb, gt)

        error_raw, error_vis = make_error_map(rgb, gt)
        save_rgb(os.path.join(error_dir, f"{out_idx:03d}.png"), error_vis)
        save_array(os.path.join(error_dir, f"{out_idx:03d}.npy"), error_raw)
        metrics_rows.append(row)

    pred_rgbs = np.stack(pred_rgbs, axis=0)
    gt_rgbs = np.stack(gt_rgbs, axis=0)
    depths = np.stack(depths, axis=0)
    bgmaps = np.stack(bgmaps, axis=0)
    valid_mask = bgmaps[..., 0] < 0.1 if bgmaps.ndim == 4 else bgmaps < 0.1
    depth_values = depths[..., 0] if depths.ndim == 4 else depths
    vmin, vmax = compute_scalar_vis_range(depth_values, valid_mask=valid_mask)

    for idx in range(depth_values.shape[0]):
        background_mask = ~valid_mask[idx]
        depth_vis = colorize_scalar_map(depth_values[idx], vmin, vmax, cmap_name="turbo", background_mask=background_mask)
        save_rgb(os.path.join(depth_dir, f"{idx:03d}.png"), depth_vis)

    summary = {
        "method": "DVGO",
        "scene": os.path.basename(os.path.normpath(cfg.data.datadir)),
        "dataset_type": cfg.data.dataset_type,
        "source": args.source,
        "config": config_path,
        "checkpoint": ckpt_path,
        "expname": cfg.expname,
        "train_time_minutes": args.train_time_minutes,
        "num_test_images": int(len(metrics_rows)),
        "avg": {
            "psnr": float(np.mean([row["psnr"] for row in metrics_rows])),
            "ssim": float(np.mean([row["ssim"] for row in metrics_rows])),
        },
        "per_image": metrics_rows,
    }
    for name in lpips_fns:
        summary["avg"][f"lpips_{name}"] = float(np.mean([row[f"lpips_{name}"] for row in metrics_rows]))

    save_array(os.path.join(output_dir, "pred_rgb_stack.npy"), pred_rgbs)
    save_array(os.path.join(output_dir, "gt_rgb_stack.npy"), gt_rgbs)
    save_array(os.path.join(output_dir, "depth_stack.npy"), depth_values)
    save_array(os.path.join(output_dir, "bgmap_stack.npy"), bgmaps)
    write_json(os.path.join(output_dir, "metrics.json"), summary)
    write_csv(os.path.join(output_dir, "metrics.csv"), metrics_rows, fieldnames=list(metrics_rows[0].keys()))


if __name__ == "__main__":
    main()

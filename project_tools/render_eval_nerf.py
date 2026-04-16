import argparse
import os
import sys
from glob import glob

import numpy as np
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NERF_ROOT = os.path.join(PROJECT_ROOT, "nerf")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, NERF_ROOT)

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
from load_blender import load_blender_data  # noqa: E402
from load_deepvoxels import load_dv_data  # noqa: E402
from load_llff import load_llff_data  # noqa: E402
import run_nerf as nerf_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="NeRF config txt file")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation artifacts")
    parser.add_argument("--ckpt", default="", help="Optional explicit checkpoint .npy path")
    parser.add_argument("--source", default="trained", help="Label such as trained or pretrained")
    parser.add_argument("--train_time_minutes", type=float, default=None, help="Optional wall-clock training time")
    parser.add_argument("--lpips_nets", nargs="*", default=["alex", "vgg"], help="LPIPS backbones to evaluate")
    return parser.parse_args()


def resolve_repo_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(NERF_ROOT, path))


def resolve_latest_ckpt(basedir, expname):
    expdir = os.path.join(basedir, expname)
    if not os.path.isdir(expdir):
        raise FileNotFoundError(f"Experiment directory not found: {expdir}")
    ckpts = [
        os.path.join(expdir, name)
        for name in sorted(os.listdir(expdir))
        if ("model_" in name and "fine" not in name and name.endswith(".npy"))
    ]
    if not ckpts:
        raise FileNotFoundError(f"No NeRF checkpoints found in {expdir}")
    return ckpts[-1]


def load_dataset(args):
    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:
            i_test = np.arange(images.shape[0])[::args.llffhold]
        near = 0.0 if not args.no_ndc else float(np.min(bds) * 0.9)
        far = 1.0 if not args.no_ndc else float(np.max(bds) * 1.0)
    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        _, _, i_test = i_split
        near, far = 2.0, 6.0
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == "deepvoxels":
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip
        )
        _, _, i_test = i_split
        hemi_r = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near, far = hemi_r - 1.0, hemi_r + 1.0
    else:
        raise NotImplementedError(f"Unsupported dataset type: {args.dataset_type}")

    H, W, focal = hwf
    hwf = [int(H), int(W), focal]
    return {
        "images": images.astype(np.float32),
        "poses": poses.astype(np.float32),
        "render_poses": render_poses,
        "i_test": np.array(i_test),
        "hwf": hwf,
        "near": near,
        "far": far,
    }


def main():
    cli_args = parse_args()
    config_path = os.path.abspath(cli_args.config)

    parser = nerf_run.config_parser()
    argv = ["--config", config_path]
    if cli_args.ckpt:
        argv += ["--ft_path", cli_args.ckpt]
    args = parser.parse_args(argv)
    args.config = resolve_repo_path(args.config)
    args.basedir = resolve_repo_path(args.basedir)
    args.datadir = resolve_repo_path(args.datadir)
    if cli_args.ckpt:
        args.ft_path = resolve_repo_path(cli_args.ckpt)
    elif args.ft_path:
        args.ft_path = resolve_repo_path(args.ft_path)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    if not args.ft_path:
        args.ft_path = resolve_latest_ckpt(args.basedir, args.expname)

    dataset = load_dataset(args)
    render_kwargs_train, render_kwargs_test, start, _, _ = nerf_run.create_nerf(args)
    del render_kwargs_train
    render_kwargs_test.update({"near": tf.cast(dataset["near"], tf.float32), "far": tf.cast(dataset["far"], tf.float32)})

    images = dataset["images"]
    poses = dataset["poses"]
    i_test = dataset["i_test"]
    H, W, focal = dataset["hwf"]

    output_dir = os.path.abspath(cli_args.output_dir)
    rgb_dir = os.path.join(output_dir, "rgb_pred")
    gt_dir = os.path.join(output_dir, "rgb_gt")
    disp_dir = os.path.join(output_dir, "disp_vis")
    disp_npy_dir = os.path.join(output_dir, "disp_raw")
    acc_npy_dir = os.path.join(output_dir, "acc_raw")
    error_dir = os.path.join(output_dir, "error_map")
    for path in [output_dir, rgb_dir, gt_dir, disp_dir, disp_npy_dir, acc_npy_dir, error_dir]:
        ensure_dir(path)

    lpips_fns = {name: LpipsComputer(name) for name in cli_args.lpips_nets}
    metrics_rows = []
    pred_rgbs = []
    gt_rgbs = []
    disps = []
    accs = []

    for out_idx, img_idx in enumerate(i_test.tolist()):
        pose = poses[img_idx, :3, :4]
        gt = images[img_idx]
        rgb, disp, acc, _ = nerf_run.render(H, W, focal, chunk=args.chunk, c2w=pose, **render_kwargs_test)
        rgb = rgb.numpy().astype(np.float32)
        disp = disp.numpy().astype(np.float32)
        acc = acc.numpy().astype(np.float32)

        pred_rgbs.append(rgb)
        gt_rgbs.append(gt)
        disps.append(disp)
        accs.append(acc)

        save_rgb(os.path.join(rgb_dir, f"{out_idx:03d}.png"), rgb)
        save_rgb(os.path.join(gt_dir, f"{out_idx:03d}.png"), gt)
        save_array(os.path.join(disp_npy_dir, f"{out_idx:03d}.npy"), disp)
        save_array(os.path.join(acc_npy_dir, f"{out_idx:03d}.npy"), acc)

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
    disps = np.stack(disps, axis=0)
    accs = np.stack(accs, axis=0)
    valid_mask = accs > 1e-3
    vmin, vmax = compute_scalar_vis_range(disps, valid_mask=valid_mask)

    for idx in range(disps.shape[0]):
        disp_vis = colorize_scalar_map(disps[idx], vmin, vmax, cmap_name="magma", background_mask=~valid_mask[idx])
        save_rgb(os.path.join(disp_dir, f"{idx:03d}.png"), disp_vis)

    summary = {
        "method": "NeRF",
        "scene": os.path.basename(os.path.normpath(args.datadir)),
        "dataset_type": args.dataset_type,
        "source": cli_args.source,
        "config": args.config,
        "checkpoint": args.ft_path,
        "expname": args.expname,
        "train_time_minutes": cli_args.train_time_minutes,
        "num_test_images": int(len(metrics_rows)),
        "checkpoint_step": int(start - 1) if start > 0 else None,
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
    save_array(os.path.join(output_dir, "disp_stack.npy"), disps)
    save_array(os.path.join(output_dir, "acc_stack.npy"), accs)
    write_json(os.path.join(output_dir, "metrics.json"), summary)
    write_csv(os.path.join(output_dir, "metrics.csv"), metrics_rows, fieldnames=list(metrics_rows[0].keys()))


if __name__ == "__main__":
    main()

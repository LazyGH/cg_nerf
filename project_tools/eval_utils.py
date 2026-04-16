import csv
import json
import os

import imageio.v2 as imageio
import numpy as np
import scipy.signal
import torch
from matplotlib import cm


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to8b(x):
    return (255 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path, image):
    imageio.imwrite(path, to8b(image))


def save_array(path, array):
    np.save(path, array)


def psnr(pred, gt):
    mse = np.mean(np.square(pred - gt), dtype=np.float64)
    mse = max(mse, 1e-12)
    return float(-10.0 * np.log10(mse))


def ssim(img0, img1, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    assert img0.shape == img1.shape and img0.shape[-1] == 3
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode="valid")

    def filt_fn(z):
        return np.stack(
            [convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :]) for i in range(z.shape[-1])],
            axis=-1,
        )

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    return float(np.mean(numer / denom))


class LpipsComputer:
    def __init__(self, net_name="alex", device=None):
        import lpips

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net_name = net_name
        self.model = lpips.LPIPS(net=net_name, version="0.1").eval().to(self.device)

    @torch.no_grad()
    def __call__(self, pred, gt):
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).contiguous().to(self.device)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).contiguous().to(self.device)
        return float(self.model(pred_t, gt_t, normalize=True).item())


def compute_scalar_vis_range(arrays, valid_mask=None, lower=5.0, upper=95.0):
    flat = np.asarray(arrays).reshape(-1)
    if valid_mask is not None:
        flat = flat[np.asarray(valid_mask).reshape(-1)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(flat, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def colorize_scalar_map(values, vmin, vmax, cmap_name="magma", background_mask=None):
    values = np.asarray(values, dtype=np.float32)
    scale = np.clip((values - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    colored = cm.get_cmap(cmap_name)(scale)[..., :3]
    if background_mask is not None:
        colored = np.where(background_mask[..., None], 1.0, colored)
    return colored.astype(np.float32)


def make_error_map(pred, gt):
    error = np.mean(np.abs(pred - gt), axis=-1)
    vmax = max(float(np.percentile(error, 99.0)), 1e-6)
    vis = colorize_scalar_map(error, 0.0, vmax, cmap_name="inferno")
    return error.astype(np.float32), vis

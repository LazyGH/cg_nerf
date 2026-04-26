import numpy as np


_SOBEL_X = np.array(
    [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
    dtype=np.float32,
)
_SOBEL_Y = np.array(
    [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
    dtype=np.float32,
)
_SMOOTH_3X3 = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    dtype=np.float32,
) / 16.0


def _conv3x3_reflect(x, kernel):
    x = np.asarray(x, dtype=np.float32)
    padded = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    return (
        kernel[0, 0] * padded[:-2, :-2]
        + kernel[0, 1] * padded[:-2, 1:-1]
        + kernel[0, 2] * padded[:-2, 2:]
        + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 0] * padded[2:, :-2]
        + kernel[2, 1] * padded[2:, 1:-1]
        + kernel[2, 2] * padded[2:, 2:]
    )


def rgb_to_luma_np(img):
    """img: [H, W, 3] float image in [0, 1]. Return [H, W] luma."""
    img = np.asarray(img, dtype=np.float32)
    return (
        0.299 * img[..., 0]
        + 0.587 * img[..., 1]
        + 0.114 * img[..., 2]
    ).astype(np.float32)


def sobel_grad_mag_np(gray):
    """Compute Sobel gradient magnitude with numpy only."""
    gray = np.asarray(gray, dtype=np.float32)
    grad_x = _conv3x3_reflect(gray, _SOBEL_X)
    grad_y = _conv3x3_reflect(gray, _SOBEL_Y)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)


def smooth3x3_np(x):
    """Optional dependency-free 3x3 smoothing."""
    return _conv3x3_reflect(x, _SMOOTH_3X3).astype(np.float32)


def build_edge_prob_map(img, alpha, eps=1e-6, smooth=False):
    """
    Return a flattened [H*W] probability vector.
    P_mix = alpha * normalized_edge + (1 - alpha) * uniform.
    """
    gray = rgb_to_luma_np(img)
    edge = sobel_grad_mag_np(gray)
    if smooth:
        edge = smooth3x3_np(edge)

    edge = np.asarray(edge, dtype=np.float64) + float(eps)
    edge_sum = edge.sum()
    if not np.isfinite(edge_sum) or edge_sum <= 0.0:
        raise ValueError("Invalid edge-map normalization sum.")

    edge_prob = (edge / edge_sum).reshape(-1)
    uniform_prob = np.full(edge_prob.shape, 1.0 / edge_prob.size, dtype=np.float64)
    mixed = float(alpha) * edge_prob + (1.0 - float(alpha)) * uniform_prob
    mixed_sum = mixed.sum()
    mixed = mixed / mixed_sum
    return mixed.astype(np.float64)


def pdf_to_cdf(prob):
    prob = np.asarray(prob, dtype=np.float64)
    cdf = np.cumsum(prob)
    cdf[-1] = 1.0
    return cdf


def cdf_to_pdf(cdf):
    cdf = np.asarray(cdf, dtype=np.float64)
    pdf = np.empty_like(cdf)
    pdf[0] = cdf[0]
    pdf[1:] = cdf[1:] - cdf[:-1]
    return pdf


def build_edge_cdfs(images, i_train, alpha, eps=1e-6, smooth=False):
    """
    Return:
      per_image_samplers: dict mapping image index -> {'prob': prob, 'cdf': cdf}
      global_cdf: CDF over concatenated train images in the same order used by rays_rgb
    """
    per_image_samplers = {}
    global_probs = []
    for img_i in np.asarray(i_train).tolist():
        prob = build_edge_prob_map(images[img_i], alpha=alpha, eps=eps, smooth=smooth)
        per_image_samplers[int(img_i)] = {
            "prob": prob,
            "cdf": pdf_to_cdf(prob),
        }
        global_probs.append(prob)

    global_probs = np.concatenate(global_probs, axis=0)
    global_probs = global_probs / global_probs.sum()
    global_cdf = pdf_to_cdf(global_probs)
    return per_image_samplers, global_cdf


def sample_from_cdf(cdf, n):
    """Fast weighted sampling with replacement using np.searchsorted."""
    cdf = np.asarray(cdf, dtype=np.float64)
    u = np.random.random_sample(int(n))
    inds = np.searchsorted(cdf, u, side="left")
    return np.clip(inds, 0, cdf.shape[0] - 1)

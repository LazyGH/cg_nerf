import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
NERF_ROOT = os.path.join(PROJECT_ROOT, "nerf")
sys.path.insert(0, NERF_ROOT)

from edge_ray_sampler import build_edge_prob_map, build_edge_cdfs, sample_from_cdf


def test_constant_image():
    img = np.full((6, 6, 3), 0.5, dtype=np.float32)
    prob = build_edge_prob_map(img, alpha=0.6, eps=1e-6, smooth=True)
    expected = np.full(prob.shape, 1.0 / prob.size, dtype=np.float64)
    assert np.isclose(prob.sum(), 1.0)
    assert np.all(np.isfinite(prob))
    assert np.all(prob > 0.0)
    assert np.allclose(prob, expected, atol=1e-6)


def test_vertical_boundary_bias():
    img = np.zeros((8, 8, 3), dtype=np.float32)
    img[:, 4:, :] = 1.0
    prob = build_edge_prob_map(img, alpha=0.8, eps=1e-6, smooth=False).reshape(8, 8)
    boundary_mass = prob[:, 3:5].mean()
    flat_mass = np.concatenate([prob[:, :2].reshape(-1), prob[:, 6:].reshape(-1)]).mean()
    assert boundary_mass > flat_mass


def test_cdf_sampling():
    images = np.zeros((2, 4, 4, 3), dtype=np.float32)
    images[0, :, 2:, :] = 1.0
    images[1, 1:3, 1:3, :] = 1.0
    samplers, global_cdf = build_edge_cdfs(images, i_train=np.array([0, 1]), alpha=0.5, smooth=True)
    samples = sample_from_cdf(global_cdf, 500)
    assert samples.min() >= 0
    assert samples.max() < global_cdf.size
    assert np.all(np.isfinite(global_cdf))
    assert np.isclose(global_cdf[-1], 1.0)
    for sampler in samplers.values():
        assert np.isclose(sampler["prob"].sum(), 1.0)
        assert np.isclose(sampler["cdf"][-1], 1.0)


def main():
    test_constant_image()
    test_vertical_boundary_bias()
    test_cdf_sampling()
    print("edge_ray_sampler smoke tests passed")


if __name__ == "__main__":
    main()

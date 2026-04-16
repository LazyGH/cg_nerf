_base_ = '../nerf/lego.py'

expname = 'dvgo_lego_t4_20min'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
    half_res=True,
)

coarse_train = dict(
    N_iters=200,
    N_rand=4096,
    pg_scale=[],
)

fine_train = dict(
    N_iters=5000,
    N_rand=4096,
    pg_scale=[1000, 2500, 4000],
)

coarse_model_and_render = dict(
    num_voxels=80**3,
    num_voxels_base=80**3,
)

fine_model_and_render = dict(
    num_voxels=128**3,
    num_voxels_base=128**3,
    stepsize=0.5,
)

_base_ = '../nerf/lego.py'

expname = 'dvgo_lego_t4_30min'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
    half_res=True,
)

coarse_train = dict(
    N_iters=1000,
    N_rand=4096,
    pg_scale=[500],
)

fine_train = dict(
    N_iters=7000,
    N_rand=4096,
    pg_scale=[1000, 2000, 3000, 5000],
)

coarse_model_and_render = dict(
    num_voxels=96**3,
    num_voxels_base=96**3,
)

fine_model_and_render = dict(
    num_voxels=140**3,
    num_voxels_base=140**3,
    stepsize=0.5,
)

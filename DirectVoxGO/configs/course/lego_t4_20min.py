_base_ = '../nerf/lego.py'

expname = 'dvgo_lego_t4_20min'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
    half_res=True,
    load2gpu_on_the_fly=True,
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=1000,
    N_rand=512,
    ray_sampler='random',
    pg_scale=[600, 1300],
    pervoxel_lr=False,
)

coarse_model_and_render = dict(
    num_voxels=32**3,
    num_voxels_base=36**3,
)

fine_model_and_render = dict(
    num_voxels=48**3,
    num_voxels_base=64**3,
    stepsize=0.5,
)

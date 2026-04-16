_base_ = '../llff/fern.py'

expname = 'dvgo_fern_t4_30min'

data = dict(
    datadir='./data/nerf_llff_data/fern',
    dataset_type='llff',
    ndc=True,
    factor=8,
    width=None,
    height=None,
    llffhold=8,
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=10000,
    N_rand=4096,
    pg_scale=[2000, 4000, 6000, 8000],
    tv_dense_before=6000,
)

fine_model_and_render = dict(
    num_voxels=192**3,
    num_voxels_base=192**3,
    stepsize=0.5,
)

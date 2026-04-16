_base_ = '../llff/fern.py'

expname = 'dvgo_fern_t4_test'

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
    N_iters=5,
    N_rand=4096,
    pg_scale=[1500, 3000, 5000],
    tv_dense_before=4000,
)

fine_model_and_render = dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    stepsize=0.5,
)

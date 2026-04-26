import os
import sys

import torch
from torch.utils.cpp_extension import load


def _extension_build_root():
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root:
        return root
    return os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")


def load_cuda_extension(name, parent_dir, sources):
    build_root = _extension_build_root()
    cuda_tag = (torch.version.cuda or "cpu").replace(".", "")
    build_dir = os.path.join(
        build_root,
        f"py{sys.version_info.major}{sys.version_info.minor}_cu{cuda_tag}",
        name,
    )
    os.makedirs(build_dir, exist_ok=True)

    try:
        return load(
            name=name,
            sources=[os.path.join(parent_dir, path) for path in sources],
            build_directory=build_dir,
            verbose=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to JIT-compile CUDA extension '{name}'. "
            f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, Python={sys.version.split()[0]}. "
            "DVGO is a 2022 codebase with custom CUDA ops. "
            "If you are on a modern Colab runtime, use the legacy environment in "
            "DirectVoxGO/environment_colab_legacy.yml."
        ) from exc

import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch as th
from filelock import FileLock
from tqdm.auto import tqdm

MODEL_PATHS = {
    "base": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt",
    "upsample": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/upsample.pt",
    "base-inpaint": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base_inpaint.pt",
    "upsample-inpaint": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/upsample_inpaint.pt",
    "clip/image-enc": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/clip_image_enc.pt",
    "clip/text-enc": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/clip_text_enc.pt",
}


@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "glide_model_cache")


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.

    If cache_dir is specified, it will be used to download the files.
    Otherwise, default_cache_dir() is used.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    local_path = os.path.join(cache_dir, url.split("/")[-1])
    with FileLock(local_path + ".lock"):
        if os.path.exists(local_path):
            return local_path
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


def load_checkpoint(
    checkpoint_name: str,
    device: th.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, th.Tensor]:
    if checkpoint_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    return th.load(path, map_location=device)

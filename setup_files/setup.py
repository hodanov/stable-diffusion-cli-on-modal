from __future__ import annotations

import os

import diffusers
from modal import Image, Mount, Secret, Stub

BASE_CACHE_PATH = "/vol/cache"
BASE_CACHE_PATH_LORA = "/vol/cache/lora"
BASE_CACHE_PATH_TEXTUAL_INVERSION = "/vol/cache/textual_inversion"
BASE_CACHE_PATH_CONTROLNET = "/vol/cache/controlnet"


def download_file(url, file_name, file_path):
    """
    Download files.
    """
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    downloaded = urlopen(req).read()
    dir_names = os.path.join(file_path, file_name)
    os.makedirs(os.path.dirname(dir_names), exist_ok=True)
    with open(dir_names, mode="wb") as f:
        f.write(downloaded)


def download_controlnet(name: str, repo_id: str, token: str):
    """
    Download a controlnet.
    """
    cache_path = os.path.join(BASE_CACHE_PATH_CONTROLNET, name)
    controlnet = diffusers.ControlNetModel.from_pretrained(
        repo_id,
        use_auth_token=token,
        cache_dir=cache_path,
    )
    controlnet.save_pretrained(cache_path, safe_serialization=True)


def download_vae(name: str, repo_id: str, token: str):
    """
    Download a vae.
    """
    cache_path = os.path.join(BASE_CACHE_PATH, name)
    vae = diffusers.AutoencoderKL.from_pretrained(
        repo_id,
        use_auth_token=token,
        cache_dir=cache_path,
    )
    vae.save_pretrained(cache_path, safe_serialization=True)


def download_model(name: str, repo_id: str, token: str):
    """
    Download a model.
    """
    cache_path = os.path.join(BASE_CACHE_PATH, name)
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        repo_id,
        use_auth_token=token,
        cache_dir=cache_path,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


def build_image():
    """
    Build the Docker image.
    """
    import yaml

    token = os.environ["HUGGING_FACE_TOKEN"]
    config = {}
    with open("/config.yml", "r") as file:
        config = yaml.safe_load(file)

    model = config.get("model")
    if model is not None:
        download_model(name=model["name"], repo_id=model["repo_id"], token=token)

    vae = config.get("vae")
    if vae is not None:
        download_vae(name=model["name"], repo_id=vae["repo_id"], token=token)

    controlnets = config.get("controlnets")
    if controlnets is not None:
        for controlnet in controlnets:
            download_controlnet(name=controlnet["name"], repo_id=controlnet["repo_id"], token=token)

    loras = config.get("loras")
    if loras is not None:
        for lora in loras:
            download_file(
                url=lora["download_url"],
                file_name=lora["name"],
                file_path=BASE_CACHE_PATH_LORA,
            )

    textual_inversions = config.get("textual_inversions")
    if textual_inversions is not None:
        for textual_inversion in textual_inversions:
            download_file(
                url=textual_inversion["download_url"],
                file_name=textual_inversion["name"],
                file_path=BASE_CACHE_PATH_TEXTUAL_INVERSION,
            )


stub = Stub("stable-diffusion-cli")
base_stub = Image.from_dockerfile(
    path="Dockerfile",
    context_mount=Mount.from_local_file("requirements.txt"),
)
stub.image = base_stub.extend(
    dockerfile_commands=[
        "FROM base",
        "COPY config.yml /",
    ],
    context_mount=Mount.from_local_file("config.yml"),
).run_function(
    build_image,
    secrets=[Secret.from_dotenv(__file__)],
)

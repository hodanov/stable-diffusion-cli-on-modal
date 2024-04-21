from __future__ import annotations

import os

import diffusers
from modal import App, Image, Mount, Secret

BASE_CACHE_PATH = "/vol/cache"
BASE_CACHE_PATH_LORA = "/vol/cache/lora"
BASE_CACHE_PATH_TEXTUAL_INVERSION = "/vol/cache/textual_inversion"
BASE_CACHE_PATH_CONTROLNET = "/vol/cache/controlnet"
BASE_CACHE_PATH_UPSCALER = "/vol/cache/upscaler"


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


def download_upscaler():
    """
    Download the stabilityai/sd-x2-latent-upscaler.
    """
    model_id = "stabilityai/sd-x2-latent-upscaler"
    upscaler = diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained(model_id)
    upscaler.save_pretrained(BASE_CACHE_PATH_UPSCALER, safe_serialization=True)


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


def download_vae(name: str, model_url: str, token: str):
    """
    Download a vae.
    """
    cache_path = os.path.join(BASE_CACHE_PATH, name)
    vae = diffusers.AutoencoderKL.from_single_file(
        pretrained_model_link_or_path=model_url,
        use_auth_token=token,
        cache_dir=cache_path,
    )
    vae.save_pretrained(cache_path, safe_serialization=True)


def download_model(name: str, model_url: str, token: str):
    """
    Download a model.
    """
    cache_path = os.path.join(BASE_CACHE_PATH, name)
    pipe = diffusers.StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=model_url,
        token=token,
        cache_dir=cache_path,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


def download_model_sdxl(name: str, model_url: str, token: str):
    """
    Download a sdxl model.
    """
    cache_path = os.path.join(BASE_CACHE_PATH, name)
    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
        pretrained_model_link_or_path=model_url,
        use_auth_token=token,
        cache_dir=cache_path,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)

    refiner_cache_path = cache_path + "-refiner"
    refiner = diffusers.StableDiffusionXLImg2ImgPipeline.from_single_file(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors",
        cache_dir=refiner_cache_path,
    )
    refiner.save_pretrained(refiner_cache_path, safe_serialization=True)


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
    use_xl = config.get("use_xl")
    if model is not None:
        if use_xl is not None and use_xl:
            download_model_sdxl(name=model["name"], model_url=model["url"], token=token)
        else:
            download_model(name=model["name"], model_url=model["url"], token=token)

    vae = config.get("vae")
    if vae is not None:
        download_vae(name=model["name"], model_url=vae["url"], token=token)

    controlnets = config.get("controlnets")
    if controlnets is not None:
        for controlnet in controlnets:
            download_controlnet(name=controlnet["name"], repo_id=controlnet["repo_id"], token=token)

    loras = config.get("loras")
    if loras is not None:
        for lora in loras:
            download_file(
                url=lora["url"],
                file_name=lora["name"],
                file_path=BASE_CACHE_PATH_LORA,
            )

    textual_inversions = config.get("textual_inversions")
    if textual_inversions is not None:
        for textual_inversion in textual_inversions:
            download_file(
                url=textual_inversion["url"],
                file_name=textual_inversion["name"],
                file_path=BASE_CACHE_PATH_TEXTUAL_INVERSION,
            )

    download_upscaler()


app = App("stable-diffusion-cli")
base_stub = Image.from_dockerfile(
    path="Dockerfile",
    context_mount=Mount.from_local_file("requirements.txt"),
)
app.image = base_stub.dockerfile_commands(
    "FROM base",
    "COPY config.yml /",
    context_mount=Mount.from_local_file("config.yml"),
).run_function(
    build_image,
    secrets=[Secret.from_dotenv(__file__)],
)

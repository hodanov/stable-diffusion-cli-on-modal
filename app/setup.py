import os
from abc import ABC, abstractmethod
from pathlib import Path

import diffusers
from modal import App, Image, Mount, Secret

BASE_CACHE_PATH = "/vol/cache"
BASE_CACHE_PATH_LORA = "/vol/cache/lora"
BASE_CACHE_PATH_TEXTUAL_INVERSION = "/vol/cache/textual_inversion"
BASE_CACHE_PATH_CONTROLNET = "/vol/cache/controlnet"
BASE_CACHE_PATH_UPSCALER = "/vol/cache/upscaler"


class StableDiffusionCLISetupInterface(ABC):
    @abstractmethod
    def download_model(self) -> None:
        pass


class StableDiffusionCLISetupSDXL(StableDiffusionCLISetupInterface):
    def __init__(self, config: dict, token: str) -> None:
        if config.get("version") != "sdxl":
            msg = "Invalid version. Must be 'sdxl'."
            raise ValueError(msg)

        if config.get("model") is None:
            msg = "Model is required. Please provide a model in config.yml."
            raise ValueError(msg)

        self.__model_name: str = config["model"]["name"]
        self.__model_url: str = config["model"]["url"]
        self.__token: str = token

    def download_model(self) -> None:
        cache_path = Path(BASE_CACHE_PATH, self.__model_name)
        pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=self.__model_url,
            use_auth_token=self.__token,
            cache_dir=cache_path,
        )
        pipe.save_pretrained(cache_path, safe_serialization=True)


class StableDiffusionCLISetupSD15(StableDiffusionCLISetupInterface):
    def __init__(self, config: dict, token: str) -> None:
        if config.get("version") != "sd15":
            msg = "Invalid version. Must be 'sd15'."
            raise ValueError(msg)

        if config.get("model") is None:
            msg = "Model is required. Please provide a model in config.yml."
            raise ValueError(msg)

        self.__model_name: str = config["model"]["name"]
        self.__model_url: str = config["model"]["url"]
        self.__token: str = token

    def download_model(self) -> None:
        cache_path = Path(BASE_CACHE_PATH, self.__model_name)
        pipe = diffusers.StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=self.__model_url,
            token=self.__token,
            cache_dir=cache_path,
        )
        pipe.save_pretrained(cache_path, safe_serialization=True)
        self.__download_upscaler()

    def __download_upscaler(self) -> None:
        upscaler = diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler",
        )
        upscaler.save_pretrained(BASE_CACHE_PATH_UPSCALER, safe_serialization=True)


class CommonSetup:
    def __init__(self, config: dict, token: str) -> None:
        self.__token: str = token
        self.__config: dict = config

    def download_setup_files(self) -> None:
        if self.__config.get("vae") is not None:
            self.__download_vae(
                name=self.__config["model"]["name"],
                model_url=self.__config["vae"]["url"],
                token=self.__token,
            )

        if self.__config.get("controlnets") is not None:
            for controlnet in self.__config["controlnets"]:
                self.__download_controlnet(
                    name=controlnet["name"],
                    repo_id=controlnet["repo_id"],
                    token=self.__token,
                )

        if self.__config.get("loras") is not None:
            for lora in self.__config["loras"]:
                self.__download_other_file(
                    url=lora["url"],
                    file_name=lora["name"],
                    file_path=BASE_CACHE_PATH_LORA,
                )

        if self.__config.get("textual_inversions") is not None:
            for textual_inversion in self.__config["textual_inversions"]:
                self.__download_other_file(
                    url=textual_inversion["url"],
                    file_name=textual_inversion["name"],
                    file_path=BASE_CACHE_PATH_TEXTUAL_INVERSION,
                )

    def __download_vae(self, name: str, model_url: str, token: str) -> None:
        cache_path = Path(BASE_CACHE_PATH, name)
        vae = diffusers.AutoencoderKL.from_single_file(
            pretrained_model_link_or_path=model_url,
            use_auth_token=token,
            cache_dir=cache_path,
        )
        vae.save_pretrained(cache_path, safe_serialization=True)

    def __download_controlnet(self, name: str, repo_id: str, token: str) -> None:
        cache_path = Path(BASE_CACHE_PATH, name)
        controlnet = diffusers.ControlNetModel.from_pretrained(
            repo_id,
            use_auth_token=token,
            cache_dir=cache_path,
        )
        controlnet.save_pretrained(cache_path, safe_serialization=True)

    def __download_other_file(self, url: str, file_name: str, file_path: str) -> None:
        """
        Download file from the given URL for LoRA or TextualInversion.
        """
        from urllib.request import Request, urlopen

        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        downloaded = urlopen(req).read()
        dir_names = Path(file_path, file_name)
        os.makedirs(os.path.dirname(dir_names), exist_ok=True)
        with open(dir_names, mode="wb") as f:
            f.write(downloaded)


def build_image() -> None:
    """
    Build the Docker image.
    """
    import yaml

    token: str = os.environ["HUGGING_FACE_TOKEN"]
    with open("/config.yml") as file:
        config: dict = yaml.safe_load(file)

    stable_diffusion_setup: StableDiffusionCLISetupInterface
    match config.get("version"):
        case "sd15":
            stable_diffusion_setup = StableDiffusionCLISetupSD15(config, token)
        case "sdxl":
            stable_diffusion_setup = StableDiffusionCLISetupSDXL(config, token)
        case _:
            msg = f"Invalid version: {config.get('version')}. Must be 'sd15' or 'sdxl'."
            raise ValueError(msg)

    stable_diffusion_setup.download_model()
    common_setup = CommonSetup(config, token)
    common_setup.download_setup_files()


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

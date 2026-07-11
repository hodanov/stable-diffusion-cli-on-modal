from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import PIL.Image
from modal import App, Image, Secret, Volume, enter, method

DEFAULT_WAN_I2V_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
MODEL_VOLUME_NAME = "wan-i2v-models"
MODEL_VOLUME_PATH = "/vol/models"

model_volume = Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app = App(
    "wan-i2v-cli",
    volumes={MODEL_VOLUME_PATH: model_volume},
)
base_stub = Image.from_dockerfile(
    path="Dockerfile",
)
app.image = base_stub.dockerfile_commands(
    "COPY config.yml /",
)


class WanI2VSetupInterface(ABC):
    @abstractmethod
    def download_model(self) -> None:
        pass


class WanI2VSetup(WanI2VSetupInterface):
    def __init__(self, config: dict, token: str) -> None:
        if config.get("wan_i2v") is None:
            msg = "wan_i2v is required in config.yml."
            raise ValueError(msg)

        model_config = config["wan_i2v"].get("model")
        if model_config is None:
            msg = "wan_i2v.model is required in config.yml."
            raise ValueError(msg)

        self.__model_name: str = model_config["name"]
        self.__repo_id: str = model_config.get("repo_id") or DEFAULT_WAN_I2V_REPO_ID
        self.__safetensors_url: str | None = model_config.get("safetensors_url")
        self.__token: str = token

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        cache_path = Path(MODEL_VOLUME_PATH) / self.__model_name
        if self.__safetensors_url:
            # Keep configs/tokenizers/vae from repo and skip only transformer weights.
            snapshot_download(
                repo_id=self.__repo_id,
                token=self.__token if self.__token != "" else None,
                local_dir=str(cache_path),
                ignore_patterns=[
                    "assets/*",
                    "examples/*",
                    "*.md",
                    "transformer/*.safetensors",
                    "transformer/*.bin",
                    "transformer/*.msgpack",
                ],
                max_workers=2,
            )
            snapshot_download(
                repo_id=self.__repo_id,
                token=self.__token if self.__token != "" else None,
                local_dir=str(cache_path),
                allow_patterns=[
                    "transformer_2/*",
                ],
                max_workers=2,
            )
            self.__download_file(
                self.__safetensors_url,
                cache_path / "transformer",
            )
            model_volume.commit()
            return

        snapshot_download(
            repo_id=self.__repo_id,
            token=self.__token if self.__token != "" else None,
            local_dir=str(cache_path),
            ignore_patterns=[
                "assets/*",
                "examples/*",
                "*.md",
            ],
            max_workers=2,
        )
        model_volume.commit()

    def __normalize_hf_url(self, url: str) -> str:
        if "huggingface.co" in url and "/blob/" in url:
            return url.replace("/blob/", "/resolve/")
        return url

    def __download_file(self, url: str, cache_path: Path) -> None:
        from urllib.request import Request, urlopen

        normalized_url = self.__normalize_hf_url(url)
        filename = self.__filename_from_url(normalized_url)
        req = Request(normalized_url, headers={"User-Agent": "Mozilla/5.0"})
        downloaded = urlopen(req).read()
        cache_path.mkdir(parents=True, exist_ok=True)
        with Path(cache_path / filename).open("wb") as f:
            f.write(downloaded)

    def __filename_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        return Path(parsed.path).name


@app.function(
    timeout=3600,
    secrets=[Secret.from_dotenv(__file__)],
)
def prepare_wan_i2v() -> None:
    import yaml

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    token: str = os.environ.get("HUGGING_FACE_TOKEN", "")
    with open("/config.yml") as file:
        config: dict = yaml.safe_load(file)

    model_volume.reload()
    wan_setup: WanI2VSetupInterface = WanI2VSetup(config, token)
    wan_setup.download_model()


@app.cls(
    gpu="A100-80GB",
    timeout=1800,
    secrets=[Secret.from_dotenv(__file__)],
)
class WanTI2V:
    """
    A class that wraps the Wan text-image-to-video pipeline.
    """

    @enter()
    def setup(self) -> None:
        import torch
        import yaml
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel

        with Path("/config.yml").open() as file:
            config = yaml.safe_load(file)
        model_config = config["wan_i2v"]["model"]
        safetensors_url = model_config.get("safetensors_url")
        model_volume.reload()
        self.__cache_path = Path(MODEL_VOLUME_PATH) / model_config["name"]
        if not Path.exists(self.__cache_path):
            msg = f"The directory '{self.__cache_path}' does not exist."
            raise ValueError(msg)

        vae = AutoencoderKLWan.from_pretrained(
            self.__cache_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe_kwargs: dict = {
            "vae": vae,
            "torch_dtype": torch.bfloat16,
            "local_files_only": True,
        }
        if safetensors_url:
            transformer_path = self.__cache_path / "transformer" / self.__filename_from_url(
                self.__normalize_hf_url(safetensors_url),
            )
            if not transformer_path.exists():
                msg = f"The file '{transformer_path}' does not exist."
                raise ValueError(msg)
            transformer = WanTransformer3DModel.from_single_file(
                transformer_path,
                torch_dtype=torch.bfloat16,
            )
            pipe_kwargs["transformer"] = transformer

        self.__pipe = WanImageToVideoPipeline.from_pretrained(
            self.__cache_path,
            **pipe_kwargs,
        )
        if hasattr(self.__pipe.transformer.config, "image_dim"):
            self.__pipe.transformer.config.image_dim = None
        if hasattr(self.__pipe, "enable_vae_slicing"):
            self.__pipe.enable_vae_slicing()
        if hasattr(self.__pipe, "enable_vae_tiling"):
            self.__pipe.enable_vae_tiling()
        self.__pipe.to("cuda")

    def __normalize_hf_url(self, url: str) -> str:
        if "huggingface.co" in url and "/blob/" in url:
            return url.replace("/blob/", "/resolve/")
        return url

    def __filename_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        return Path(parsed.path).name

    def __target_size_for_image(
        self,
        image: PIL.Image.Image,
        height: int,
        width: int,
        use_image_aspect: bool,
    ) -> tuple[int, int]:
        if not use_image_aspect:
            return height, width

        max_area = min(1280 * 704, height * width)
        aspect_ratio = image.height / image.width
        height = round((max_area * aspect_ratio) ** 0.5)
        width = round((max_area / aspect_ratio) ** 0.5)
        mod_value = self.__pipe.vae_scale_factor_spatial * self.__pipe.transformer.config.patch_size[1]
        height = max(mod_value, round(height / mod_value) * mod_value)
        width = max(mod_value, round(width / mod_value) * mod_value)

        return int(height), int(width)

    @method()
    def run_inference(
        self,
        *,
        prompt: str,
        n_prompt: str,
        image_bytes: bytes | None,
        height: int = 704,
        width: int = 1280,
        steps: int = 50,
        seed: int = 1,
        num_frames: int = 121,
        fps: int = 24,
        guidance_scale: float = 5.0,
        use_image_aspect: bool = True,
    ) -> bytes:
        """
        Runs the Wan text-image-to-video pipeline and returns an mp4 binary.
        """
        import tempfile

        import torch
        from diffusers.utils import export_to_video

        if image_bytes is None:
            msg = "image_bytes is required for TI2V."
            raise ValueError(msg)

        with io.BytesIO(image_bytes) as buf:
            image = PIL.Image.open(buf).convert("RGB")
        height, width = self.__target_size_for_image(
            image=image,
            height=height,
            width=width,
            use_image_aspect=use_image_aspect,
        )
        if image.size != (width, height):
            image = image.resize((width, height), resample=PIL.Image.LANCZOS)

        generator = torch.Generator("cuda").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": n_prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "generator": generator,
        }

        output = self.__pipe(**kwargs)
        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            export_to_video(frames, tmp.name, fps=fps)
            tmp.seek(0)
            return tmp.read()

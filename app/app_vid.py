from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import PIL.Image
from modal import App, Image, Secret, Volume, enter, method

if TYPE_CHECKING:
    from diffusers import WanTransformer3DModel

DEFAULT_WAN_I2V_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
MODEL_VOLUME_NAME = "wan-i2v-models"
MODEL_VOLUME_PATH = "/vol/models"
# Wan recommends flow_shift 3.0 for 480P-area outputs and 5.0 for 720P-area
# outputs; switch at the midpoint of the two areas.
FLOW_SHIFT_720P_AREA_THRESHOLD = (480 * 832 + 720 * 1280) // 2
# Official Wan negative prompt; generating with an empty negative prompt
# noticeably degrades quality (overexposure, mushy faces, extra limbs).
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: RUF001

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


def dequantize_comfy_scaled_fp8(state_dict: dict) -> dict:
    """
    Dequantizes ComfyUI-style scaled-fp8 tensors in place.

    ComfyUI checkpoints quantized to float8 store a per-tensor `weight_scale`
    (and a `comfy_quant` descriptor) next to each fp8 weight. diffusers'
    from_single_file silently drops those keys and would load the raw fp8
    values with wrong magnitudes, so multiply the scales back in first.
    """
    import torch

    for key in [k for k in state_dict if k.endswith(".weight_scale")]:
        weight_key = key.removesuffix("_scale")
        scale = state_dict.pop(key)
        weight = state_dict[weight_key]
        state_dict[weight_key] = (weight.to(torch.float32) * scale).to(torch.bfloat16)
    for key in [k for k in state_dict if k.endswith(".comfy_quant")]:
        del state_dict[key]
    return state_dict


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
        self.__safetensors_url_low: str | None = model_config.get("safetensors_url_low")
        if self.__safetensors_url_low and not self.__safetensors_url:
            msg = "wan_i2v.model.safetensors_url is required when safetensors_url_low is set."
            raise ValueError(msg)
        self.__token: str = token

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        cache_path = Path(MODEL_VOLUME_PATH) / self.__model_name
        if self.__safetensors_url:
            # Keep configs/tokenizers/vae from repo and skip only transformer weights.
            ignore_patterns = [
                "assets/*",
                "examples/*",
                "*.md",
                "transformer/*.safetensors",
                "transformer/*.bin",
                "transformer/*.msgpack",
            ]
            if self.__safetensors_url_low:
                ignore_patterns += [
                    "transformer_2/*.safetensors",
                    "transformer_2/*.bin",
                    "transformer_2/*.msgpack",
                ]
            snapshot_download(
                repo_id=self.__repo_id,
                token=self.__token if self.__token != "" else None,
                local_dir=str(cache_path),
                ignore_patterns=ignore_patterns,
                max_workers=2,
            )
            if not self.__safetensors_url_low:
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
            if self.__safetensors_url_low:
                self.__download_file(
                    self.__safetensors_url_low,
                    cache_path / "transformer_2",
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
    gpu="B200",
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
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        with Path("/config.yml").open() as file:
            config = yaml.safe_load(file)
        model_config = config["wan_i2v"]["model"]
        safetensors_url = model_config.get("safetensors_url")
        safetensors_url_low = model_config.get("safetensors_url_low")
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
            pipe_kwargs["transformer"] = self.__load_transformer(safetensors_url, "transformer")
        if safetensors_url_low:
            pipe_kwargs["transformer_2"] = self.__load_transformer(safetensors_url_low, "transformer_2")

        self.__pipe = WanImageToVideoPipeline.from_pretrained(
            self.__cache_path,
            **pipe_kwargs,
        )
        if hasattr(self.__pipe.transformer.config, "image_dim"):
            self.__pipe.transformer.config.image_dim = None
        # Both 14B experts, the text encoder and the VAE need ~68GB resident.
        # Keep everything on the GPU when it fits (B200/H200); on smaller GPUs
        # (e.g. A100-80GB) that OOMs, so fall back to per-component offload.
        min_resident_vram = 100 * 1024**3
        if torch.cuda.get_device_properties(0).total_memory >= min_resident_vram:
            self.__pipe.to("cuda")
        else:
            # WanImageToVideoPipeline has no enable_vae_slicing/enable_vae_tiling;
            # call them on the Wan VAE directly. Tiled decode blends overlapping
            # 256px tiles and softens detail, so keep it for low-VRAM GPUs only.
            self.__pipe.vae.enable_slicing()
            self.__pipe.vae.enable_tiling()
            self.__pipe.enable_model_cpu_offload()

    def __load_transformer(self, url: str, subfolder: str) -> WanTransformer3DModel:
        import torch
        from diffusers import WanTransformer3DModel
        from safetensors.torch import load_file

        transformer_path = self.__cache_path / subfolder / self.__filename_from_url(
            self.__normalize_hf_url(url),
        )
        if not transformer_path.exists():
            msg = f"The file '{transformer_path}' does not exist."
            raise ValueError(msg)
        state_dict = dequantize_comfy_scaled_fp8(load_file(transformer_path))
        # ComfyUI-style Wan 2.2 checkpoints get misdetected as Wan 2.1 I2V by
        # from_single_file's config inference, leaving image cross-attention
        # params on the meta device. Pin the config to the downloaded repo.
        transformer = WanTransformer3DModel.from_single_file(
            state_dict,
            config=str(self.__cache_path),
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        )
        # from_single_file honors _keep_in_fp32_modules only for float16, so
        # with bfloat16 it casts the whole model down. Restore the float32
        # modules that the from_pretrained path would keep.
        fp32_modules = transformer._keep_in_fp32_modules  # noqa: SLF001
        for name, param in transformer.named_parameters():
            if any(m in name.split(".") for m in fp32_modules):
                param.data = param.data.float()
        return transformer

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
        guidance_scale_2: float | None = None,
        use_image_aspect: bool = True,
    ) -> bytes:
        """
        Runs the Wan text-image-to-video pipeline and returns an mp4 binary.
        """
        import tempfile

        import torch
        from diffusers import UniPCMultistepScheduler
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

        # The repo scheduler config ships the 480P flow_shift (3.0); rebuild the
        # scheduler with the value matching the output area.
        flow_shift = 5.0 if height * width >= FLOW_SHIFT_720P_AREA_THRESHOLD else 3.0
        self.__pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.__pipe.scheduler.config,
            flow_shift=flow_shift,
        )

        generator = torch.Generator("cuda").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": n_prompt or DEFAULT_NEGATIVE_PROMPT,
            "image": image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            # None falls back to guidance_scale for the low-noise expert.
            "guidance_scale_2": guidance_scale_2,
            "num_inference_steps": steps,
            "generator": generator,
        }

        output = self.__pipe(**kwargs)
        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            # The default quality (5/10) encodes at a bitrate low enough to
            # smear fine detail; keep the encode near-lossless.
            export_to_video(frames, tmp.name, fps=fps, quality=10)
            tmp.seek(0)
            return tmp.read()

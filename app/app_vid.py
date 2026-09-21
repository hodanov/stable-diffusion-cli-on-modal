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
    import numpy as np
    from diffusers import WanTransformer3DModel

DEFAULT_WAN_I2V_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
MODEL_VOLUME_NAME = "wan-i2v-models"
MODEL_VOLUME_PATH = "/vol/models"
# Wan recommends flow_shift 3.0 for 480P-area outputs and 5.0 for 720P-area
# outputs; switch at the midpoint of the two areas.
FLOW_SHIFT_720P_AREA_THRESHOLD = (480 * 832 + 720 * 1280) // 2
# Official Wan negative prompt; generating with an empty negative prompt
# noticeably degrades quality (overexposure, mushy faces, extra limbs).
DEFAULT_NEGATIVE_PROMPT = "Garish colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, image, still, overall grayish cast, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless image, cluttered background, three legs, crowded background, walking backwards."
# Post-processing weights, loaded via spandrel: the anime-video Real-ESRGAN
# Compact model for upscaling and GFPGAN v1.4 for face restoration.
REALESRGAN_WEIGHT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
GFPGAN_WEIGHT_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)
POSTPROCESS_DIR_NAME = "postprocess"
# The Real-ESRGAN model outputs 4x; resize its output down to this factor to
# balance detail recovery against file size and encode time.
UPSCALE_FACTOR = 2

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


def ensure_http_url(url: str) -> None:
    """Reject non-HTTP(S) URLs so urlopen never reads file: or custom schemes."""
    if urlparse(url).scheme not in ("http", "https"):
        msg = f"Only http(s) URLs are supported: {url}"
        raise ValueError(msg)


def normalize_hf_url(url: str) -> str:
    """Rewrite a Hugging Face blob URL to its raw-file (resolve) form."""
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def filename_from_url(url: str) -> str:
    """Return the file name a URL points at, ignoring query and fragment."""
    parsed = urlparse(url)
    return Path(parsed.path).name


def resolve_flow_shift(height: int, width: int, override: float | None) -> float:
    """
    Pick the scheduler flow_shift for an output size.

    The repo scheduler config ships the 480P value; Wan recommends 5.0 once the
    output area reaches 720P. A config override always wins.
    """
    if override is not None:
        return override
    return 5.0 if height * width >= FLOW_SHIFT_720P_AREA_THRESHOLD else 3.0


def target_size_for_image(
    image_size: tuple[int, int],
    height: int,
    width: int,
    *,
    use_image_aspect: bool,
    mod_value: int,
) -> tuple[int, int]:
    """
    Fit the requested output size to the input image's aspect ratio.

    Both sides are rounded to a multiple of mod_value (the pipeline's VAE
    scale factor times the transformer patch size), which the pipeline
    requires, and the area never exceeds the requested one.
    """
    if not use_image_aspect:
        return height, width

    image_width, image_height = image_size
    max_area = min(1280 * 704, height * width)
    aspect_ratio = image_height / image_width
    height = round((max_area * aspect_ratio) ** 0.5)
    width = round((max_area / aspect_ratio) ** 0.5)
    height = max(mod_value, round(height / mod_value) * mod_value)
    width = max(mod_value, round(width / mod_value) * mod_value)

    return int(height), int(width)


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

    def __download_file(self, url: str, cache_path: Path) -> None:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        normalized_url = normalize_hf_url(url)
        filename = filename_from_url(normalized_url)
        headers = {"User-Agent": "Mozilla/5.0"}
        if self.__token:
            # Gated/private Hugging Face repos 401 without this; snapshot_download
            # already sends it via its own token= param, but direct file URLs
            # need it attached manually.
            headers["Authorization"] = f"Bearer {self.__token}"
        ensure_http_url(normalized_url)
        # The scheme is restricted to http(s) above.
        req = Request(normalized_url, headers=headers)  # noqa: S310
        try:
            downloaded = urlopen(req).read()  # noqa: S310
        except HTTPError as e:
            # The raw HTTPError holds an open response stream that Modal can't
            # pickle across the container boundary, which masks the real
            # status/reason behind a SerializationError. Re-raise as a plain
            # exception so the actual cause reaches the caller.
            msg = f"Failed to download {normalized_url}: HTTP {e.code} {e.reason}"
            raise ValueError(msg) from e
        cache_path.mkdir(parents=True, exist_ok=True)
        with Path(cache_path / filename).open("wb") as f:
            f.write(downloaded)


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
    with Path("/config.yml").open() as file:
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
        lora_url = model_config.get("lora_url")
        lora_url_low = model_config.get("lora_url_low")
        # Model-level override for the scheduler flow_shift; distill LoRAs are
        # trained at a fixed shift (lightx2v: 5.0), which should win over the
        # area-based default.
        self.__flow_shift_override = model_config.get("flow_shift")
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
            pipe_kwargs["transformer"] = self.__load_transformer(
                safetensors_url,
                "transformer",
            )
        if safetensors_url_low:
            pipe_kwargs["transformer_2"] = self.__load_transformer(
                safetensors_url_low,
                "transformer_2",
            )

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

        # Distill/accelerator LoRAs (e.g. lightx2v 4-step): load_lora_weights
        # targets the high-noise expert by default; load_into_transformer_2
        # targets the low-noise expert. Strengths are set per expert.
        if lora_url:
            self.__pipe.load_lora_weights(
                str(self.__ensure_lora_file(lora_url)),
                adapter_name="accel_high",
            )
            lora_scale = float(model_config.get("lora_scale", 1.0))
            self.__pipe.transformer.set_adapters(["accel_high"], weights=[lora_scale])
        if lora_url_low:
            self.__pipe.load_lora_weights(
                str(self.__ensure_lora_file(lora_url_low)),
                adapter_name="accel_low",
                load_into_transformer_2=True,
            )
            lora_scale_low = float(model_config.get("lora_scale_low", 1.0))
            self.__pipe.transformer_2.set_adapters(
                ["accel_low"],
                weights=[lora_scale_low],
            )

        # Post-processing models are loaded lazily on the first request that
        # asks for them; see __ensure_postprocessors.
        self.__upscaler = None
        self.__face_restorer = None
        self.__face_helper = None

    def __load_transformer(self, url: str, subfolder: str) -> WanTransformer3DModel:
        import torch
        from diffusers import WanTransformer3DModel
        from safetensors.torch import load_file

        transformer_path = (
            self.__cache_path
            / subfolder
            / filename_from_url(
                normalize_hf_url(url),
            )
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

    def __ensure_lora_file(self, url: str) -> Path:
        normalized_url = normalize_hf_url(url)
        lora_path = self.__cache_path / "loras" / filename_from_url(normalized_url)
        if not lora_path.exists():
            self.__download_file(normalized_url, lora_path.parent)
            model_volume.commit()
        return lora_path

    def __download_file(self, url: str, cache_path: Path) -> None:
        from urllib.request import Request, urlopen

        filename = filename_from_url(url)
        ensure_http_url(url)
        # The scheme is restricted to http(s) above.
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
        downloaded = urlopen(req).read()  # noqa: S310
        cache_path.mkdir(parents=True, exist_ok=True)
        with Path(cache_path / filename).open("wb") as f:
            f.write(downloaded)

    def __ensure_postprocessors(self) -> None:
        if self.__face_helper is not None:
            return

        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from spandrel import ModelLoader

        weights_path = Path(MODEL_VOLUME_PATH) / POSTPROCESS_DIR_NAME
        for url in (REALESRGAN_WEIGHT_URL, GFPGAN_WEIGHT_URL):
            if not (weights_path / filename_from_url(url)).exists():
                self.__download_file(url, weights_path)

        loader = ModelLoader()
        realesrgan_path = weights_path / filename_from_url(REALESRGAN_WEIGHT_URL)
        gfpgan_path = weights_path / filename_from_url(GFPGAN_WEIGHT_URL)
        self.__upscaler = loader.load_from_file(str(realesrgan_path)).to("cuda").eval()
        self.__face_restorer = loader.load_from_file(str(gfpgan_path)).to("cuda").eval()
        # FaceRestoreHelper downloads its detection/parsing weights into
        # model_rootpath on first construction, so keep them on the volume too.
        self.__face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            use_parse=True,
            det_model="retinaface_resnet50",
            model_rootpath=str(weights_path),
            device="cuda",
        )
        model_volume.commit()

    def __postprocess_frames(
        self,
        frames: np.ndarray,
        *,
        use_upscaler: bool,
        use_face_restore: bool,
    ) -> list[np.ndarray]:
        import numpy as np
        import torch

        self.__ensure_postprocessors()
        processed = []
        for frame in frames:
            image = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
            if use_upscaler:
                tensor = (
                    torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")
                )
                with torch.no_grad():
                    upscaled = self.__upscaler(tensor)
                target_size = (
                    image.shape[0] * UPSCALE_FACTOR,
                    image.shape[1] * UPSCALE_FACTOR,
                )
                upscaled = torch.nn.functional.interpolate(
                    upscaled,
                    size=target_size,
                    mode="area",
                )
                image = upscaled.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            if use_face_restore:
                image = self.__restore_faces(image)
            processed.append(image)
        return processed

    def __restore_faces(self, image: np.ndarray) -> np.ndarray:
        """
        Restores every detected face in an RGB float32 [0, 1] frame with GFPGAN
        and pastes the results back, returning the frame in the same format.
        """
        import numpy as np
        import torch

        bgr = (image[..., ::-1] * 255.0).round().astype(np.uint8)
        helper = self.__face_helper
        helper.clean_all()
        helper.read_image(bgr)
        num_faces = helper.get_face_landmarks_5(
            only_center_face=False,
            eye_dist_threshold=5,
        )
        if num_faces == 0:
            return image

        helper.align_warp_face()
        for cropped_face in helper.cropped_faces:
            face = cropped_face[..., ::-1].astype(np.float32) / 255.0
            tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to("cuda")
            with torch.no_grad():
                restored = self.__face_restorer(tensor)
            restored_face = (
                restored.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            )
            helper.add_restored_face(
                (restored_face[..., ::-1] * 255.0).round().astype(np.uint8),
            )

        helper.get_inverse_affine(None)
        restored_bgr = helper.paste_faces_to_input_image()
        return restored_bgr[..., ::-1].astype(np.float32) / 255.0

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
        use_upscaler: bool = False,
        use_face_restore: bool = False,
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
        height, width = target_size_for_image(
            image_size=image.size,
            height=height,
            width=width,
            use_image_aspect=use_image_aspect,
            mod_value=(
                self.__pipe.vae_scale_factor_spatial
                * self.__pipe.transformer.config.patch_size[1]
            ),
        )
        if image.size != (width, height):
            image = image.resize((width, height), resample=PIL.Image.LANCZOS)

        # Rebuild the scheduler with the flow_shift matching the output size.
        flow_shift = resolve_flow_shift(height, width, self.__flow_shift_override)
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
        if use_upscaler or use_face_restore:
            frames = self.__postprocess_frames(
                frames,
                use_upscaler=use_upscaler,
                use_face_restore=use_face_restore,
            )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            # The default quality (5/10) encodes at a bitrate low enough to
            # smear fine detail; keep the encode near-lossless.
            export_to_video(frames, tmp.name, fps=fps, quality=10)
            tmp.seek(0)
            return tmp.read()

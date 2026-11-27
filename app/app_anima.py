from __future__ import annotations

import io
import math
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from modal import App, Image, Secret, Volume, enter, method

DEFAULT_ANIMA_REPO_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"
MODEL_VOLUME_NAME = "anima-models"
MODEL_VOLUME_PATH = "/vol/models"
# Keep the single-file download out of the volume; only the converted
# components are persisted. The path is container-local and single-tenant.
HF_CACHE_DIR = "/tmp/hf_cache"  # noqa: S108
LORA_DIR_NAME = "loras"
# The pipeline's own default; Turbo checkpoints want 1.0 (set it in config.yml).
DEFAULT_GUIDANCE_SCALE = 4.0
# Recommended by the Anima model card.
DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"
# The model card recommends outputs between 512^2 and 1536^2 pixels.
MIN_AREA = 512 * 512
MAX_AREA = 1536 * 1536
# VAE scale factor (8) times the transformer patch size (2).
SIZE_MULTIPLE = 16
UPSCALE_FACTOR = 2.0
UPSCALE_STRENGTH = 0.3
# Single-file checkpoints keep the LLM adapter (diffusers'
# AnimaTextConditioner) next to the Cosmos DiT. The original Cosmos layout
# prefixes every key with `net.`; ComfyUI repacks (e.g. anima-turbo-v1.1) use
# the full-model prefix instead. Keys are normalized to `net.`, the prefix the
# diffusers Cosmos converter strips.
COSMOS_PREFIX = "net."
CHECKPOINT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", COSMOS_PREFIX)
LLM_ADAPTER_PREFIX = "llm_adapter."
# Written last by download_model(), so a run that failed half-way (e.g. in
# the conversion) is retried instead of being skipped as already downloaded.
DOWNLOAD_COMPLETE_MARKER = ".download_complete"

model_volume = Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
app_image = Image.from_dockerfile(
    path="Dockerfile",
).dockerfile_commands(
    "COPY config.yml /",
)
app = App(
    "anima-cli",
    image=app_image,
    volumes={MODEL_VOLUME_PATH: model_volume},
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


def split_anima_checkpoint(state_dict: dict) -> tuple[dict, dict]:
    """
    Split a single-file Anima checkpoint into transformer and text conditioner weights.

    The transformer half is keyed with the `net.` prefix the Cosmos converter
    expects; the text conditioner half has every prefix removed, which is
    already the AnimaTextConditioner layout.
    """
    transformer_state_dict = {}
    text_conditioner_state_dict = {}
    for key, value in state_dict.items():
        bare_key = key
        for prefix in CHECKPOINT_PREFIXES:
            if key.startswith(prefix):
                bare_key = key.removeprefix(prefix)
                break
        if bare_key.startswith(LLM_ADAPTER_PREFIX):
            text_conditioner_state_dict[bare_key.removeprefix(LLM_ADAPTER_PREFIX)] = value
        else:
            transformer_state_dict[COSMOS_PREFIX + bare_key] = value
    return transformer_state_dict, text_conditioner_state_dict


def round_to_multiple(value: float, multiple: int = SIZE_MULTIPLE) -> int:
    """Round half up (not Python's half-to-even) to a positive multiple."""
    return max(multiple, math.floor(value / multiple + 0.5) * multiple)


def resolve_size(height: int, width: int) -> tuple[int, int]:
    """Round the size to what the pipeline accepts and keep it in the model's range."""
    height = round_to_multiple(height)
    width = round_to_multiple(width)
    area = height * width
    if not MIN_AREA <= area <= MAX_AREA:
        msg = f"{width}x{height} is outside the supported area (512x512 to 1536x1536 pixels)."
        raise ValueError(msg)
    return height, width


def upscale_size(height: int, width: int) -> tuple[int, int]:
    """
    Return the size of the img2img refine pass.

    Doubles the image like the SDXL refiner does, but caps the area at the
    model's upper limit, since Anima degrades above 1536^2 pixels.
    """
    factor = min(UPSCALE_FACTOR, math.sqrt(MAX_AREA / (height * width)))
    new_height = round_to_multiple(height * factor)
    new_width = round_to_multiple(width * factor)
    # Rounding up can overshoot the cap by a row of patches; step back down.
    while new_height * new_width > MAX_AREA:
        if new_height >= new_width:
            new_height -= SIZE_MULTIPLE
        else:
            new_width -= SIZE_MULTIPLE
    return new_height, new_width


def download_file(url: str, dest_dir: Path, token: str) -> Path:
    """Stream a file to dest_dir, sending the HF token for gated repos."""
    import shutil
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    normalized_url = normalize_hf_url(url)
    ensure_http_url(normalized_url)
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    dest = dest_dir / filename_from_url(normalized_url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # The scheme is restricted to http(s) above.
    req = Request(normalized_url, headers=headers)  # noqa: S310
    try:
        with urlopen(req) as response, dest.open("wb") as f:  # noqa: S310
            shutil.copyfileobj(response, f, length=16 * 1024 * 1024)
    except HTTPError as e:
        # Modal can't pickle the open response held by HTTPError; re-raise a
        # plain exception so the actual status reaches the caller.
        msg = f"Failed to download {normalized_url}: HTTP {e.code} {e.reason}"
        raise ValueError(msg) from e
    return dest


def convert_single_file(checkpoint_path: Path, model_path: Path) -> None:
    """
    Replace the transformer and text conditioner in model_path with a single-file checkpoint.

    The configs come from the Diffusers repo already in model_path (all Anima
    variants share the architecture), so nothing is fetched from the gated
    nvidia Cosmos repo that from_single_file would otherwise ask for.
    """
    import torch
    from accelerate import init_empty_weights
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
    )
    from safetensors.torch import load_file

    transformer_state_dict, text_conditioner_state_dict = split_anima_checkpoint(
        load_file(checkpoint_path, device="cpu"),
    )
    transformer_state_dict = convert_cosmos_transformer_checkpoint_to_diffusers(
        transformer_state_dict,
    )

    for model_class, subfolder, state_dict in (
        (CosmosTransformer3DModel, "transformer", transformer_state_dict),
        (AnimaTextConditioner, "text_conditioner", text_conditioner_state_dict),
    ):
        component_path = model_path / subfolder
        with init_empty_weights():
            model = model_class.from_config(model_class.load_config(component_path))
        # strict=True lists every missing/unexpected key if the checkpoint
        # does not match the architecture.
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(torch.bfloat16).save_pretrained(component_path, safe_serialization=True)


class AnimaSetup:
    def __init__(self, config: dict, token: str) -> None:
        anima_config = config.get("anima")
        if anima_config is None:
            msg = "anima is required in config.yml."
            raise ValueError(msg)

        model_config = anima_config.get("model")
        if model_config is None or not model_config.get("name"):
            msg = "anima.model.name is required in config.yml."
            raise ValueError(msg)

        self.__model_name: str = model_config["name"]
        self.__repo_id: str = model_config.get("repo_id") or DEFAULT_ANIMA_REPO_ID
        self.__transformer_url: str | None = model_config.get("transformer_url")
        self.__loras: list[dict] = anima_config.get("loras") or []
        for lora in self.__loras:
            if not lora.get("url"):
                msg = "Each entry in anima.loras needs a url."
                raise ValueError(msg)
        self.__token: str = token

    @property
    def model_path(self) -> Path:
        return Path(MODEL_VOLUME_PATH) / self.__model_name

    @property
    def is_downloaded(self) -> bool:
        return (self.model_path / DOWNLOAD_COMPLETE_MARKER).exists()

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        ignore_patterns = ["*.md"]
        if self.__transformer_url:
            # Keep the configs; the weights come from the single file below.
            ignore_patterns += [
                "transformer/*.safetensors",
                "text_conditioner/*.safetensors",
            ]
        snapshot_download(
            repo_id=self.__repo_id,
            token=self.__token or None,
            local_dir=str(self.model_path),
            ignore_patterns=ignore_patterns,
        )
        if self.__transformer_url:
            checkpoint_path = download_file(
                self.__transformer_url,
                Path(HF_CACHE_DIR),
                self.__token,
            )
            convert_single_file(checkpoint_path, self.model_path)
            checkpoint_path.unlink()
        self.model_path.mkdir(parents=True, exist_ok=True)
        (self.model_path / DOWNLOAD_COMPLETE_MARKER).touch()

    def download_loras(self) -> None:
        lora_dir = self.model_path / LORA_DIR_NAME
        for lora in self.__loras:
            if (lora_dir / filename_from_url(normalize_hf_url(lora["url"]))).exists():
                continue
            download_file(lora["url"], lora_dir, self.__token)


@app.function(
    timeout=3600,
    secrets=[Secret.from_dotenv(__file__)],
)
def prepare_anima() -> None:
    """
    Download the Anima model and LoRAs into the model volume.

    The model download is skipped once it has completed. A run that failed
    half-way is redone on the next call. To force a re-download, remove it with
    `modal volume rm -r anima-models /<model name>`.
    """
    import yaml

    token: str = os.environ.get("HUGGING_FACE_TOKEN", "")
    with Path("/config.yml").open() as file:
        config: dict = yaml.safe_load(file)

    model_volume.reload()
    setup = AnimaSetup(config, token)
    if setup.is_downloaded:
        print(f"The model already exists at '{setup.model_path}'. Skipping the download.")  # noqa: T201
    else:
        setup.download_model()
    setup.download_loras()
    model_volume.commit()


@app.cls(
    # 2B in bf16 fits any 24GB+ GPU. A single type can sit unscheduled for
    # minutes when Modal is out of capacity, so fall back in this order.
    gpu=["L4", "A10G", "L40S"],
    # Allow for a cold host after preemption, like the SDXL app.
    timeout=600,
    startup_timeout=900,
    secrets=[Secret.from_dotenv(__file__)],
)
class AnimaTxt2Img:
    """
    A class that wraps the Anima modular pipeline.
    """

    @enter()
    def setup(self) -> None:
        import torch
        import yaml
        from diffusers import ModularPipeline

        start_time = time.perf_counter()
        with Path("/config.yml").open() as file:
            config = yaml.safe_load(file)
        anima_config = config["anima"]
        model_config = anima_config["model"]
        self.__default_guidance_scale = float(
            model_config.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
        )
        model_volume.reload()
        model_path = Path(MODEL_VOLUME_PATH) / model_config["name"]
        if not model_path.exists():
            msg = f"The directory '{model_path}' does not exist. Run `make prep_anima` first to download the model into the volume."
            raise ValueError(msg)

        self.__pipe = ModularPipeline.from_pretrained(str(model_path))
        # modular_model_index.json points every component at the Hub repo;
        # load from the volume instead, which also picks up the converted
        # transformer and text conditioner of a single-file variant.
        self.__pipe.load_components(
            pretrained_model_name_or_path=str(model_path),
            dtype=torch.bfloat16,
        )
        self.__pipe.to("cuda")

        loras = anima_config.get("loras") or []
        if loras:
            adapter_names = []
            for index, lora in enumerate(loras):
                adapter_name = f"lora_{index}"
                lora_path = model_path / LORA_DIR_NAME / filename_from_url(normalize_hf_url(lora["url"]))
                self.__pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                adapter_names.append(adapter_name)
            self.__pipe.set_adapters(
                adapter_names,
                adapter_weights=[float(lora.get("scale", 1.0)) for lora in loras],
            )
        print(f"Loaded the pipeline in {time.perf_counter() - start_time:.1f}s.")  # noqa: T201

    @method()
    def run_inference(
        self,
        *,
        prompt: str,
        n_prompt: str,
        height: int = 1024,
        width: int = 1024,
        steps: int = 10,
        seed: int = 1,
        guidance_scale: float | None = None,
        output_format: str = "png",
        use_upscaler: bool = False,
    ) -> list[bytes]:
        """
        Runs the Anima pipeline on the given prompt and outputs images.
        """
        import PIL.Image
        import pillow_avif  # noqa: F401
        import torch
        from diffusers.guiders import ClassifierFreeGuidance

        height, width = resolve_size(height, width)
        # CFG lives in the guider component. At 1.0 the guider skips the
        # unconditional pass, which is what Turbo checkpoints expect.
        self.__pipe.update_components(
            guider=ClassifierFreeGuidance(
                guidance_scale=guidance_scale if guidance_scale is not None else self.__default_guidance_scale,
            ),
        )
        common_kwargs = {
            "prompt": prompt,
            "negative_prompt": n_prompt or DEFAULT_NEGATIVE_PROMPT,
            "num_inference_steps": steps,
            "generator": torch.Generator("cuda").manual_seed(seed),
        }
        generated_image = self.__pipe(
            height=height,
            width=width,
            output="images",
            **common_kwargs,
        )[0]
        generated_images = [generated_image]

        if use_upscaler:
            refine_height, refine_width = upscale_size(height, width)
            base_image = generated_image.convert("RGB").resize(
                (refine_width, refine_height),
                resample=PIL.Image.LANCZOS,
            )
            refined_image = self.__pipe(
                image=base_image,
                strength=UPSCALE_STRENGTH,
                height=refine_height,
                width=refine_width,
                output="images",
                **common_kwargs,
            )[0]
            generated_images.append(refined_image)

        image_output = []
        for image in generated_images:
            with io.BytesIO() as buf:
                image.save(buf, format=output_format)
                image_output.append(buf.getvalue())

        return image_output

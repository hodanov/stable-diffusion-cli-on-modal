from __future__ import annotations

import io
import os

import PIL.Image
from huggingface_hub import login
from modal import App, Image, Secret, enter, method

app = App("wan-i2v-cli")
base_stub = Image.from_dockerfile(
    path="Dockerfile",
)
app.image = base_stub.dockerfile_commands(
    "COPY config.yml /",
)


@app.cls(
    gpu="A100-80GB",
    timeout=1800,
    secrets=[Secret.from_dotenv(__file__)],
)
class WanTI2V:
    """
    A class that wraps the Wan text-image-to-video pipeline.
    """

    model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    @enter()
    def setup(self) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        token = os.environ.get("HUGGING_FACE_TOKEN", "")
        if token != "":
            login(token)

        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            use_auth_token=token if token != "" else None,
        )
        self.__pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            use_auth_token=token if token != "" else None,
        )
        self.__pipe.to("cuda")

    def __target_size_for_image(
        self,
        image: PIL.Image.Image,
        height: int,
        width: int,
        use_image_aspect: bool,
    ) -> tuple[int, int]:
        if not use_image_aspect:
            return height, width

        max_area = 1280 * 704
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

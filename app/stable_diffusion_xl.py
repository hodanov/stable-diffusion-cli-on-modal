from __future__ import annotations

import io
import os

import PIL.Image
from modal import Secret, enter, method
from setup import BASE_CACHE_PATH, app


@app.cls(
    gpu="A10G",
    secrets=[Secret.from_dotenv(__file__)],
)
class SDXLTxt2Img:
    """
    A class that wraps the Stable Diffusion pipeline and scheduler.
    """

    @enter()
    def _setup(self):
        import diffusers
        import torch
        import yaml

        config = {}
        with open("/config.yml", "r") as file:
            config = yaml.safe_load(file)
        self.cache_path = os.path.join(BASE_CACHE_PATH, config["model"]["name"])
        if os.path.exists(self.cache_path):
            print(f"The directory '{self.cache_path}' exists.")
        else:
            print(f"The directory '{self.cache_path}' does not exist.")

        self.pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
            self.cache_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.refiner = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.cache_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

    def _count_token(self, p: str, n: str) -> int:
        """
        Count the number of tokens in the prompt and negative prompt.
        """
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            self.cache_path,
            subfolder="tokenizer",
        )
        token_size_p = len(tokenizer.tokenize(p))
        token_size_n = len(tokenizer.tokenize(n))
        token_size = token_size_p
        if token_size_p <= token_size_n:
            token_size = token_size_n

        max_embeddings_multiples = 1
        max_length = tokenizer.model_max_length - 2
        if token_size > max_length:
            max_embeddings_multiples = token_size // max_length + 1

        print(f"token_size: {token_size}, max_embeddings_multiples: {max_embeddings_multiples}")

        return max_embeddings_multiples

    @method()
    def run_inference(
        self,
        prompt: str,
        n_prompt: str,
        height: int = 1024,
        width: int = 1024,
        steps: int = 30,
        seed: int = 1,
        use_upscaler: bool = False,
        output_format: str = "png",
    ) -> list[bytes]:
        """
        Runs the Stable Diffusion pipeline on the given prompt and outputs images.
        """
        import pillow_avif  # noqa
        import torch

        generator = torch.Generator("cuda").manual_seed(seed)
        self.pipe.to("cuda")
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        generated_image = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            guidance_scale=7,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=steps,
        ).images[0]

        generated_images = [generated_image]

        if use_upscaler:
            self.refiner.to("cuda")
            self.refiner.enable_vae_tiling()
            self.refiner.enable_xformers_memory_efficient_attention()
            base_image = self._double_image_size(generated_image)
            image = self.refiner(
                prompt=prompt,
                negative_prompt=n_prompt,
                num_inference_steps=50,
                strength=0.3,
                guidance_scale=7.5,
                generator=generator,
                image=base_image,
            ).images[0]
            generated_images.append(image)

        image_output = []
        for image in generated_images:
            with io.BytesIO() as buf:
                image.save(buf, format=output_format)
                image_output.append(buf.getvalue())

        return image_output

    def _double_image_size(self, image: PIL.Image.Image) -> PIL.Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        return image.resize((width * 2, height * 2), resample=PIL.Image.LANCZOS)

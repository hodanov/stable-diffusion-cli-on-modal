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

        self.pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
            self.cache_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.refiner_cache_path = self.cache_path + "-refiner"
        self.refiner = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.refiner_cache_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        # controlnets = config.get("controlnets")
        # if controlnets is not None:
        #     for controlnet in controlnets:
        #         path = os.path.join(BASE_CACHE_PATH_CONTROLNET, controlnet["name"])
        #         controlnet = diffusers.ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)
        #         self.controlnet_pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
        #             self.cache_path,
        #             controlnet=controlnet,
        #             custom_pipeline="lpw_stable_diffusion",
        #             scheduler=self.pipe.scheduler,
        #             vae=self.pipe.vae,
        #             torch_dtype=torch.float16,
        #             use_safetensors=True,
        #         )

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
        generated_images = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            height=height,
            width=width,
            generator=generator,
        ).images
        base_images = generated_images

        for image in base_images:
            image = self._resize_image(image=image, scale_factor=2)
            self.refiner.to("cuda")
            refined_images = self.refiner(
                prompt=prompt,
                negative_prompt=n_prompt,
                num_inference_steps=steps,
                strength=0.1,
                # guidance_scale=7.5,
                generator=generator,
                image=image,
            ).images
        generated_images.extend(refined_images)
        base_images = refined_images

        """
        Fix the generated images by the control_v11f1e_sd15_tile when `fix_by_controlnet_tile` is `True`.
        https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile
        """
        # if fix_by_controlnet_tile:
        #     max_embeddings_multiples = self._count_token(p=prompt, n=n_prompt)
        #     self.controlnet_pipe.to("cuda")
        #     self.controlnet_pipe.enable_vae_tiling()
        #     self.controlnet_pipe.enable_xformers_memory_efficient_attention()
        #     for image in base_images:
        #         image = self._resize_image(image=image, scale_factor=2)
        #         with torch.autocast("cuda"):
        #             fixed_by_controlnet = self.controlnet_pipe(
        #                 prompt=prompt * batch_size,
        #                 negative_prompt=n_prompt * batch_size,
        #                 num_inference_steps=steps,
        #                 strength=0.3,
        #                 guidance_scale=7.5,
        #                 max_embeddings_multiples=max_embeddings_multiples,
        #                 generator=generator,
        #                 image=image,
        #             ).images
        #     generated_images.extend(fixed_by_controlnet)
        #     base_images = fixed_by_controlnet

        # if use_upscaler:
        #     upscaled = self._upscale(
        #         base_images=base_images,
        #         half_precision=False,
        #         tile=700,
        #         upscaler=upscaler,
        #     )
        #     generated_images.extend(upscaled)

        image_output = []
        for image in generated_images:
            with io.BytesIO() as buf:
                image.save(buf, format=output_format)
                image_output.append(buf.getvalue())

        return image_output

    def _resize_image(self, image: PIL.Image.Image, scale_factor: int) -> PIL.Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        img = image.resize((width * scale_factor, height * scale_factor), resample=PIL.Image.LANCZOS)
        return img

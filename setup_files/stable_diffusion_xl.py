from __future__ import annotations

import io
import os

import PIL.Image
from modal import Secret, method
from setup import BASE_CACHE_PATH, stub


@stub.cls(
    gpu="A10G",
    secrets=[Secret.from_dotenv(__file__)],
)
class SDXLTxt2Img:
    """
    A class that wraps the Stable Diffusion pipeline and scheduler.
    """

    def __enter__(self):
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

    @method()
    def run_inference(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        seed: int = 1,
        upscaler: str = "",
        use_face_enhancer: bool = False,
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
            height=height,
            width=width,
            generator=generator,
        ).images
        base_images = generated_images

        for image in base_images:
            self.refiner.to("cuda")
            refined_images = self.refiner(
                prompt=prompt,
                image=image,
            ).images
        generated_images.extend(refined_images)
        base_images = refined_images

        if upscaler != "":
            upscaled = self._upscale(
                base_images=base_images,
                half_precision=False,
                tile=700,
                upscaler=upscaler,
                use_face_enhancer=use_face_enhancer,
            )
            generated_images.extend(upscaled)

        image_output = []
        for image in generated_images:
            with io.BytesIO() as buf:
                image.save(buf, format=output_format)
                image_output.append(buf.getvalue())

        return image_output

    def _upscale(
        self,
        base_images: list[PIL.Image],
        half_precision: bool = False,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        upscaler: str = "",
        use_face_enhancer: bool = False,
    ) -> list[PIL.Image]:
        """
        Upscale the generated images by the upscaler when `upscaler` is selected.
        The upscaler can be selected from the following list:
        - `RealESRGAN_x4plus`
        - `RealESRNet_x4plus`
        - `RealESRGAN_x4plus_anime_6B`
        - `RealESRGAN_x2plus`
        https://github.com/xinntao/Real-ESRGAN
        """
        import numpy
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from gfpgan import GFPGANer
        from realesrgan import RealESRGANer
        from tqdm import tqdm

        model_name = upscaler
        if model_name == "RealESRGAN_x4plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == "RealESRNet_x4plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == "RealESRGAN_x2plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            raise NotImplementedError("Model name not supported")

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=os.path.join(BASE_CACHE_PATH, "esrgan", f"{model_name}.pth"),
            dni_weight=None,
            model=upscale_model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half_precision,
            gpu_id=None,
        )

        if use_face_enhancer:
            face_enhancer = GFPGANer(
                model_path=os.path.join(BASE_CACHE_PATH, "esrgan", "GFPGANv1.3.pth"),
                upscale=netscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )

        upscaled_imgs = []
        for img in base_images:
            img = numpy.array(img)
            if use_face_enhancer:
                _, _, enhance_result = face_enhancer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
            else:
                enhance_result, _ = upsampler.enhance(img)

            upscaled_imgs.append(PIL.Image.fromarray(enhance_result))

        return upscaled_imgs

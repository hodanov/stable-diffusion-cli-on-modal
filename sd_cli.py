from __future__ import annotations

import io
import os
import time

from modal import Image, Mount, Secret, Stub, method

import util

BASE_CACHE_PATH = "/vol/cache"


def download_models():
    """
    Downloads the model from Hugging Face and saves it to the cache path using
    diffusers.StableDiffusionPipeline.from_pretrained().
    """
    import diffusers

    hugging_face_token = os.environ["HUGGING_FACE_TOKEN"]
    model_repo_id = os.environ["MODEL_REPO_ID"]
    cache_path = os.path.join(BASE_CACHE_PATH, os.environ["MODEL_NAME"])

    vae = diffusers.AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        use_auth_token=hugging_face_token,
        cache_dir=cache_path,
    )
    vae.save_pretrained(cache_path, safe_serialization=True)

    scheduler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
        model_repo_id,
        subfolder="scheduler",
        use_auth_token=hugging_face_token,
        cache_dir=cache_path,
    )
    scheduler.save_pretrained(cache_path, safe_serialization=True)

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_repo_id,
        use_auth_token=hugging_face_token,
        cache_dir=cache_path,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


stub_image = Image.from_dockerfile(
    path="./Dockerfile",
    context_mount=Mount.from_local_file("./requirements.txt"),
).run_function(
    download_models,
    secrets=[Secret.from_dotenv(__file__)],
)
stub = Stub("stable-diffusion-cli")
stub.image = stub_image


@stub.cls(gpu="A10G", secrets=[Secret.from_dotenv(__file__)])
class StableDiffusion:
    """
    A class that wraps the Stable Diffusion pipeline and scheduler.
    """

    def __enter__(self):
        import diffusers
        import torch

        cache_path = os.path.join(BASE_CACHE_PATH, os.environ["MODEL_NAME"])
        if os.path.exists(cache_path):
            print(f"The directory '{cache_path}' exists.")
        else:
            print(f"The directory '{cache_path}' does not exist. Download models...")
            download_models()

        torch.backends.cuda.matmul.allow_tf32 = True

        vae = diffusers.AutoencoderKL.from_pretrained(
            cache_path,
            subfolder="vae",
        )

        scheduler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
            cache_path,
            subfolder="scheduler",
        )

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            cache_path,
            scheduler=scheduler,
            vae=vae,
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(self, inputs: dict[str, int | str]) -> list[bytes]:
        """
        Runs the Stable Diffusion pipeline on the given prompt and outputs images.
        """
        import torch

        with torch.inference_mode():
            with torch.autocast("cuda"):
                base_images = self.pipe(
                    [inputs["prompt"]] * int(inputs["batch_size"]),
                    negative_prompt=[inputs["n_prompt"]] * int(inputs["batch_size"]),
                    height=inputs["height"],
                    width=inputs["width"],
                    num_inference_steps=inputs["steps"],
                    guidance_scale=7.5,
                    max_embeddings_multiples=inputs["max_embeddings_multiples"],
                ).images

        if inputs["upscaler"] != "":
            uplcaled_images = self.upscale(
                base_images=base_images,
                model_name="RealESRGAN_x4plus",
                scale_factor=4,
                half_precision=False,
                tile=700,
            )
            base_images.extend(uplcaled_images)

        image_output = []
        for image in base_images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())

        return image_output

    @method()
    def upscale(
        self,
        base_images: list[Image.Image],
        model_name: str = "RealESRGAN_x4plus",
        scale_factor: float = 4,
        half_precision: bool = False,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
    ) -> list[Image.Image]:
        """
        Upscales the given images using the given model.
        """
        import numpy
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from PIL import Image
        from realesrgan import RealESRGANer
        from tqdm import tqdm

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

        torch.cuda.empty_cache()
        upscaled_imgs = []
        with tqdm(total=len(base_images)) as progress_bar:
            for i, img in enumerate(base_images):
                img = numpy.array(img)
                enhance_result = upsampler.enhance(img)[0]
                upscaled_imgs.append(Image.fromarray(enhance_result))
                progress_bar.update(1)
        torch.cuda.empty_cache()

        return upscaled_imgs


@stub.local_entrypoint()
def entrypoint(
    prompt: str,
    n_prompt: str,
    height: int = 512,
    width: int = 512,
    samples: int = 5,
    batch_size: int = 1,
    steps: int = 20,
    upscaler: str = "",
):
    """
    This function is the entrypoint for the Runway CLI.
    The function pass the given prompt to StableDiffusion on Modal,
    gets back a list of images and outputs images to local.
    """

    inputs: dict[str, int | str] = {
        "prompt": prompt,
        "n_prompt": n_prompt,
        "height": height,
        "width": width,
        "samples": samples,
        "batch_size": batch_size,
        "steps": steps,
        "upscaler": upscaler,  # sd_x2_latent_upscaler, sd_x4_upscaler
        # seed=-1
    }

    inputs["max_embeddings_multiples"] = util.count_token(p=prompt, n=n_prompt)
    directory = util.make_directory()

    sd = StableDiffusion()
    for i in range(samples):
        start_time = time.time()
        images = sd.run_inference.call(inputs)
        util.save_images(directory, images, i)
        total_time = time.time() - start_time
        print(f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image).")

    util.save_prompts(inputs)

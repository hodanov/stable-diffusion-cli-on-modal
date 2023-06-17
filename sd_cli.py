from __future__ import annotations

import io
import os
import time
from urllib.request import Request, urlopen

from modal import Image, Mount, Secret, Stub, method

import util

BASE_CACHE_PATH = "/vol/cache"
BASE_CACHE_PATH_LORA = "/vol/cache/lora"


def download_loras():
    """
    Download LoRA.
    """
    lora_names = os.getenv("LORA_NAMES").split(",")
    lora_download_urls = os.getenv("LORA_DOWNLOAD_URLS").split(",")

    for name, url in zip(lora_names, lora_download_urls):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        downloaded = urlopen(req).read()

        dir_names = os.path.join(BASE_CACHE_PATH_LORA, name)
        os.makedirs(os.path.dirname(dir_names), exist_ok=True)
        with open(dir_names, mode="wb") as f:
            f.write(downloaded)


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


def build_image():
    """
    Build the Docker image.
    """
    download_models()

    if os.environ["LORA_NAMES"] != "":
        download_loras()


stub_image = Image.from_dockerfile(
    path="./Dockerfile",
    context_mount=Mount.from_local_file("./requirements.txt"),
).run_function(
    build_image,
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

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            cache_path,
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16,
        )

        self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
            cache_path,
            subfolder="scheduler",
        )

        if os.environ["USE_VAE"] == "true":
            self.pipe.vae = diffusers.AutoencoderKL.from_pretrained(
                cache_path,
                subfolder="vae",
            )

        self.pipe.to("cuda")

        if os.environ["LORA_NAMES"] != "":
            lora_names = os.getenv("LORA_NAMES").split(",")
            for lora_name in lora_names:
                path_to_lora = os.path.join(BASE_CACHE_PATH_LORA, lora_name)
                if os.path.exists(path_to_lora):
                    print(f"The directory '{path_to_lora}' exists.")
                else:
                    print(f"The directory '{path_to_lora}' does not exist. Download loras...")
                    download_loras()
                self.pipe.load_lora_weights(".", weight_name=path_to_lora)

        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def count_token(self, p: str, n: str) -> int:
        """
        Count the number of tokens in the prompt and negative prompt.
        """
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
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
    def run_inference(self, inputs: dict[str, int | str]) -> list[bytes]:
        """
        Runs the Stable Diffusion pipeline on the given prompt and outputs images.
        """
        import torch

        generator = torch.Generator("cuda").manual_seed(inputs["seed"])
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
                    generator=generator,
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
        https://github.com/xinntao/Real-ESRGAN
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
    upscaler: str,
    height: int = 512,
    width: int = 512,
    samples: int = 5,
    batch_size: int = 1,
    steps: int = 20,
    seed: int = -1,
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
        "upscaler": upscaler,
        "seed": seed,
    }

    directory = util.make_directory()

    sd = StableDiffusion()
    inputs["max_embeddings_multiples"] = sd.count_token(p=prompt, n=n_prompt)
    for i in range(samples):
        if seed == -1:
            inputs["seed"] = util.generate_seed()
        start_time = time.time()
        images = sd.run_inference.call(inputs)
        util.save_images(directory, images, int(inputs["seed"]), i)
        total_time = time.time() - start_time
        print(f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image).")

    util.save_prompts(inputs)

from __future__ import annotations

import io
import os
from urllib.request import Request, urlopen

from modal import Image, Mount, Secret, Stub, method
from modal.cls import ClsMixin

BASE_CACHE_PATH = "/vol/cache"
BASE_CACHE_PATH_LORA = "/vol/cache/lora"
BASE_CACHE_PATH_TEXTUAL_INVERSION = "/vol/cache/textual_inversion"


def download_files(urls, file_names, file_path):
    """
    Download files.
    """
    file_names = file_names.split(",")
    urls = urls.split(",")

    for file_name, url in zip(file_names, urls):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        downloaded = urlopen(req).read()

        dir_names = os.path.join(file_path, file_name)
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
        download_files(
            os.getenv("LORA_DOWNLOAD_URLS"),
            os.getenv("LORA_NAMES"),
            BASE_CACHE_PATH_LORA,
        )

    if os.environ["TEXTUAL_INVERSION_NAMES"] != "":
        download_files(
            os.getenv("TEXTUAL_INVERSION_DOWNLOAD_URLS"),
            os.getenv("TEXTUAL_INVERSION_NAMES"),
            BASE_CACHE_PATH_TEXTUAL_INVERSION,
        )


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
class StableDiffusion(ClsMixin):
    """
    A class that wraps the Stable Diffusion pipeline and scheduler.
    """

    def __enter__(self):
        import diffusers
        import torch

        self.cache_path = os.path.join(BASE_CACHE_PATH, os.environ["MODEL_NAME"])
        if os.path.exists(self.cache_path):
            print(f"The directory '{self.cache_path}' exists.")
        else:
            print(f"The directory '{self.cache_path}' does not exist. Download models...")
            download_models()

        torch.backends.cuda.matmul.allow_tf32 = True

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            self.cache_path,
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16,
        )

        # TODO: Add support for other schedulers.
        self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
            # self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            self.cache_path,
            subfolder="scheduler",
        )

        if os.environ["USE_VAE"] == "true":
            self.pipe.vae = diffusers.AutoencoderKL.from_pretrained(
                self.cache_path,
                subfolder="vae",
            )

        self.pipe.to("cuda")

        if os.environ["LORA_NAMES"] != "":
            names = os.environ["LORA_NAMES"].split(",")
            urls = os.environ["LORA_DOWNLOAD_URLS"].split(",")
            for name, url in zip(names, urls):
                path = os.path.join(BASE_CACHE_PATH_LORA, name)
                if os.path.exists(path):
                    print(f"The directory '{path}' exists.")
                else:
                    print(f"The directory '{path}' does not exist. Download it...")
                    download_files(url, name, BASE_CACHE_PATH_LORA)
                self.pipe.load_lora_weights(".", weight_name=path)

        if os.environ["TEXTUAL_INVERSION_NAMES"] != "":
            names = os.environ["TEXTUAL_INVERSION_NAMES"].split(",")
            urls = os.environ["TEXTUAL_INVERSION_DOWNLOAD_URLS"].split(",")
            for name, url in zip(names, urls):
                path = os.path.join(BASE_CACHE_PATH_TEXTUAL_INVERSION, name)
                if os.path.exists(path):
                    print(f"The directory '{path}' exists.")
                else:
                    print(f"The directory '{path}' does not exist. Download it...")
                    download_files(url, name, BASE_CACHE_PATH_TEXTUAL_INVERSION)
                self.pipe.load_textual_inversion(path)

        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def count_token(self, p: str, n: str) -> int:
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
        height: int = 512,
        width: int = 512,
        samples: int = 1,
        batch_size: int = 1,
        steps: int = 30,
        seed: int = 1,
        upscaler: str = "",
        use_face_enhancer: bool = False,
        use_hires_fix: bool = False,
    ) -> list[bytes]:
        """
        Runs the Stable Diffusion pipeline on the given prompt and outputs images.
        """
        import torch

        max_embeddings_multiples = self.count_token(p=prompt, n=n_prompt)
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            with torch.autocast("cuda"):
                base_images = self.pipe.text2img(
                    prompt * batch_size,
                    negative_prompt=n_prompt * batch_size,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    max_embeddings_multiples=max_embeddings_multiples,
                    generator=generator,
                ).images

        if upscaler != "":
            upscaled = self.upscale(
                base_images=base_images,
                half_precision=False,
                tile=700,
                upscaler=upscaler,
                use_face_enhancer=use_face_enhancer,
                use_hires_fix=use_hires_fix,
            )
            base_images.extend(upscaled)
            if use_hires_fix:
                torch.cuda.empty_cache()
                for img in upscaled:
                    with torch.inference_mode():
                        with torch.autocast("cuda"):
                            hires_fixed = self.pipe.img2img(
                                prompt=prompt * batch_size,
                                negative_prompt=n_prompt * batch_size,
                                num_inference_steps=steps,
                                strength=0.3,
                                guidance_scale=7.5,
                                max_embeddings_multiples=max_embeddings_multiples,
                                generator=generator,
                                image=img,
                            ).images
                    base_images.extend(hires_fixed)
                torch.cuda.empty_cache()

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
        half_precision: bool = False,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        upscaler: str = "",
        use_face_enhancer: bool = False,
        use_hires_fix: bool = False,
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

        from gfpgan import GFPGANer

        if use_face_enhancer:
            face_enhancer = GFPGANer(
                model_path=os.path.join(BASE_CACHE_PATH, "esrgan", "GFPGANv1.3.pth"),
                upscale=netscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )

        torch.cuda.empty_cache()
        upscaled_imgs = []
        with tqdm(total=len(base_images)) as progress_bar:
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

                upscaled_imgs.append(Image.fromarray(enhance_result))
                progress_bar.update(1)

        torch.cuda.empty_cache()

        return upscaled_imgs

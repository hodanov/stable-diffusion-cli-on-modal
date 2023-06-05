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

        scheduler = diffusers.EulerAncestralDiscreteScheduler.from_pretrained(
            cache_path,
            subfolder="scheduler",
        )

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            cache_path,
            scheduler=scheduler,
            custom_pipeline="lpw_stable_diffusion",
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        self.upscaler = diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        ).to("cuda")
        self.upscaler.enable_xformers_memory_efficient_attention()

        # model_id = "stabilityai/stable-diffusion-x4-upscaler"
        # self.upscaler = diffusers.StableDiffusionUpscalePipeline.from_pretrained(
        #     , revision="fp16", torch_dtype=torch.float16
        # ).to("cuda")
        # self.upscaler.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(self, inputs: dict[str, int | str]) -> list[bytes]:
        """
        Runs the Stable Diffusion pipeline on the given prompt and outputs images.
        """
        import torch

        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    [inputs["prompt"]] * int(inputs["batch_size"]),
                    negative_prompt=[inputs["n_prompt"]] * int(inputs["batch_size"]),
                    height=inputs["height"],
                    width=inputs["width"],
                    num_inference_steps=inputs["steps"],
                    guidance_scale=7.5,
                    max_embeddings_multiples=inputs["max_embeddings_multiples"],
                ).images

        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())

        if inputs["upscaler"] != "":
            upscaled_images = self.upscaler(
                prompt=inputs["prompt"],
                image=images,
                num_inference_steps=inputs["steps"],
                guidance_scale=0,
            ).images
            for image in upscaled_images:
                with io.BytesIO() as buf:
                    image.save(buf, format="PNG")
                    image_output.append(buf.getvalue())

        return image_output


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

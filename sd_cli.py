from __future__ import annotations
import io
import os
import time
from datetime import date
from pathlib import Path
from modal import Image, Secret, Stub, method

stub = Stub("stable-diffusion-cli")

BASE_CACHE_PATH = "/vol/cache"


def download_models():
    """
    Downloads the model from Hugging Face and saves it to the cache path using
    diffusers.StableDiffusionPipeline.from_pretrained().
    """
    import diffusers
    import torch

    hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]
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
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


stub_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torch",
        "torchvision",
        "transformers~=4.25.1",
        "triton",
        "safetensors",
        "torch>=2.0",
    )
    .pip_install("xformers", pre=True)
    .run_function(
        download_models,
        secrets=[Secret.from_dotenv(__file__)],
    )
)
stub.image = stub_image


# @stub.cls(gpu="A10G", secrets=[Secret.from_name("my-huggingface-secret")])
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
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,  # important if steps are <= 10
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            cache_path,
            scheduler=scheduler,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(
        self,
        prompt: str,
        n_prompt: str,
        steps: int = 30,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> list[bytes]:
        """
        Runs the Stable Diffusion pipeline on the given prompt and outputs images.
        """
        import torch

        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    [prompt] * batch_size,
                    negative_prompt=[n_prompt] * batch_size,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                ).images

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


@stub.local_entrypoint()
def entrypoint(
    prompt: str,
    n_prompt: str,
    samples: int = 5,
    steps: int = 30,
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
):
    """
    This function is the entrypoint for the Runway CLI.
    The function pass the given prompt to StableDiffusion on Modal,
    gets back a list of images and outputs images to local.

    The function is called with the following arguments:
    - prompt: the prompt to run inference on
    - n_prompt: the negative prompt to run inference on
    - samples: the number of samples to generate
    - steps: the number of steps to run inference for
    - batch_size: the batch size to use
    - height: the height of the output image
    - width: the width of the output image
    """
    print(f"steps => {steps}, sapmles => {samples}, batch_size => {batch_size}")

    directory = Path(f"./outputs/{date.today().strftime('%Y-%m-%d')}")
    if not directory.exists():
        directory.mkdir(exist_ok=True, parents=True)

    stable_diffusion = StableDiffusion()
    for i in range(samples):
        start_time = time.time()
        images = stable_diffusion.run_inference.call(
            prompt,
            n_prompt,
            steps,
            batch_size,
            height,
            width,
        )
        total_time = time.time() - start_time
        print(
            f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image)."
        )
        for j, image_bytes in enumerate(images):
            output_path = directory / f"output_{j}_{i}.png"
            print(f"Saving it to {output_path}")
            with open(output_path, "wb") as file:
                file.write(image_bytes)

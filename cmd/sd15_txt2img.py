import time

import modal
import util

app = modal.App("run-stable-diffusion-cli")
run_inference = modal.Function.from_name("stable-diffusion-cli", "SDXLTxt2Img.run_inference")


@app.local_entrypoint()
def main(
    prompt: str,
    n_prompt: str,
    height: int = 512,
    width: int = 512,
    samples: int = 5,
    batch_size: int = 1,
    steps: int = 20,
    seed: int = -1,
    use_upscaler: str = "",
    fix_by_controlnet_tile: str = "False",
    output_format: str = "png",
):
    """
    This function is the entrypoint for the Runway CLI.
    The function pass the given prompt to StableDiffusion on Modal,
    gets back a list of images and outputs images to local.
    """
    directory = util.make_directory()
    seed_generated = seed
    for i in range(samples):
        if seed == -1:
            seed_generated = util.generate_seed()
        start_time = time.time()
        images = run_inference.remote(
            prompt=prompt,
            n_prompt=n_prompt,
            height=height,
            width=width,
            batch_size=batch_size,
            steps=steps,
            seed=seed_generated,
            use_upscaler=use_upscaler == "True",
            fix_by_controlnet_tile=fix_by_controlnet_tile == "True",
            output_format=output_format,
        )
        util.save_images(directory, images, seed_generated, i, output_format)
        total_time = time.time() - start_time
        print(f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image).")

    prompts: dict[str, int | str] = {
        "prompt": prompt,
        "n_prompt": n_prompt,
        "height": height,
        "width": width,
        "samples": samples,
        "batch_size": batch_size,
        "steps": steps,
    }
    util.save_prompts(prompts)

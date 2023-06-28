import time

import modal

stub = modal.Stub("run-stable-diffusion-cli")
stub.run_inference = modal.Function.from_name("stable-diffusion-cli", "StableDiffusion.run_inference")


@stub.local_entrypoint()
def main(
    prompt: str,
    n_prompt: str,
    height: int = 512,
    width: int = 512,
    samples: int = 5,
    batch_size: int = 1,
    steps: int = 20,
    seed: int = -1,
    upscaler: str = "",
    use_face_enhancer: str = "False",
    use_hires_fix: str = "False",
):
    """
    This function is the entrypoint for the Runway CLI.
    The function pass the given prompt to StableDiffusion on Modal,
    gets back a list of images and outputs images to local.
    """
    import util

    directory = util.make_directory()
    seed_generated = seed
    for i in range(samples):
        if seed == -1:
            seed_generated = util.generate_seed()
        start_time = time.time()
        images = stub.app.run_inference.call(
            prompt=prompt,
            n_prompt=n_prompt,
            height=height,
            width=width,
            batch_size=batch_size,
            steps=steps,
            seed=seed_generated,
            upscaler=upscaler,
            use_face_enhancer=use_face_enhancer == "True",
            use_hires_fix=use_hires_fix == "True",
        )
        util.save_images(directory, images, seed_generated, i)
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

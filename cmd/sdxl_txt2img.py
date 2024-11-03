import logging
import time

import domain
import modal

app = modal.App("run-stable-diffusion-cli")
run_inference = modal.Function.from_name(
    "stable-diffusion-cli",
    "SDXLTxt2Img.run_inference",
)


@app.local_entrypoint()
def main(
    prompt: str,
    n_prompt: str,
    height: int = 1024,
    width: int = 1024,
    samples: int = 5,
    steps: int = 20,
    seed: int = -1,
    use_upscaler: str = "False",
    output_format: str = "png",
) -> None:
    """This function is the entrypoint for the Runway CLI.
    The function pass the given prompt to StableDiffusion on Modal,
    gets back a list of images and outputs images to local.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run-stable-diffusion-cli")

    output_directory = domain.OutputDirectory()
    directory_path = output_directory.make_directory()
    logger.info("Made a directory: %s", directory_path)

    prompts = domain.Prompts(prompt, n_prompt, height, width, samples, steps)
    sd_output_manager = domain.StableDiffusionOutputManger(prompts, directory_path)

    for sample_index in range(samples):
        new_seed = domain.Seed(seed)
        start_time = time.time()
        images = run_inference.remote(
            prompt=prompt,
            n_prompt=n_prompt,
            height=height,
            width=width,
            steps=steps,
            seed=new_seed.value,
            use_upscaler=use_upscaler == "True",
            output_format=output_format,
        )

        for generated_image_index, image_bytes in enumerate(images):
            saved_path = sd_output_manager.save_image(
                image_bytes,
                new_seed.value,
                sample_index,
                generated_image_index,
                output_format,
            )
            logger.info("Saved image to the: %s", saved_path)

        total_time = time.time() - start_time
        logger.info("Sample %s, took %ss (%ss / image).", sample_index, total_time, (total_time) / len(images))

    saved_prompts_path = sd_output_manager.save_prompts()
    logger.info("Saved prompts: %s", saved_prompts_path)

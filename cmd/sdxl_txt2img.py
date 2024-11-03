from __future__ import annotations

import logging
import time

import modal
from domain import OutputDirectory, Prompts, Seed, StableDiffusionOutputManger
from infrasctucture import new_txt2img


@modal.App("run-stable-diffusion-cli").local_entrypoint()
def main(
    version: str,
    prompt: str,
    n_prompt: str,
    height: int = 1024,
    width: int = 1024,
    samples: int = 5,
    steps: int = 20,
    seed: int = -1,
    use_upscaler: str = "False",
    fix_by_controlnet_tile: str = "True",
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

    output_directory = OutputDirectory()
    directory_path = output_directory.make_directory()
    logger.info("Made a directory: %s", directory_path)

    prompts = Prompts(prompt, n_prompt, height, width, samples, steps)
    sd_output_manager = StableDiffusionOutputManger(prompts, directory_path)

    txt2img = new_txt2img(
        version,
        prompts,
        output_format,
        use_upscaler=use_upscaler == "True",
        fix_by_controlnet_tile=fix_by_controlnet_tile == "True",
    )

    for sample_index in range(samples):
        start_time = time.time()
        new_seed = Seed(seed)
        images = txt2img.run_inference(new_seed)
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

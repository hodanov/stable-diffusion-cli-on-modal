from __future__ import annotations

import logging
import time
from pathlib import Path

import modal
from domain import OutputDirectory, Seed, VideoOutputManager, VideoPrompts
from infrastructure import new_ti2v


@modal.App("run-wan-ti2v-cli").local_entrypoint()
def main(
    prompt: str,
    n_prompt: str,
    image_path: str,
    height: int = 704,
    width: int = 1280,
    samples: int = 1,
    steps: int = 50,
    seed: int = -1,
    num_frames: int = 121,
    fps: int = 24,
    guidance_scale: float = 5.0,
    use_image_aspect: str = "True",
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run-wan-ti2v-cli")

    image_file = Path(image_path)
    if not image_file.exists():
        msg = f"image_path does not exist: {image_file}"
        raise FileNotFoundError(msg)

    output_directory = OutputDirectory()
    directory_path = output_directory.make_directory()
    logger.info("Made a directory: %s", directory_path)

    prompts = VideoPrompts(
        prompt=prompt,
        n_prompt=n_prompt,
        height=height,
        width=width,
        samples=samples,
        steps=steps,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        use_image_aspect=use_image_aspect == "True",
        image_path=image_path,
    )
    output_manager = VideoOutputManager(prompts, directory_path)

    ti2v = new_ti2v(
        prompts=prompts,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        use_image_aspect=use_image_aspect == "True",
    )

    with image_file.open("rb") as f:
        image_bytes = f.read()

    for sample_index in range(samples):
        start_time = time.time()
        new_seed = Seed(seed)
        video = ti2v.run_inference(new_seed, image_bytes)
        saved_path = output_manager.save_video(video, new_seed.value, sample_index)
        logger.info("Saved video to the: %s", saved_path)
        total_time = time.time() - start_time
        logger.info(
            "Sample %s, took %ss.",
            sample_index,
            round(total_time, 3),
        )

    saved_prompts_path = output_manager.save_prompts()
    logger.info("Saved prompts: %s", saved_prompts_path)

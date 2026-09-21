from __future__ import annotations

import logging
import time

import modal
from domain import (
    InputImage,
    OutputDirectory,
    Seed,
    VideoOutputManager,
    VideoPrompts,
    parse_bool_flag,
    unset_if_negative,
)
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
    guidance_scale_2: float = -1.0,
    use_image_aspect: str = "True",
    use_upscaler: str = "False",
    use_face_restore: str = "False",
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run-wan-ti2v-cli")

    # Decode before anything starts a GPU container, so an unusable image fails
    # here instead of after the container has spun up.
    input_image = InputImage.from_path(image_path)
    logger.info("Loaded input image: %s (%s)", image_path, input_image.source_format)

    output_directory = OutputDirectory()
    directory_path = output_directory.make_directory()
    logger.info("Made a directory: %s", directory_path)

    # -1 means "unset"; the low-noise expert then reuses guidance_scale.
    resolved_guidance_scale_2 = unset_if_negative(guidance_scale_2)
    resolved_use_image_aspect = parse_bool_flag(use_image_aspect)
    resolved_use_upscaler = parse_bool_flag(use_upscaler)
    resolved_use_face_restore = parse_bool_flag(use_face_restore)
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
        guidance_scale_2=resolved_guidance_scale_2,
        use_image_aspect=resolved_use_image_aspect,
        use_upscaler=resolved_use_upscaler,
        use_face_restore=resolved_use_face_restore,
        image_path=image_path,
    )
    output_manager = VideoOutputManager(prompts, directory_path)

    ti2v = new_ti2v(
        prompts=prompts,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        guidance_scale_2=resolved_guidance_scale_2,
        use_image_aspect=resolved_use_image_aspect,
        use_upscaler=resolved_use_upscaler,
        use_face_restore=resolved_use_face_restore,
    )

    for sample_index in range(samples):
        start_time = time.time()
        new_seed = Seed(seed)
        video = ti2v.run_inference(new_seed, input_image.png_bytes)
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

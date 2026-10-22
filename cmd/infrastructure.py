from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from domain import Prompts, Seed, VideoPrompts

# How often to report that a spawned call is still running, and when to give up.
# A sample normally takes a few minutes including a cold start; the margin covers
# a preempted container being restarted on another host.
POLL_INTERVAL_SECONDS = 30
MAX_WAIT_SECONDS = 15 * 60

logger = logging.getLogger(__name__)


def wait_for_result[T](
    call: modal.FunctionCall[T],
    *,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    max_wait: float = MAX_WAIT_SECONDS,
) -> T:
    """
    Wait for a spawned call, logging while it runs and cancelling it on give-up.

    Modal restarts a preempted input on a new container without telling the
    caller, and `modal run` does not stream the deployed app's logs, so a plain
    `.remote()` would wait silently and without limit.
    """
    started = time.monotonic()
    logger.info("Waiting for %s.", call.object_id)
    try:
        while True:
            try:
                return call.get(timeout=poll_interval)
            except TimeoutError:
                elapsed = time.monotonic() - started
                if elapsed >= max_wait:
                    call.cancel()
                    msg = f"Gave up on {call.object_id} after {elapsed:.0f}s. Check it on the Modal dashboard."
                    raise TimeoutError(msg) from None
                logger.info(
                    "Still waiting for %s (%.0fs elapsed).", call.object_id, elapsed
                )
    except KeyboardInterrupt:
        # A spawned call keeps running on its GPU after the local CLI exits.
        call.cancel()
        raise


class Txt2ImgInterface(ABC):
    @abstractmethod
    def run_inference(self, seed: Seed) -> list[bytes]:
        pass


class Ti2VInterface(ABC):
    @abstractmethod
    def run_inference(self, seed: Seed, image_bytes: bytes | None) -> bytes:
        pass


class SDXLTxt2Img(Txt2ImgInterface):
    def __init__(
        self,
        prompts: Prompts,
        output_format: str,
        *,
        use_upscaler: bool,
    ) -> None:
        self.__prompts = prompts
        self.__output_format = output_format
        self.__use_upscaler = use_upscaler
        self.__sdxl_txt2_img = modal.Cls.from_name(
            "sdxl-cli",
            "SDXLTxt2Img",
        )

    def run_inference(self, seed: Seed) -> list[bytes]:
        call = self.__sdxl_txt2_img().run_inference.spawn(
            prompt=self.__prompts.prompt,
            n_prompt=self.__prompts.n_prompt,
            height=self.__prompts.height,
            width=self.__prompts.width,
            steps=self.__prompts.steps,
            seed=seed.value,
            use_upscaler=self.__use_upscaler,
            output_format=self.__output_format,
        )
        return wait_for_result(call)


class WanTI2V(Ti2VInterface):
    def __init__(
        self,
        prompts: VideoPrompts,
        *,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        guidance_scale_2: float | None,
        use_image_aspect: bool,
        use_upscaler: bool,
        use_face_restore: bool,
    ) -> None:
        self.__prompts = prompts
        self.__num_frames = num_frames
        self.__fps = fps
        self.__guidance_scale = guidance_scale
        self.__guidance_scale_2 = guidance_scale_2
        self.__use_image_aspect = use_image_aspect
        self.__use_upscaler = use_upscaler
        self.__use_face_restore = use_face_restore
        self.__wan_ti2v = modal.Cls.from_name(
            "wan-i2v-cli",
            "WanTI2V",
        )

    def run_inference(self, seed: Seed, image_bytes: bytes | None) -> bytes:
        return self.__wan_ti2v().run_inference.remote(
            prompt=self.__prompts.prompt,
            n_prompt=self.__prompts.n_prompt,
            height=self.__prompts.height,
            width=self.__prompts.width,
            steps=self.__prompts.steps,
            seed=seed.value,
            num_frames=self.__num_frames,
            fps=self.__fps,
            guidance_scale=self.__guidance_scale,
            guidance_scale_2=self.__guidance_scale_2,
            use_image_aspect=self.__use_image_aspect,
            use_upscaler=self.__use_upscaler,
            use_face_restore=self.__use_face_restore,
            image_bytes=image_bytes,
        )


def new_txt2img(
    version: str,
    prompts: Prompts,
    output_format: str,
    *,
    use_upscaler: bool,
) -> Txt2ImgInterface:
    match version:
        case "sdxl":
            return SDXLTxt2Img(
                prompts=prompts,
                use_upscaler=use_upscaler,
                output_format=output_format,
            )
        case _:
            msg = f"Invalid version: {version}. Only 'sdxl' is supported now."
            raise ValueError(msg)


def new_ti2v(
    prompts: VideoPrompts,
    *,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    guidance_scale_2: float | None,
    use_image_aspect: bool,
    use_upscaler: bool,
    use_face_restore: bool,
) -> Ti2VInterface:
    return WanTI2V(
        prompts=prompts,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        use_image_aspect=use_image_aspect,
        use_upscaler=use_upscaler,
        use_face_restore=use_face_restore,
    )

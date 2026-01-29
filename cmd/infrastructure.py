from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from domain import Prompts, Seed, VideoPrompts


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
            "stable-diffusion-cli",
            "SDXLTxt2Img",
        )

    def run_inference(self, seed: Seed) -> list[bytes]:
        return self.__sdxl_txt2_img().run_inference.remote(
            prompt=self.__prompts.prompt,
            n_prompt=self.__prompts.n_prompt,
            height=self.__prompts.height,
            width=self.__prompts.width,
            steps=self.__prompts.steps,
            seed=seed.value,
            use_upscaler=self.__use_upscaler,
            output_format=self.__output_format,
        )


class WanTI2V(Ti2VInterface):
    def __init__(
        self,
        prompts: VideoPrompts,
        *,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        use_image_aspect: bool,
    ) -> None:
        self.__prompts = prompts
        self.__num_frames = num_frames
        self.__fps = fps
        self.__guidance_scale = guidance_scale
        self.__use_image_aspect = use_image_aspect
        self.__wan_ti2v = modal.Cls.from_name(
            "stable-diffusion-cli",
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
            use_image_aspect=self.__use_image_aspect,
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
    use_image_aspect: bool,
) -> Ti2VInterface:
    return WanTI2V(
        prompts=prompts,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        use_image_aspect=use_image_aspect,
    )

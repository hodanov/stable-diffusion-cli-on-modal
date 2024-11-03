from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from domain import Prompts, Seed


class Txt2ImgInterface(ABC):
    @abstractmethod
    def run_inference(self, seed: Seed) -> list[bytes]:
        pass


class SDXLTxt2Img(Txt2ImgInterface):
    def __init__(self, prompts: Prompts, output_format: str, *, use_upscaler: bool) -> None:
        self.__prompts = prompts
        self.__output_format = output_format
        self.__use_upscaler = use_upscaler
        self.__run_inference = modal.Function.from_name(
            "stable-diffusion-cli",
            "SDXLTxt2Img.run_inference",
        )

    def run_inference(self, seed: Seed) -> list[bytes]:
        return self.__run_inference.remote(
            prompt=self.__prompts.prompt,
            n_prompt=self.__prompts.n_prompt,
            height=self.__prompts.height,
            width=self.__prompts.width,
            steps=self.__prompts.steps,
            seed=seed.value,
            use_upscaler=self.__use_upscaler,
            output_format=self.__output_format,
        )


class SD15Txt2Img(Txt2ImgInterface):
    def __init__(
        self,
        prompts: Prompts,
        output_format: str,
        *,
        use_upscaler: bool,
        fix_by_controlnet_tile: bool,
    ) -> None:
        self.__prompts = prompts
        self.__output_format = output_format
        self.__use_upscaler = use_upscaler
        self.__fix_by_controlnet_tile = fix_by_controlnet_tile
        self.__run_inference = modal.Function.from_name(
            "stable-diffusion-cli",
            "SD15.run_txt2img_inference",
        )

    def run_inference(self, seed: Seed) -> list[bytes]:
        return self.__run_inference.remote(
            prompt=self.__prompts.prompt,
            n_prompt=self.__prompts.n_prompt,
            height=self.__prompts.height,
            width=self.__prompts.width,
            batch_size=1,
            steps=self.__prompts.steps,
            seed=seed.value,
            use_upscaler=self.__use_upscaler,
            fix_by_controlnet_tile=self.__fix_by_controlnet_tile,
            output_format=self.__output_format,
        )


def new_txt2img(
    version: str,
    prompts: Prompts,
    output_format: str,
    *,
    use_upscaler: bool,
    fix_by_controlnet_tile: bool,
) -> Txt2ImgInterface:
    match version:
        case "sd15":
            return SD15Txt2Img(
                prompts=prompts,
                output_format=output_format,
                use_upscaler=use_upscaler,
                fix_by_controlnet_tile=fix_by_controlnet_tile,
            )
        case "sdxl":
            return SDXLTxt2Img(
                prompts=prompts,
                use_upscaler=use_upscaler,
                output_format=output_format,
            )
        case _:
            msg = f"Invalid version: {version}. Must be 'sd15' or 'sdxl'."
            raise ValueError(msg)

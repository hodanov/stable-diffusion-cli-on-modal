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

"""Utility functions for the script."""

from __future__ import annotations

import secrets
import time
from datetime import date
from pathlib import Path


class Seed:
    def __init__(self, seed: int) -> None:
        if seed != -1:
            self.__value = seed
            return

        self.__value = self.__generate_seed()

    def __generate_seed(self) -> int:
        max_limit_value = 4294967295
        return secrets.randbelow(max_limit_value)

    @property
    def value(self) -> int:
        return self.__value


class Prompts:
    def __init__(
        self,
        prompt: str,
        n_prompt: str,
        height: int,
        width: int,
        samples: int,
        steps: int,
    ) -> None:
        if prompt == "":
            msg = "prompt should not be empty."
            raise ValueError(msg)

        if n_prompt == "":
            msg = "n_prompt should not be empty."
            raise ValueError(msg)

        if height <= 0:
            msg = "height should be positive."
            raise ValueError(msg)

        if width <= 0:
            msg = "width should be positive."
            raise ValueError(msg)

        if samples <= 0:
            msg = "samples should be positive."
            raise ValueError(msg)

        if steps <= 0:
            msg = "steps should be positive."
            raise ValueError(msg)

        self.__dict: dict[str, int | str] = {
            "prompt": prompt,
            "n_prompt": n_prompt,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
        }

    @property
    def dict(self) -> dict[str, int | str]:
        return self.__dict


class OutputDirectory:
    def __init__(self) -> None:
        self.__output_directory_name = "outputs"
        self.__date_today = date.today().strftime("%Y-%m-%d")
        self.__make_path()

    def __make_path(self) -> None:
        self.__path = Path(f"{self.__output_directory_name}/{self.__date_today}")

    def make_directory(self) -> Path:
        """Make a directory for saving outputs."""
        if not self.__path.exists():
            self.__path.mkdir(exist_ok=True, parents=True)

        return self.__path


class StableDiffusionOutputManger:
    def __init__(self, prompts: Prompts, output_directory: Path) -> None:
        self.__prompts = prompts
        self.__output_directory = output_directory

    def save_prompts(self) -> str:
        """Save prompts to a file."""
        prompts_filename = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        output_path = f"{self.__output_directory}/prompts_{prompts_filename}.txt"
        with Path(output_path).open("wb") as file:
            for name, value in self.__prompts.dict.items():
                file.write(f"{name} = {value!r}\n".encode())

        return output_path

    def save_image(
        self,
        image: bytes,
        seed: int,
        i: int,
        j: int,
        output_format: str = "png",
    ) -> str:
        """Save image to a file."""
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        filename = f"{formatted_time}_{seed}_{i}_{j}.{output_format}"
        output_path = f"{self.__output_directory}/{filename}"
        with Path(output_path).open("wb") as file:
            file.write(image)

        return output_path

"""Utility functions for the script."""

from __future__ import annotations

import io
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

import PIL.Image


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

        self.__prompt = prompt
        self.__n_prompt = n_prompt
        self.__height = height
        self.__width = width
        self.__samples = samples
        self.__steps = steps

    @property
    def prompt(self) -> str:
        return self.__prompt

    @property
    def n_prompt(self) -> str:
        return self.__n_prompt

    @property
    def height(self) -> int:
        return self.__height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def samples(self) -> int:
        return self.__samples

    @property
    def steps(self) -> int:
        return self.__steps


class VideoPrompts:
    def __init__(
        self,
        *,
        prompt: str,
        n_prompt: str,
        height: int,
        width: int,
        samples: int,
        steps: int,
        num_frames: int,
        fps: int,
        guidance_scale: float,
        guidance_scale_2: float | None,
        use_image_aspect: bool,
        use_upscaler: bool,
        use_face_restore: bool,
        image_path: str,
    ) -> None:
        if prompt == "":
            msg = "prompt should not be empty."
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

        if num_frames <= 0:
            msg = "num_frames should be positive."
            raise ValueError(msg)

        if fps <= 0:
            msg = "fps should be positive."
            raise ValueError(msg)

        if image_path == "":
            msg = "image_path should not be empty."
            raise ValueError(msg)

        if guidance_scale_2 is not None and guidance_scale_2 <= 0:
            msg = "guidance_scale_2 should be positive."
            raise ValueError(msg)

        self.__prompt = prompt
        self.__n_prompt = n_prompt
        self.__height = height
        self.__width = width
        self.__samples = samples
        self.__steps = steps
        self.__num_frames = num_frames
        self.__fps = fps
        self.__guidance_scale = guidance_scale
        self.__guidance_scale_2 = guidance_scale_2
        self.__use_image_aspect = use_image_aspect
        self.__use_upscaler = use_upscaler
        self.__use_face_restore = use_face_restore
        self.__image_path = image_path

    @property
    def prompt(self) -> str:
        return self.__prompt

    @property
    def n_prompt(self) -> str:
        return self.__n_prompt

    @property
    def height(self) -> int:
        return self.__height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def samples(self) -> int:
        return self.__samples

    @property
    def steps(self) -> int:
        return self.__steps

    @property
    def num_frames(self) -> int:
        return self.__num_frames

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def guidance_scale(self) -> float:
        return self.__guidance_scale

    @property
    def guidance_scale_2(self) -> float | None:
        return self.__guidance_scale_2

    @property
    def use_image_aspect(self) -> bool:
        return self.__use_image_aspect

    @property
    def use_upscaler(self) -> bool:
        return self.__use_upscaler

    @property
    def use_face_restore(self) -> bool:
        return self.__use_face_restore

    @property
    def image_path(self) -> str:
        return self.__image_path


class InputImage:
    """A source image for TI2V, normalized to PNG bytes."""

    def __init__(self, png_bytes: bytes, source_format: str) -> None:
        self.__png_bytes = png_bytes
        self.__source_format = source_format

    @classmethod
    def from_path(cls, image_path: str) -> InputImage:
        """
        Load an image file and normalize it to PNG bytes.

        Any format Pillow can decode is accepted (PNG, AVIF, JPEG, WebP, ...).
        Normalizing here keeps the inference container independent of which
        decoders its own Pillow build ships with, and rejects unreadable files
        before a GPU container is started.
        """
        image_file = Path(image_path)
        if not image_file.exists():
            msg = f"image_path does not exist: {image_file}"
            raise FileNotFoundError(msg)

        try:
            with PIL.Image.open(image_file) as image:
                source_format = image.format or "unknown"
                # The pipeline feeds the image as RGB anyway; converting here
                # also forces the decode, so a broken file fails at this point.
                rgb_image = image.convert("RGB")
        except PIL.UnidentifiedImageError as e:
            msg = f"image_path is not an image Pillow can decode: {image_file}"
            raise ValueError(msg) from e

        with io.BytesIO() as buf:
            rgb_image.save(buf, format="PNG")
            return cls(buf.getvalue(), source_format)

    @property
    def png_bytes(self) -> bytes:
        return self.__png_bytes

    @property
    def source_format(self) -> str:
        return self.__source_format


class OutputDirectory:
    def __init__(self) -> None:
        self.__output_directory_name = "outputs"
        self.__date_today = datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d")
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
            file.writelines(f"{key} = {value!r}\n".encode() for key, value in vars(self.__prompts).items())

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


class VideoOutputManager:
    def __init__(self, prompts: VideoPrompts, output_directory: Path) -> None:
        self.__prompts = prompts
        self.__output_directory = output_directory

    def save_prompts(self) -> str:
        prompts_filename = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        output_path = f"{self.__output_directory}/prompts_{prompts_filename}.txt"
        with Path(output_path).open("wb") as file:
            file.writelines(f"{key} = {value!r}\n".encode() for key, value in vars(self.__prompts).items())

        return output_path

    def save_video(self, video: bytes, seed: int, i: int) -> str:
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        filename = f"{formatted_time}_{seed}_{i}.mp4"
        output_path = f"{self.__output_directory}/{filename}"
        with Path(output_path).open("wb") as file:
            file.write(video)

        return output_path

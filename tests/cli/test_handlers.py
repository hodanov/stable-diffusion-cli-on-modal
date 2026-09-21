from __future__ import annotations

from typing import TYPE_CHECKING, Any

import PIL.Image
import pytest
import ti2v_handler
import txt2img_handler

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from domain import Prompts, Seed, VideoPrompts


class FakeTxt2Img:
    def __init__(self, images_per_call: int) -> None:
        self.images_per_call = images_per_call
        self.seeds: list[int] = []

    def run_inference(self, seed: Seed) -> list[bytes]:
        self.seeds.append(seed.value)
        return [f"image-{seed.value}-{i}".encode() for i in range(self.images_per_call)]


class FakeTi2V:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bytes | None]] = []

    def run_inference(self, seed: Seed, image_bytes: bytes | None) -> bytes:
        self.calls.append((seed.value, image_bytes))
        return b"mp4-bytes"


@pytest.fixture
def txt2img_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    recorded: dict[str, Any] = {"client": FakeTxt2Img(images_per_call=2)}

    def fake_new_txt2img(
        version: str,
        prompts: Prompts,
        output_format: str,
        *,
        use_upscaler: bool,
    ) -> FakeTxt2Img:
        recorded["version"] = version
        recorded["prompts"] = prompts
        recorded["output_format"] = output_format
        recorded["use_upscaler"] = use_upscaler
        return recorded["client"]

    monkeypatch.setattr(txt2img_handler, "new_txt2img", fake_new_txt2img)
    return recorded


@pytest.fixture
def ti2v_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    recorded: dict[str, Any] = {"client": FakeTi2V()}

    def fake_new_ti2v(prompts: VideoPrompts, **kwargs: object) -> FakeTi2V:
        recorded["prompts"] = prompts
        recorded.update(kwargs)
        return recorded["client"]

    monkeypatch.setattr(ti2v_handler, "new_ti2v", fake_new_ti2v)
    return recorded


def make_image(path: Path, image_format: str = "PNG") -> Path:
    PIL.Image.new("RGB", (8, 6)).save(path, format=image_format)
    return path


class TestTxt2ImgMain:
    def test_saves_every_generated_image_and_the_prompts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        txt2img_factory: dict[str, Any],
        raw_entrypoint: Callable[[Any], Callable[..., Any]],
    ) -> None:
        monkeypatch.chdir(tmp_path)

        raw_entrypoint(txt2img_handler.main)(
            version="sdxl",
            prompt="a cat",
            n_prompt="blurry",
            height=512,
            width=768,
            samples=3,
            steps=20,
            seed=42,
            use_upscaler="True",
            output_format="avif",
        )

        assert txt2img_factory["version"] == "sdxl"
        assert txt2img_factory["output_format"] == "avif"
        assert txt2img_factory["use_upscaler"] is True
        assert txt2img_factory["prompts"].prompt == "a cat"
        assert txt2img_factory["client"].seeds == [42, 42, 42]

        output_directory = next((tmp_path / "outputs").iterdir())
        images = sorted(p.name for p in output_directory.glob("*.avif"))
        prompt_files = list(output_directory.glob("prompts_*.txt"))
        # 3 samples x 2 images per inference call.
        assert len(images) == 6
        assert len(prompt_files) == 1
        assert "_Prompts__prompt = 'a cat'" in prompt_files[0].read_text()

    def test_false_flag_is_passed_through(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        txt2img_factory: dict[str, Any],
        raw_entrypoint: Callable[[Any], Callable[..., Any]],
    ) -> None:
        monkeypatch.chdir(tmp_path)

        raw_entrypoint(txt2img_handler.main)(
            version="sdxl",
            prompt="a cat",
            n_prompt="",
            samples=1,
            use_upscaler="False",
        )

        assert txt2img_factory["use_upscaler"] is False

    def test_invalid_prompts_fail_before_inference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        txt2img_factory: dict[str, Any],
        raw_entrypoint: Callable[[Any], Callable[..., Any]],
    ) -> None:
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="prompt should not be empty"):
            raw_entrypoint(txt2img_handler.main)(version="sdxl", prompt="", n_prompt="")

        assert txt2img_factory["client"].seeds == []


class TestTi2VMain:
    def test_saves_a_video_per_sample_and_the_prompts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ti2v_factory: dict[str, Any],
        raw_entrypoint: Callable[[Any], Callable[..., Any]],
    ) -> None:
        image_path = make_image(tmp_path / "input.png")
        monkeypatch.chdir(tmp_path)

        raw_entrypoint(ti2v_handler.main)(
            prompt="a cat",
            n_prompt="blurry",
            image_path=str(image_path),
            height=704,
            width=1280,
            samples=2,
            steps=6,
            seed=7,
            num_frames=81,
            fps=16,
            guidance_scale=1.0,
            guidance_scale_2=-1.0,
            use_image_aspect="True",
            use_upscaler="False",
            use_face_restore="False",
        )

        assert ti2v_factory["guidance_scale_2"] is None
        assert ti2v_factory["use_image_aspect"] is True
        assert ti2v_factory["use_upscaler"] is False
        assert ti2v_factory["prompts"].guidance_scale_2 is None

        calls = ti2v_factory["client"].calls
        assert [seed for seed, _ in calls] == [7, 7]
        # The handler decodes the image once and reuses the PNG bytes.
        assert calls[0][1] == calls[1][1]
        assert calls[0][1].startswith(b"\x89PNG")

        output_directory = next((tmp_path / "outputs").iterdir())
        assert len(list(output_directory.glob("*.mp4"))) == 2
        assert len(list(output_directory.glob("prompts_*.txt"))) == 1

    def test_positive_guidance_scale_2_is_kept(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ti2v_factory: dict[str, Any],
        raw_entrypoint: Callable[[Any], Callable[..., Any]],
    ) -> None:
        image_path = make_image(tmp_path / "input.avif", image_format="AVIF")
        monkeypatch.chdir(tmp_path)

        raw_entrypoint(ti2v_handler.main)(
            prompt="a cat",
            n_prompt="",
            image_path=str(image_path),
            samples=1,
            guidance_scale_2=3.5,
            use_face_restore="True",
        )

        assert ti2v_factory["guidance_scale_2"] == 3.5
        assert ti2v_factory["use_face_restore"] is True

    def test_unreadable_image_fails_before_inference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ti2v_factory: dict[str, Any],
        raw_entrypoint: Callable[[Any], Callable[..., Any]],
    ) -> None:
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            raw_entrypoint(ti2v_handler.main)(
                prompt="a cat",
                n_prompt="",
                image_path=str(tmp_path / "missing.png"),
            )

        assert ti2v_factory["client"].calls == []
        assert not (tmp_path / "outputs").exists()

from __future__ import annotations

import io
import time
from datetime import date
from typing import TYPE_CHECKING, Any

import domain
import PIL.Image
import pytest
from domain import (
    InputImage,
    OutputDirectory,
    Prompts,
    Seed,
    StableDiffusionOutputManger,
    VideoOutputManager,
    VideoPrompts,
    parse_bool_flag,
    unset_if_negative,
)
from PIL import features

if TYPE_CHECKING:
    from pathlib import Path

FIXED_EPOCH = 1_700_000_000.0
FIXED_TIMESTAMP = time.strftime("%Y%m%d%H%M%S", time.localtime(FIXED_EPOCH))


@pytest.fixture
def fixed_time(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(domain.time, "time", lambda: FIXED_EPOCH)
    return FIXED_TIMESTAMP


def prompts_kwargs(**overrides: object) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": "a cat",
        "n_prompt": "blurry",
        "height": 512,
        "width": 768,
        "samples": 2,
        "steps": 20,
    }
    kwargs.update(overrides)
    return kwargs


def video_prompts_kwargs(**overrides: object) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        **prompts_kwargs(),
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 5.0,
        "guidance_scale_2": None,
        "use_image_aspect": True,
        "use_upscaler": False,
        "use_face_restore": False,
        "image_path": "input.png",
    }
    kwargs.update(overrides)
    return kwargs


class TestParseBoolFlag:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("True", True),
            ("False", False),
            # Only the exact string "True" enables a flag.
            ("true", False),
            ("1", False),
            ("", False),
        ],
    )
    def test_parse(self, value: str, *, expected: bool) -> None:
        assert parse_bool_flag(value) is expected


class TestUnsetIfNegative:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (-1.0, None),
            (-0.5, None),
            (0.0, 0.0),
            (1.5, 1.5),
        ],
    )
    def test_resolve(self, value: float, expected: float | None) -> None:
        assert unset_if_negative(value) == expected


class TestSeed:
    def test_keeps_given_value(self) -> None:
        assert Seed(42).value == 42

    def test_keeps_zero(self) -> None:
        assert Seed(0).value == 0

    def test_minus_one_generates_random_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def fake_randbelow(limit: int) -> int:
            calls.append(limit)
            return 12345

        monkeypatch.setattr(domain.secrets, "randbelow", fake_randbelow)

        assert Seed(-1).value == 12345
        assert calls == [4294967295]


class TestPrompts:
    def test_exposes_values(self) -> None:
        prompts = Prompts(**prompts_kwargs())

        assert prompts.prompt == "a cat"
        assert prompts.n_prompt == "blurry"
        assert prompts.height == 512
        assert prompts.width == 768
        assert prompts.samples == 2
        assert prompts.steps == 20

    def test_allows_empty_negative_prompt(self) -> None:
        assert Prompts(**prompts_kwargs(n_prompt="")).n_prompt == ""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"prompt": ""}, "prompt should not be empty"),
            ({"height": 0}, "height should be positive"),
            ({"width": -1}, "width should be positive"),
            ({"samples": 0}, "samples should be positive"),
            ({"steps": 0}, "steps should be positive"),
        ],
    )
    def test_rejects_invalid_values(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            Prompts(**prompts_kwargs(**overrides))


class TestVideoPrompts:
    def test_exposes_values(self) -> None:
        prompts = VideoPrompts(**video_prompts_kwargs(guidance_scale_2=3.5))

        assert prompts.prompt == "a cat"
        assert prompts.n_prompt == "blurry"
        assert prompts.height == 512
        assert prompts.width == 768
        assert prompts.samples == 2
        assert prompts.steps == 20
        assert prompts.num_frames == 81
        assert prompts.fps == 16
        assert prompts.guidance_scale == 5.0
        assert prompts.guidance_scale_2 == 3.5
        assert prompts.use_image_aspect is True
        assert prompts.use_upscaler is False
        assert prompts.use_face_restore is False
        assert prompts.image_path == "input.png"

    def test_allows_unset_guidance_scale_2(self) -> None:
        assert VideoPrompts(**video_prompts_kwargs()).guidance_scale_2 is None

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"prompt": ""}, "prompt should not be empty"),
            ({"height": 0}, "height should be positive"),
            ({"width": 0}, "width should be positive"),
            ({"samples": -1}, "samples should be positive"),
            ({"steps": 0}, "steps should be positive"),
            ({"num_frames": 0}, "num_frames should be positive"),
            ({"fps": 0}, "fps should be positive"),
            ({"image_path": ""}, "image_path should not be empty"),
            ({"guidance_scale_2": 0.0}, "guidance_scale_2 should be positive"),
            ({"guidance_scale_2": -1.0}, "guidance_scale_2 should be positive"),
        ],
    )
    def test_rejects_invalid_values(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            VideoPrompts(**video_prompts_kwargs(**overrides))


class TestInputImage:
    @pytest.mark.parametrize(
        ("save_format", "mode", "suffix"),
        [
            ("PNG", "RGB", ".png"),
            ("PNG", "RGBA", ".png"),
            ("PNG", "P", ".png"),
            ("JPEG", "RGB", ".jpg"),
            ("WEBP", "RGB", ".webp"),
            pytest.param(
                "AVIF",
                "RGB",
                ".avif",
                marks=pytest.mark.skipif(
                    not features.check("avif"), reason="Pillow built without AVIF"
                ),
            ),
        ],
    )
    def test_normalizes_to_rgb_png(
        self, tmp_path: Path, save_format: str, mode: str, suffix: str
    ) -> None:
        source = tmp_path / f"input{suffix}"
        PIL.Image.new(mode, (8, 6)).save(source, format=save_format)

        input_image = InputImage.from_path(str(source))

        assert input_image.source_format == save_format
        with PIL.Image.open(io.BytesIO(input_image.png_bytes)) as decoded:
            assert decoded.format == "PNG"
            assert decoded.mode == "RGB"
            assert decoded.size == (8, 6)

    def test_ignores_file_extension(self, tmp_path: Path) -> None:
        source = tmp_path / "actually_jpeg.png"
        PIL.Image.new("RGB", (4, 4)).save(source, format="JPEG")

        assert InputImage.from_path(str(source)).source_format == "JPEG"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            InputImage.from_path(str(tmp_path / "missing.png"))

    def test_non_image_file_raises_value_error(self, tmp_path: Path) -> None:
        source = tmp_path / "not_image.png"
        source.write_text("hello")

        with pytest.raises(ValueError, match="not an image Pillow can decode"):
            InputImage.from_path(str(source))


class TestOutputDirectory:
    def test_makes_dated_directory_under_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FixedDate(date):
            @classmethod
            def today(cls) -> FixedDate:
                return cls(2026, 1, 2)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(domain, "date", FixedDate)

        path = OutputDirectory().make_directory()

        assert str(path) == "outputs/2026-01-02"
        assert (tmp_path / "outputs" / "2026-01-02").is_dir()

    def test_existing_directory_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        output_directory = OutputDirectory()

        first = output_directory.make_directory()
        (first / "keep.txt").write_text("kept")
        second = output_directory.make_directory()

        assert first == second
        assert (second / "keep.txt").read_text() == "kept"


class TestStableDiffusionOutputManger:
    def test_save_image(self, tmp_path: Path, fixed_time: str) -> None:
        manager = StableDiffusionOutputManger(Prompts(**prompts_kwargs()), tmp_path)

        saved = manager.save_image(
            b"image-bytes", seed=7, i=1, j=2, output_format="avif"
        )

        assert saved == f"{tmp_path}/{fixed_time}_7_1_2.avif"
        assert (tmp_path / f"{fixed_time}_7_1_2.avif").read_bytes() == b"image-bytes"

    def test_save_image_defaults_to_png(self, tmp_path: Path, fixed_time: str) -> None:
        manager = StableDiffusionOutputManger(Prompts(**prompts_kwargs()), tmp_path)

        saved = manager.save_image(b"x", seed=7, i=0, j=0)

        assert saved.endswith(f"{fixed_time}_7_0_0.png")

    def test_save_prompts(self, tmp_path: Path, fixed_time: str) -> None:
        manager = StableDiffusionOutputManger(Prompts(**prompts_kwargs()), tmp_path)

        saved = manager.save_prompts()

        assert saved == f"{tmp_path}/prompts_{fixed_time}.txt"
        # Keys are the name-mangled private attributes of Prompts.
        assert (tmp_path / f"prompts_{fixed_time}.txt").read_text().splitlines() == [
            "_Prompts__prompt = 'a cat'",
            "_Prompts__n_prompt = 'blurry'",
            "_Prompts__height = 512",
            "_Prompts__width = 768",
            "_Prompts__samples = 2",
            "_Prompts__steps = 20",
        ]


class TestVideoOutputManager:
    def test_save_video(self, tmp_path: Path, fixed_time: str) -> None:
        manager = VideoOutputManager(VideoPrompts(**video_prompts_kwargs()), tmp_path)

        saved = manager.save_video(b"mp4-bytes", seed=9, i=3)

        assert saved == f"{tmp_path}/{fixed_time}_9_3.mp4"
        assert (tmp_path / f"{fixed_time}_9_3.mp4").read_bytes() == b"mp4-bytes"

    def test_save_prompts(self, tmp_path: Path, fixed_time: str) -> None:
        manager = VideoOutputManager(VideoPrompts(**video_prompts_kwargs()), tmp_path)

        saved = manager.save_prompts()

        assert saved == f"{tmp_path}/prompts_{fixed_time}.txt"
        lines = (tmp_path / f"prompts_{fixed_time}.txt").read_text().splitlines()
        assert lines[0] == "_VideoPrompts__prompt = 'a cat'"
        assert "_VideoPrompts__guidance_scale_2 = None" in lines
        assert "_VideoPrompts__image_path = 'input.png'" in lines
        assert len(lines) == 14

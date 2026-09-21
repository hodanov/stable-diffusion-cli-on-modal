from __future__ import annotations

from pathlib import Path
from typing import Any

import app_img
import PIL.Image
import pytest
from app_img import (
    BASE_CACHE_PATH,
    BASE_CACHE_PATH_CONTROLNET,
    BASE_CACHE_PATH_LORA,
    BASE_CACHE_PATH_TEXTUAL_INVERSION,
    CommonSetup,
    StableDiffusionCLISetupSDXL,
    double_image_size,
)

MODEL_URL = "https://example.com/model.safetensors"


def make_config(**overrides: object) -> dict[str, Any]:
    config: dict[str, Any] = {
        "version": "sdxl",
        "model": {"name": "my-sdxl", "url": MODEL_URL},
    }
    config.update(overrides)
    return config


class Logins:
    def __init__(self) -> None:
        self.tokens: list[str] = []


@pytest.fixture
def logins(monkeypatch: pytest.MonkeyPatch) -> Logins:
    recorded = Logins()
    monkeypatch.setattr(app_img, "login", recorded.tokens.append)
    return recorded


class TestStableDiffusionCLISetupSDXL:
    def test_accepts_a_valid_config(self, logins: Logins) -> None:
        StableDiffusionCLISetupSDXL(make_config(), "hf-token")

        assert logins.tokens == ["hf-token"]

    def test_empty_token_does_not_log_in(self, logins: Logins) -> None:
        StableDiffusionCLISetupSDXL(make_config(), "")

        assert logins.tokens == []

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({"model": {"name": "x", "url": MODEL_URL}}, "Invalid version"),
            (make_config(version="sd15"), "Invalid version"),
            ({"version": "sdxl"}, "Model is required"),
        ],
    )
    def test_rejects_invalid_config(
        self, config: dict[str, Any], message: str, logins: Logins
    ) -> None:
        with pytest.raises(ValueError, match=message):
            StableDiffusionCLISetupSDXL(config, "hf-token")

        assert logins.tokens == []


class Downloads:
    """Records what download_setup_files() would have fetched."""

    def __init__(self) -> None:
        self.vaes: list[tuple[str, Path]] = []
        self.controlnets: list[tuple[str, Path]] = []
        self.files: list[tuple[str, Path]] = []


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> Downloads:
    recorded = Downloads()

    class FakeModel:
        def __init__(self, source: str, sink: list[tuple[str, Path]]) -> None:
            self.source = source
            self.sink = sink

        def save_pretrained(self, cache_path: Path, **_kwargs: object) -> None:
            self.sink.append((self.source, Path(cache_path)))

    class FakeAutoencoderKL:
        @staticmethod
        def from_single_file(
            pretrained_model_link_or_path: str, **_kwargs: object
        ) -> FakeModel:
            return FakeModel(pretrained_model_link_or_path, recorded.vaes)

    class FakeControlNetModel:
        @staticmethod
        def from_pretrained(repo_id: str, **_kwargs: object) -> FakeModel:
            return FakeModel(repo_id, recorded.controlnets)

    class FakeResponse:
        def read(self) -> bytes:
            return b"weights"

    def fake_urlopen(request: Any) -> FakeResponse:  # noqa: ANN401 - urllib Request
        recorded.files.append((request.full_url, Path()))
        return FakeResponse()

    monkeypatch.setattr(app_img.diffusers, "AutoencoderKL", FakeAutoencoderKL)
    monkeypatch.setattr(app_img.diffusers, "ControlNetModel", FakeControlNetModel)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return recorded


class TestCommonSetupDownloadSetupFiles:
    def test_downloads_nothing_extra_for_a_minimal_config(
        self, downloads: Downloads
    ) -> None:
        CommonSetup(make_config(), "").download_setup_files()

        assert downloads.vaes == []
        assert downloads.controlnets == []
        assert downloads.files == []

    def test_vae_is_saved_under_the_model_directory(self, downloads: Downloads) -> None:
        config = make_config(
            vae={"name": "sdxl-vae", "url": "https://example.com/vae.safetensors"}
        )

        CommonSetup(config, "").download_setup_files()

        # The VAE is saved next to the model, not under its own name.
        assert downloads.vaes == [
            ("https://example.com/vae.safetensors", Path(BASE_CACHE_PATH, "my-sdxl")),
        ]

    def test_every_controlnet_is_downloaded(self, downloads: Downloads) -> None:
        config = make_config(
            controlnets=[
                {"name": "canny", "repo_id": "diffusers/controlnet-canny-sdxl-1.0"},
                {"name": "depth", "repo_id": "diffusers/controlnet-depth-sdxl-1.0"},
            ],
        )

        CommonSetup(config, "").download_setup_files()

        assert downloads.controlnets == [
            (
                "diffusers/controlnet-canny-sdxl-1.0",
                Path(BASE_CACHE_PATH_CONTROLNET) / "canny",
            ),
            (
                "diffusers/controlnet-depth-sdxl-1.0",
                Path(BASE_CACHE_PATH_CONTROLNET) / "depth",
            ),
        ]

    def test_loras_and_textual_inversions_are_written_to_their_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        downloads: Downloads,
    ) -> None:
        lora_dir = tmp_path / "lora"
        embedding_dir = tmp_path / "textual_inversion"
        monkeypatch.setattr(app_img, "BASE_CACHE_PATH_LORA", str(lora_dir))
        monkeypatch.setattr(
            app_img, "BASE_CACHE_PATH_TEXTUAL_INVERSION", str(embedding_dir)
        )
        config = make_config(
            loras=[
                {
                    "name": "style.safetensors",
                    "url": "https://example.com/style.safetensors",
                }
            ],
            textual_inversions=[
                {"name": "neg.pt", "url": "https://example.com/neg.pt"}
            ],
        )

        CommonSetup(config, "").download_setup_files()

        assert [url for url, _ in downloads.files] == [
            "https://example.com/style.safetensors",
            "https://example.com/neg.pt",
        ]
        assert (lora_dir / "style.safetensors").read_bytes() == b"weights"
        assert (embedding_dir / "neg.pt").read_bytes() == b"weights"

    def test_default_directories_are_inside_the_volume(self) -> None:
        assert BASE_CACHE_PATH_LORA.endswith("/lora")
        assert BASE_CACHE_PATH_TEXTUAL_INVERSION.endswith("/textual_inversion")

    def test_non_http_url_is_rejected(self, downloads: Downloads) -> None:
        config = make_config(
            loras=[{"name": "style.safetensors", "url": "file:///etc/passwd"}]
        )

        with pytest.raises(ValueError, match=r"Only http\(s\) URLs are supported"):
            CommonSetup(config, "").download_setup_files()

        assert downloads.files == []


class TestDoubleImageSize:
    def test_doubles_both_sides(self) -> None:
        doubled = double_image_size(PIL.Image.new("RGB", (30, 20)))

        assert doubled.size == (60, 40)

    def test_converts_to_rgb(self) -> None:
        doubled = double_image_size(PIL.Image.new("RGBA", (8, 8)))

        assert doubled.mode == "RGB"
        assert doubled.size == (16, 16)

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import app_anima
import pytest
from app_anima import (
    DEFAULT_ANIMA_REPO_ID,
    MAX_AREA,
    AnimaSetup,
    resolve_size,
    split_anima_checkpoint,
    upscale_size,
)

if TYPE_CHECKING:
    from pathlib import Path

TRANSFORMER_URL = "https://huggingface.co/org/repo/blob/main/split_files/diffusion_models/anima-turbo.safetensors"


class TestSplitAnimaCheckpoint:
    def test_separates_the_llm_adapter_from_the_transformer(self) -> None:
        state_dict = {
            "net.x_embedder.proj.1.weight": 1,
            "net.blocks.0.self_attn.q_proj.weight": 2,
            "net.llm_adapter.embed.weight": 3,
            "net.llm_adapter.blocks.0.cross_attn.k_proj.weight": 4,
        }

        transformer, text_conditioner = split_anima_checkpoint(state_dict)

        # The Cosmos converter strips `net.` itself, so the transformer keeps it.
        assert transformer == {
            "net.x_embedder.proj.1.weight": 1,
            "net.blocks.0.self_attn.q_proj.weight": 2,
        }
        assert text_conditioner == {
            "embed.weight": 3,
            "blocks.0.cross_attn.k_proj.weight": 4,
        }


class TestResolveSize:
    @pytest.mark.parametrize(
        ("height", "width", "expected"),
        [
            (1024, 1024, (1024, 1024)),
            (1000, 1500, (1008, 1504)),
            (512, 512, (512, 512)),
            (1536, 1536, (1536, 1536)),
        ],
    )
    def test_rounds_to_a_multiple_of_16(
        self, height: int, width: int, expected: tuple[int, int]
    ) -> None:
        assert resolve_size(height, width) == expected

    @pytest.mark.parametrize(("height", "width"), [(256, 512), (2048, 1536)])
    def test_rejects_sizes_outside_the_supported_area(self, height: int, width: int) -> None:
        with pytest.raises(ValueError, match="outside the supported area"):
            resolve_size(height, width)


class TestUpscaleSize:
    def test_small_images_are_doubled(self) -> None:
        assert upscale_size(512, 768) == (1024, 1536)

    def test_large_images_are_capped_at_the_max_area(self) -> None:
        assert upscale_size(1024, 1024) == (1536, 1536)

    @pytest.mark.parametrize(("height", "width"), [(832, 1216), (1024, 1536), (1536, 1536)])
    def test_result_never_exceeds_the_max_area(self, height: int, width: int) -> None:
        new_height, new_width = upscale_size(height, width)

        assert new_height * new_width <= MAX_AREA
        assert new_height % 16 == 0
        assert new_width % 16 == 0


def make_config(loras: list[dict[str, Any]] | None = None, **model_overrides: object) -> dict[str, Any]:
    model: dict[str, Any] = {"name": "anima-turbo"}
    model.update(model_overrides)
    anima: dict[str, Any] = {"model": model}
    if loras is not None:
        anima["loras"] = loras
    return {"anima": anima}


class Downloads:
    """Records what AnimaSetup would have fetched and converted."""

    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []
        self.files: list[tuple[str, Path, str]] = []
        self.conversions: list[tuple[Path, Path]] = []


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Downloads:
    recorded = Downloads()

    def fake_snapshot_download(**kwargs: Any) -> str:  # noqa: ANN401 - mirrors snapshot_download
        recorded.snapshots.append(kwargs)
        return kwargs.get("local_dir", "")

    def fake_download_file(url: str, dest_dir: Path, token: str) -> Path:
        recorded.files.append((url, dest_dir, token))
        dest = tmp_path / "downloaded.safetensors"
        dest.write_bytes(b"")
        return dest

    def fake_convert(checkpoint_path: Path, model_path: Path) -> None:
        recorded.conversions.append((checkpoint_path, model_path))

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(app_anima, "download_file", fake_download_file)
    monkeypatch.setattr(app_anima, "convert_single_file", fake_convert)
    monkeypatch.setattr(app_anima, "MODEL_VOLUME_PATH", str(tmp_path / "vol"))
    return recorded


class TestAnimaSetupValidation:
    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({}, "anima is required"),
            ({"anima": {}}, "anima.model.name is required"),
            ({"anima": {"model": {}}}, "anima.model.name is required"),
            (make_config(loras=[{"name": "x"}]), "needs a url"),
        ],
    )
    def test_rejects_invalid_config(self, config: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            AnimaSetup(config, "")


class TestAnimaSetupDownloadModel:
    def test_downloads_the_default_repo_without_a_transformer_url(
        self, downloads: Downloads
    ) -> None:
        expected_auth = "hf-token"

        AnimaSetup(make_config(), expected_auth).download_model()

        assert len(downloads.snapshots) == 1
        snapshot = downloads.snapshots[0]
        assert snapshot["repo_id"] == DEFAULT_ANIMA_REPO_ID
        assert snapshot["token"] == expected_auth
        assert snapshot["local_dir"].endswith("/anima-turbo")
        assert snapshot["ignore_patterns"] == ["*.md"]
        assert downloads.files == []
        assert downloads.conversions == []

    def test_repo_id_can_be_overridden_and_empty_token_is_none(
        self, downloads: Downloads
    ) -> None:
        AnimaSetup(make_config(repo_id="me/anima"), "").download_model()

        assert downloads.snapshots[0]["repo_id"] == "me/anima"
        assert downloads.snapshots[0]["token"] is None

    def test_transformer_url_skips_repo_weights_and_converts_the_single_file(
        self, downloads: Downloads
    ) -> None:
        setup = AnimaSetup(make_config(transformer_url=TRANSFORMER_URL), "hf-token")

        setup.download_model()

        skipped = downloads.snapshots[0]["ignore_patterns"]
        assert "transformer/*.safetensors" in skipped
        assert "text_conditioner/*.safetensors" in skipped
        assert [(url, token) for url, _, token in downloads.files] == [(TRANSFORMER_URL, "hf-token")]
        assert len(downloads.conversions) == 1
        checkpoint_path, model_path = downloads.conversions[0]
        assert model_path == setup.model_path
        # The single file is removed once converted.
        assert not checkpoint_path.exists()


class TestAnimaSetupDownloadLoras:
    def test_downloads_missing_loras_into_the_model_directory(
        self, downloads: Downloads
    ) -> None:
        setup = AnimaSetup(
            make_config(
                loras=[
                    {"url": "https://example.com/a.safetensors"},
                    {"url": "https://example.com/b.safetensors", "scale": 0.5},
                ],
            ),
            "",
        )
        lora_dir = setup.model_path / "loras"
        lora_dir.mkdir(parents=True)
        (lora_dir / "a.safetensors").write_bytes(b"")

        setup.download_loras()

        assert [(url, dest) for url, dest, _ in downloads.files] == [
            ("https://example.com/b.safetensors", lora_dir),
        ]


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_once = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if self.read_once:
            return b""
        self.read_once = True
        return self.body


class TestDownloadFile:
    def test_streams_a_blob_url_as_resolve_with_the_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        requests: list[Any] = []

        def fake_urlopen(req: Any) -> FakeResponse:  # noqa: ANN401 - mirrors urlopen
            requests.append(req)
            return FakeResponse(b"weights")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        expected_auth = "hf-token"

        dest = app_anima.download_file(TRANSFORMER_URL, tmp_path / "out", expected_auth)

        assert dest == tmp_path / "out" / "anima-turbo.safetensors"
        assert dest.read_bytes() == b"weights"
        assert "/resolve/" in requests[0].full_url
        assert requests[0].get_header("Authorization") == f"Bearer {expected_auth}"

    def test_rejects_non_http_urls(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Only http"):
            app_anima.download_file("file:///etc/passwd", tmp_path, "")
